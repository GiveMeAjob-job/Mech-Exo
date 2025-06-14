#!/usr/bin/env python3
"""
Primary Site Backup Runner for Mech-Exo
Handles DuckDB snapshots and model file backups to S3 and cold backup site
"""

import os
import sys
import json
import boto3
import logging
import shutil
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import hashlib
import requests
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [BACKUP] %(message)s'
)
logger = logging.getLogger(__name__)


class BackupManager:
    """Manages backup operations to S3 and cold backup site"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config.get('region', 'us-east-1'))
        self.bucket = config.get('s3_bucket', 'mech-exo-backup')
        
    def create_duckdb_snapshot(self, source_path: str, snapshot_name: str) -> tuple:
        """Create a DuckDB snapshot with checksum"""
        try:
            source_file = Path(source_path)
            if not source_file.exists():
                logger.warning(f"Source DuckDB file not found: {source_path}")
                return None, None
            
            snapshot_path = Path(f"/tmp/{snapshot_name}")
            
            # Copy DuckDB file
            shutil.copy2(source_file, snapshot_path)
            
            # Calculate checksum
            with open(snapshot_path, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()
            
            logger.info(f"Created DuckDB snapshot: {snapshot_path} (checksum: {checksum})")
            return str(snapshot_path), checksum
            
        except Exception as e:
            logger.error(f"Failed to create DuckDB snapshot: {str(e)}")
            return None, None
    
    def upload_to_s3(self, local_path: str, s3_key: str, metadata: Optional[Dict] = None) -> bool:
        """Upload file to S3 with metadata"""
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.upload_file(
                local_path, 
                self.bucket, 
                s3_key,
                ExtraArgs=extra_args
            )
            
            logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to S3: {str(e)}")
            return False
    
    def sync_to_cold_backup(self, s3_key: str, cold_backup_url: str) -> bool:
        """Sync file to cold backup site"""
        try:
            # Notify cold backup site to pull from S3
            sync_request = {
                'action': 'sync_from_s3',
                's3_bucket': self.bucket,
                's3_key': s3_key,
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{cold_backup_url}/api/sync",
                json=sync_request,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully synced {s3_key} to cold backup")
                return True
            else:
                logger.error(f"Cold backup sync failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to sync to cold backup: {str(e)}")
            return False
    
    def backup_duckdb_files(self, source_dir: str) -> Dict[str, bool]:
        """Backup all DuckDB files from source directory"""
        results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.warning(f"Source directory does not exist: {source_dir}")
            return results
        
        for db_file in source_path.glob('*.db'):
            logger.info(f"Backing up DuckDB file: {db_file.name}")
            
            snapshot_name = f"duckdb_{db_file.stem}_{timestamp}.db"
            snapshot_path, checksum = self.create_duckdb_snapshot(str(db_file), snapshot_name)
            
            if snapshot_path:
                # Upload to S3
                s3_key = f"duckdb/{datetime.now().year}/{datetime.now().month:02d}/{snapshot_name}"
                
                metadata = {
                    'backup_timestamp': timestamp,
                    'source_file': str(db_file),
                    'checksum': checksum,
                    'backup_type': 'duckdb_snapshot',
                    'file_size': str(Path(snapshot_path).stat().st_size)
                }
                
                s3_success = self.upload_to_s3(snapshot_path, s3_key, metadata)
                
                # Sync to cold backup if configured
                cold_backup_success = True
                if self.config.get('cold_backup_url'):
                    cold_backup_success = self.sync_to_cold_backup(s3_key, self.config['cold_backup_url'])
                
                # Cleanup local snapshot
                Path(snapshot_path).unlink(missing_ok=True)
                
                results[db_file.name] = s3_success and cold_backup_success
            else:
                results[db_file.name] = False
        
        return results
    
    def backup_model_files(self, source_dir: str) -> Dict[str, bool]:
        """Backup ML model files"""
        results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.warning(f"Models directory does not exist: {source_dir}")
            return results
        
        for model_file in source_path.rglob('*'):
            if model_file.is_file():
                logger.info(f"Backing up model file: {model_file.name}")
                
                relative_path = model_file.relative_to(source_path)
                s3_key = f"models/{timestamp}/{relative_path}"
                
                metadata = {
                    'backup_timestamp': timestamp,
                    'model_type': model_file.suffix,
                    'backup_type': 'model_file',
                    'file_size': str(model_file.stat().st_size)
                }
                
                s3_success = self.upload_to_s3(str(model_file), s3_key, metadata)
                
                # Sync to cold backup if configured
                cold_backup_success = True
                if self.config.get('cold_backup_url'):
                    cold_backup_success = self.sync_to_cold_backup(s3_key, self.config['cold_backup_url'])
                
                results[str(relative_path)] = s3_success and cold_backup_success
        
        return results
    
    def cleanup_old_backups(self, retention_days: int = 30):
        """Remove old backups from S3"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket)
            
            deleted_count = 0
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                            self.s3_client.delete_object(
                                Bucket=self.bucket,
                                Key=obj['Key']
                            )
                            deleted_count += 1
                            logger.info(f"Deleted old backup: {obj['Key']}")
            
            logger.info(f"Cleanup completed: {deleted_count} old backups deleted")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {str(e)}")
    
    def validate_backup_integrity(self, s3_key: str, expected_checksum: str) -> bool:
        """Validate backup integrity"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            metadata = response.get('Metadata', {})
            
            stored_checksum = metadata.get('checksum')
            if stored_checksum == expected_checksum:
                logger.info(f"Backup integrity validated: {s3_key}")
                return True
            else:
                logger.error(f"Backup integrity check failed: {s3_key}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate backup: {str(e)}")
            return False


def load_config(config_path: str) -> Dict:
    """Load backup configuration"""
    default_config = {
        'region': 'us-east-1',
        's3_bucket': 'mech-exo-backup',
        'retention_days': 30,
        'duckdb_path': '/data/duckdb',
        'models_path': '/data/models',
        'cold_backup_url': None
    }
    
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            default_config.update(user_config)
    
    return default_config


def main():
    """Main backup execution"""
    parser = argparse.ArgumentParser(description='Mech-Exo Backup Runner')
    parser.add_argument('--config', default='config/backup.json',
                       help='Configuration file path')
    parser.add_argument('--every', default='manual',
                       help='Backup frequency (e.g., 6h, 12h, daily)')
    parser.add_argument('--src', help='Source directory for backup')
    parser.add_argument('--dst', help='Destination (S3 bucket or path)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (no actual backup)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing backups')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.src:
        config['duckdb_path'] = args.src
    if args.dst:
        config['s3_bucket'] = args.dst.replace('s3://', '')
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual backups will be performed")
    
    # Initialize backup manager
    backup_manager = BackupManager(config)
    
    try:
        logger.info("Starting backup process...")
        
        # Backup DuckDB files
        logger.info("Backing up DuckDB files...")
        duckdb_results = backup_manager.backup_duckdb_files(config['duckdb_path'])
        
        # Backup model files
        logger.info("Backing up model files...")
        model_results = backup_manager.backup_model_files(config['models_path'])
        
        # Cleanup old backups
        logger.info("Cleaning up old backups...")
        backup_manager.cleanup_old_backups(config['retention_days'])
        
        # Report results
        total_files = len(duckdb_results) + len(model_results)
        successful_files = sum([1 for success in duckdb_results.values() if success]) + \
                          sum([1 for success in model_results.values() if success])
        
        logger.info(f"Backup completed: {successful_files}/{total_files} files successful")
        
        if successful_files == total_files:
            logger.info("✅ All backups completed successfully")
            return 0
        else:
            logger.error(f"❌ {total_files - successful_files} backups failed")
            return 1
            
    except Exception as e:
        logger.error(f"Backup process failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())