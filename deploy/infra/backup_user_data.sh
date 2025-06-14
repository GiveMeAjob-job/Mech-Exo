#!/bin/bash
# User data script for Mech-Exo backup instance
# Installs and configures backup services

set -e

# Variables from Terraform
S3_BUCKET="${s3_bucket}"
REGION="${region}"

# Update system
apt-get update
apt-get upgrade -y

# Install required packages
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    docker.io \
    docker-compose \
    awscli \
    cron \
    htop \
    jq \
    curl \
    wget \
    unzip

# Configure Docker
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# Create backup directories
mkdir -p /opt/mech-exo-backup
mkdir -p /data/duckdb
mkdir -p /data/models
mkdir -p /var/log/mech-exo

# Mount EBS volume for data storage
# Wait for volume to be attached
while [ ! -e /dev/xvdf ]; do
    echo "Waiting for EBS volume to be attached..."
    sleep 5
done

# Format volume if not already formatted
if ! blkid /dev/xvdf; then
    mkfs.ext4 /dev/xvdf
fi

# Mount volume
mount /dev/xvdf /data
echo '/dev/xvdf /data ext4 defaults,nofail 0 2' >> /etc/fstab

# Set permissions
chown -R ubuntu:ubuntu /data
chown -R ubuntu:ubuntu /opt/mech-exo-backup

# Create backup configuration
cat > /opt/mech-exo-backup/config.json << EOF
{
    "s3_bucket": "$S3_BUCKET",
    "region": "$REGION",
    "backup_paths": {
        "duckdb": "/data/duckdb",
        "models": "/data/models",
        "config": "/opt/mech-exo-backup/config"
    },
    "retention_days": 30,
    "backup_interval_hours": 6,
    "health_check_port": 8050,
    "metrics_port": 8000
}
EOF

# Create backup script
cat > /opt/mech-exo-backup/backup_runner.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Backup Runner for Mech-Exo Disaster Recovery
Handles DuckDB snapshots and model file backups to S3
"""

import os
import sys
import json
import boto3
import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [BACKUP] %(message)s',
    handlers=[
        logging.FileHandler('/var/log/mech-exo/backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupRunner:
    def __init__(self, config_path='/opt/mech-exo-backup/config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.s3_client = boto3.client('s3', region_name=self.config['region'])
        self.bucket = self.config['s3_bucket']
        
    def create_duckdb_snapshot(self, source_path, snapshot_name):
        """Create a DuckDB snapshot"""
        try:
            snapshot_path = Path(f"/tmp/{snapshot_name}")
            
            # Copy DuckDB file
            if Path(source_path).exists():
                shutil.copy2(source_path, snapshot_path)
                
                # Calculate checksum
                with open(snapshot_path, 'rb') as f:
                    checksum = hashlib.md5(f.read()).hexdigest()
                
                logger.info(f"Created DuckDB snapshot: {snapshot_path} (checksum: {checksum})")
                return str(snapshot_path), checksum
            else:
                logger.warning(f"Source DuckDB file not found: {source_path}")
                return None, None
                
        except Exception as e:
            logger.error(f"Failed to create DuckDB snapshot: {str(e)}")
            return None, None
    
    def upload_to_s3(self, local_path, s3_key, metadata=None):
        """Upload file to S3"""
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
    
    def backup_duckdb(self):
        """Backup DuckDB files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        duckdb_path = Path(self.config['backup_paths']['duckdb'])
        
        for db_file in duckdb_path.glob('*.db'):
            snapshot_name = f"duckdb_{db_file.stem}_{timestamp}.db"
            snapshot_path, checksum = self.create_duckdb_snapshot(str(db_file), snapshot_name)
            
            if snapshot_path:
                s3_key = f"duckdb/{datetime.now().year}/{datetime.now().month:02d}/{snapshot_name}"
                
                metadata = {
                    'backup_timestamp': timestamp,
                    'source_file': str(db_file),
                    'checksum': checksum,
                    'backup_type': 'duckdb_snapshot'
                }
                
                success = self.upload_to_s3(snapshot_path, s3_key, metadata)
                
                # Cleanup local snapshot
                Path(snapshot_path).unlink(missing_ok=True)
                
                if success:
                    logger.info(f"Successfully backed up {db_file.name}")
                else:
                    logger.error(f"Failed to backup {db_file.name}")
    
    def backup_models(self):
        """Backup ML model files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        models_path = Path(self.config['backup_paths']['models'])
        
        if not models_path.exists():
            logger.warning("Models directory does not exist")
            return
        
        for model_file in models_path.rglob('*'):
            if model_file.is_file():
                relative_path = model_file.relative_to(models_path)
                s3_key = f"models/{timestamp}/{relative_path}"
                
                metadata = {
                    'backup_timestamp': timestamp,
                    'model_type': model_file.suffix,
                    'backup_type': 'model_file'
                }
                
                success = self.upload_to_s3(str(model_file), s3_key, metadata)
                
                if success:
                    logger.info(f"Successfully backed up model: {relative_path}")
                else:
                    logger.error(f"Failed to backup model: {relative_path}")
    
    def cleanup_old_backups(self):
        """Remove old backups from S3"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config['retention_days'])
            
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
    
    def run_backup(self):
        """Run complete backup process"""
        logger.info("Starting backup process...")
        
        try:
            # Backup DuckDB files
            self.backup_duckdb()
            
            # Backup model files
            self.backup_models()
            
            # Cleanup old backups
            self.cleanup_old_backups()
            
            logger.info("Backup process completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Backup process failed: {str(e)}")
            return False

def main():
    backup_runner = BackupRunner()
    success = backup_runner.run_backup()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Make backup script executable
chmod +x /opt/mech-exo-backup/backup_runner.py

# Create health check service
cat > /opt/mech-exo-backup/health_server.py << 'PYTHON_HEALTH'
#!/usr/bin/env python3
"""
Health check server for backup instance
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import os
from datetime import datetime

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/healthz':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Check backup process status
            health_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "region": os.environ.get('AWS_DEFAULT_REGION', 'unknown'),
                "instance_type": "backup",
                "last_backup": self.get_last_backup_time(),
                "disk_usage": self.get_disk_usage(),
                "services": {
                    "backup_runner": self.check_backup_runner(),
                    "s3_access": self.check_s3_access()
                }
            }
            
            self.wfile.write(json.dumps(health_data, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def get_last_backup_time(self):
        try:
            result = subprocess.run(['find', '/var/log/mech-exo', '-name', 'backup.log', '-exec', 'tail', '-1', '{}', ';'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and 'completed successfully' in result.stdout:
                return "recent"
            return "unknown"
        except:
            return "error"
    
    def get_disk_usage(self):
        try:
            result = subprocess.run(['df', '-h', '/data'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return lines[1].split()[4]  # Usage percentage
            return "unknown"
        except:
            return "error"
    
    def check_backup_runner(self):
        return os.path.exists('/opt/mech-exo-backup/backup_runner.py')
    
    def check_s3_access(self):
        try:
            result = subprocess.run(['aws', 's3', 'ls'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8050), HealthHandler)
    print("Health server running on port 8050")
    server.serve_forever()
PYTHON_HEALTH

chmod +x /opt/mech-exo-backup/health_server.py

# Create systemd service for health check
cat > /etc/systemd/system/mech-exo-backup-health.service << EOF
[Unit]
Description=Mech-Exo Backup Health Check Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/mech-exo-backup
ExecStart=/usr/bin/python3 /opt/mech-exo-backup/health_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start health service
systemctl enable mech-exo-backup-health
systemctl start mech-exo-backup-health

# Create cron job for backup (every 6 hours)
cat > /tmp/backup_cron << EOF
0 */6 * * * /usr/bin/python3 /opt/mech-exo-backup/backup_runner.py >> /var/log/mech-exo/backup_cron.log 2>&1
EOF

crontab -u ubuntu /tmp/backup_cron
rm /tmp/backup_cron

# Create log rotation
cat > /etc/logrotate.d/mech-exo-backup << EOF
/var/log/mech-exo/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
}
EOF

# Set up AWS credentials from instance profile (already configured)
echo "export AWS_DEFAULT_REGION=$REGION" >> /home/ubuntu/.bashrc

# Final setup
chown -R ubuntu:ubuntu /opt/mech-exo-backup
chown -R ubuntu:ubuntu /var/log/mech-exo

logger "Mech-Exo backup instance setup completed"
echo "Backup instance setup completed at $(date)" > /opt/mech-exo-backup/setup_complete.txt