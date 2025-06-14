#!/usr/bin/env python3
"""
Blue/Green Deployment Script for Mech-Exo Risk Control

Implements zero-downtime deployment strategy:
- Blue = Current production environment
- Green = New deployment candidate
- Health checks before traffic switching
- Automatic rollback capability

Usage:
    python scripts/deploy_blue_green.py --image mech-exo:v0.5.1-rc --dry-run
    python scripts/deploy_blue_green.py --image mech-exo:v0.5.1-rc --promote
"""

import os
import sys
import time
import json
import requests
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    image: str
    namespace: str = "default"
    app_name: str = "mech-exo"
    blue_suffix: str = "blue"
    green_suffix: str = "green"
    health_timeout: int = 300  # 5 minutes
    health_interval: int = 10   # 10 seconds
    rollback_timeout: int = 60  # 1 minute


@dataclass
class HealthCheck:
    """Health check result"""
    endpoint: str
    status_code: int
    response_time: float
    healthy: bool
    error: Optional[str] = None


class BlueGreenDeployer:
    """Blue/Green deployment manager"""
    
    def __init__(self, config: DeploymentConfig, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        self.deployment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Deployment state
        self.blue_url = None
        self.green_url = None
        self.current_active = None
        self.deployment_log = []
        
        logger.info(f"🚀 Blue/Green Deployer initialized")
        logger.info(f"   Image: {config.image}")
        logger.info(f"   Namespace: {config.namespace}")
        logger.info(f"   Dry Run: {dry_run}")
        logger.info(f"   Deployment ID: {self.deployment_id}")
        
    def log_step(self, step: str, success: bool, message: str, data: Optional[Dict] = None):
        """Log deployment step"""
        entry = {
            'step': step,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        self.deployment_log.append(entry)
        
        status = "✅" if success else "❌"
        logger.info(f"{status} {step}: {message}")
        
    def run_command(self, command: List[str], timeout: int = 60) -> Tuple[bool, str, str]:
        """Run shell command with timeout"""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would run: {' '.join(command)}")
            return True, f"[DRY-RUN] Command: {' '.join(command)}", ""
            
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)
            
    def check_prerequisites(self) -> bool:
        """Check deployment prerequisites"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check kubectl availability
        success, stdout, stderr = self.run_command(['kubectl', 'version', '--client'])
        if not success:
            self.log_step("prerequisites", False, f"kubectl not available: {stderr}")
            return False
            
        # Check cluster connection
        success, stdout, stderr = self.run_command(['kubectl', 'cluster-info'])
        if not success:
            self.log_step("prerequisites", False, f"Cannot connect to cluster: {stderr}")
            return False
            
        # Check namespace exists
        success, stdout, stderr = self.run_command([
            'kubectl', 'get', 'namespace', self.config.namespace
        ])
        if not success:
            logger.warning(f"Namespace {self.config.namespace} does not exist, creating...")
            success, stdout, stderr = self.run_command([
                'kubectl', 'create', 'namespace', self.config.namespace
            ])
            if not success:
                self.log_step("prerequisites", False, f"Cannot create namespace: {stderr}")
                return False
                
        # Check image availability (if not dry run)
        if not self.dry_run:
            success, stdout, stderr = self.run_command([
                'docker', 'inspect', self.config.image
            ])
            if not success:
                logger.warning(f"Image {self.config.image} not found locally, will attempt pull")
                
        self.log_step("prerequisites", True, "All prerequisites satisfied")
        return True
        
    def detect_current_deployment(self) -> Optional[str]:
        """Detect currently active deployment (blue or green)"""
        logger.info("🔍 Detecting current active deployment...")
        
        # Check which deployment is receiving traffic via service selector
        success, stdout, stderr = self.run_command([
            'kubectl', 'get', 'service', self.config.app_name,
            '-n', self.config.namespace,
            '-o', 'jsonpath={.spec.selector.version}'
        ])
        
        if success and stdout.strip():
            active = stdout.strip()
            logger.info(f"   Current active: {active}")
            self.current_active = active
            return active
        else:
            logger.info("   No active deployment detected (first deployment)")
            self.current_active = None
            return None
            
    def determine_target_environment(self) -> str:
        """Determine target deployment environment"""
        if self.current_active == self.config.blue_suffix:
            target = self.config.green_suffix
        else:
            target = self.config.blue_suffix
            
        logger.info(f"🎯 Target environment: {target}")
        return target
        
    def deploy_green_environment(self, target: str) -> bool:
        """Deploy to target environment"""
        logger.info(f"🚀 Deploying {self.config.image} to {target} environment...")
        
        # Generate deployment manifest
        deployment_manifest = self._generate_deployment_manifest(target)
        service_manifest = self._generate_service_manifest(target)
        
        if self.dry_run:
            self.log_step(
                "deploy_green", 
                True, 
                f"[DRY-RUN] Would deploy to {target}",
                {"manifest_lines": len(deployment_manifest.split('\n'))}
            )
            return True
            
        # Apply deployment
        success = self._apply_manifest(deployment_manifest, f"deployment-{target}")
        if not success:
            return False
            
        # Apply service
        success = self._apply_manifest(service_manifest, f"service-{target}")
        if not success:
            return False
            
        # Wait for deployment to be ready
        success = self._wait_for_deployment_ready(target)
        if not success:
            return False
            
        self.log_step("deploy_green", True, f"Successfully deployed to {target}")
        return True
        
    def _generate_deployment_manifest(self, environment: str) -> str:
        """Generate Kubernetes deployment manifest"""
        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {self.config.app_name}-{environment}
  namespace: {self.config.namespace}
  labels:
    app: {self.config.app_name}
    version: {environment}
    deployment-id: {self.deployment_id}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: {self.config.app_name}
      version: {environment}
  template:
    metadata:
      labels:
        app: {self.config.app_name}
        version: {environment}
    spec:
      containers:
      - name: {self.config.app_name}
        image: {self.config.image}
        ports:
        - containerPort: 8050
          name: dashboard
        - containerPort: 8000
          name: metrics
        envFrom:
        - secretRef:
            name: mech-exo-secrets
        - configMapRef:
            name: mech-exo-config
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8050
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8050
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
"""
        return manifest.strip()
        
    def _generate_service_manifest(self, environment: str) -> str:
        """Generate Kubernetes service manifest"""
        manifest = f"""
apiVersion: v1
kind: Service
metadata:
  name: {self.config.app_name}-{environment}
  namespace: {self.config.namespace}
  labels:
    app: {self.config.app_name}
    version: {environment}
spec:
  selector:
    app: {self.config.app_name}
    version: {environment}
  ports:
  - name: dashboard
    port: 8050
    targetPort: 8050
  - name: metrics
    port: 8000
    targetPort: 8000
  type: ClusterIP
"""
        return manifest.strip()
        
    def _apply_manifest(self, manifest: str, name: str) -> bool:
        """Apply Kubernetes manifest"""
        manifest_file = f"/tmp/{name}-{self.deployment_id}.yaml"
        
        try:
            with open(manifest_file, 'w') as f:
                f.write(manifest)
                
            success, stdout, stderr = self.run_command([
                'kubectl', 'apply', '-f', manifest_file
            ])
            
            # Cleanup
            os.unlink(manifest_file)
            
            if success:
                logger.info(f"   ✅ Applied {name}")
                return True
            else:
                logger.error(f"   ❌ Failed to apply {name}: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ Error applying {name}: {str(e)}")
            return False
            
    def _wait_for_deployment_ready(self, environment: str) -> bool:
        """Wait for deployment to be ready"""
        logger.info(f"⏳ Waiting for {environment} deployment to be ready...")
        
        start_time = time.time()
        
        while time.time() - start_time < self.config.health_timeout:
            success, stdout, stderr = self.run_command([
                'kubectl', 'get', 'deployment', f"{self.config.app_name}-{environment}",
                '-n', self.config.namespace,
                '-o', 'jsonpath={.status.readyReplicas}'
            ])
            
            if success and stdout.strip() == "2":  # 2 replicas ready
                logger.info(f"   ✅ {environment} deployment ready")
                return True
                
            time.sleep(10)
            logger.info(f"   ⏳ Still waiting... ({int(time.time() - start_time)}s)")
            
        logger.error(f"   ❌ {environment} deployment not ready after {self.config.health_timeout}s")
        return False
        
    def perform_health_checks(self, environment: str) -> List[HealthCheck]:
        """Perform comprehensive health checks"""
        logger.info(f"🏥 Performing health checks on {environment}...")
        
        # Get service URL
        service_url = self._get_service_url(environment)
        if not service_url:
            return [HealthCheck("/healthz", 0, 0, False, "Service URL not available")]
            
        endpoints = [
            "/healthz",
            "/riskz", 
            "/api/risk/current",
            "/metrics"
        ]
        
        health_checks = []
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = requests.get(
                    f"{service_url}{endpoint}",
                    timeout=10
                )
                response_time = time.time() - start_time
                
                # Check response criteria
                healthy = (
                    response.status_code == 200 and
                    response_time < 2.0  # 2 second threshold
                )
                
                if endpoint == "/riskz" and healthy:
                    # Additional validation for risk endpoint
                    try:
                        data = response.json()
                        required_fields = ['var_95', 'var_99', 'today_pnl_pct']
                        healthy = all(field in data for field in required_fields)
                    except:
                        healthy = False
                
                check = HealthCheck(
                    endpoint=endpoint,
                    status_code=response.status_code,
                    response_time=response_time,
                    healthy=healthy
                )
                
                health_checks.append(check)
                
                status = "✅" if healthy else "❌"
                logger.info(f"   {status} {endpoint}: {response.status_code} ({response_time:.2f}s)")
                
            except Exception as e:
                check = HealthCheck(
                    endpoint=endpoint,
                    status_code=0,
                    response_time=0,
                    healthy=False,
                    error=str(e)
                )
                health_checks.append(check)
                logger.warning(f"   ❌ {endpoint}: Error - {str(e)}")
                
        return health_checks
        
    def _get_service_url(self, environment: str) -> Optional[str]:
        """Get service URL for environment"""
        if self.dry_run:
            return f"http://mech-exo-{environment}.{self.config.namespace}.svc.cluster.local:8050"
            
        # In real deployment, this would use kubectl port-forward or ingress
        success, stdout, stderr = self.run_command([
            'kubectl', 'get', 'service', f"{self.config.app_name}-{environment}",
            '-n', self.config.namespace,
            '-o', 'jsonpath={.spec.clusterIP}'
        ])
        
        if success and stdout.strip():
            return f"http://{stdout.strip()}:8050"
        else:
            return None
            
    def switch_traffic(self, target_environment: str) -> bool:
        """Switch traffic to target environment"""
        logger.info(f"🔄 Switching traffic to {target_environment}...")
        
        # Update main service selector to point to target environment
        success, stdout, stderr = self.run_command([
            'kubectl', 'patch', 'service', self.config.app_name,
            '-n', self.config.namespace,
            '-p', f'{{"spec":{{"selector":{{"version":"{target_environment}"}}}}}}'
        ])
        
        if success:
            self.log_step("switch_traffic", True, f"Traffic switched to {target_environment}")
            return True
        else:
            self.log_step("switch_traffic", False, f"Failed to switch traffic: {stderr}")
            return False
            
    def rollback(self) -> bool:
        """Rollback to previous environment"""
        if not self.current_active:
            logger.error("❌ No previous environment to rollback to")
            return False
            
        logger.info(f"🔄 Rolling back to {self.current_active}...")
        
        success = self.switch_traffic(self.current_active)
        
        if success:
            self.log_step("rollback", True, f"Rolled back to {self.current_active}")
        else:
            self.log_step("rollback", False, "Rollback failed")
            
        return success
        
    def cleanup_old_environment(self, environment: str) -> bool:
        """Clean up old environment"""
        logger.info(f"🧹 Cleaning up {environment} environment...")
        
        # Delete deployment
        success1, _, stderr1 = self.run_command([
            'kubectl', 'delete', 'deployment', f"{self.config.app_name}-{environment}",
            '-n', self.config.namespace,
            '--ignore-not-found'
        ])
        
        # Delete service
        success2, _, stderr2 = self.run_command([
            'kubectl', 'delete', 'service', f"{self.config.app_name}-{environment}",
            '-n', self.config.namespace,
            '--ignore-not-found'
        ])
        
        success = success1 and success2
        
        if success:
            self.log_step("cleanup", True, f"Cleaned up {environment}")
        else:
            self.log_step("cleanup", False, f"Cleanup failed: {stderr1} {stderr2}")
            
        return success
        
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate deployment report"""
        end_time = datetime.now()
        
        # Calculate deployment duration
        start_time = datetime.fromisoformat(self.deployment_log[0]['timestamp'])
        duration = (end_time - start_time).total_seconds()
        
        # Count success/failure
        total_steps = len(self.deployment_log)
        successful_steps = sum(1 for step in self.deployment_log if step['success'])
        
        report = {
            'deployment_id': self.deployment_id,
            'image': self.config.image,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'success_rate': (successful_steps / total_steps * 100) if total_steps > 0 else 0,
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'current_active': self.current_active,
            'dry_run': self.dry_run,
            'steps': self.deployment_log
        }
        
        return report
        
    def deploy(self, promote: bool = False) -> bool:
        """Execute complete deployment process"""
        logger.info("🎯 Starting Blue/Green Deployment")
        logger.info("=" * 50)
        
        try:
            # Step 1: Prerequisites
            if not self.check_prerequisites():
                return False
                
            # Step 2: Detect current deployment
            self.detect_current_deployment()
            
            # Step 3: Determine target environment
            target = self.determine_target_environment()
            
            # Step 4: Deploy to target (green) environment
            if not self.deploy_green_environment(target):
                logger.error("❌ Green deployment failed")
                return False
                
            # Step 5: Health checks
            health_checks = self.perform_health_checks(target)
            healthy_checks = [c for c in health_checks if c.healthy]
            
            health_success = len(healthy_checks) >= 3  # At least 3/4 endpoints healthy
            
            if not health_success:
                logger.error(f"❌ Health checks failed: {len(healthy_checks)}/{len(health_checks)} passed")
                if not self.dry_run:
                    logger.info("🧹 Cleaning up failed deployment...")
                    self.cleanup_old_environment(target)
                return False
                
            self.log_step("health_checks", True, f"Health checks passed: {len(healthy_checks)}/{len(health_checks)}")
            
            # Step 6: Traffic switch (only if promote=True)
            if promote:
                if not self.switch_traffic(target):
                    logger.error("❌ Traffic switch failed, attempting rollback...")
                    if self.current_active:
                        self.rollback()
                    return False
                    
                # Step 7: Cleanup old environment (after successful switch)
                if self.current_active:
                    time.sleep(30)  # Wait a bit before cleanup
                    self.cleanup_old_environment(self.current_active)
                    
                logger.info("✅ Deployment completed successfully and promoted to production!")
            else:
                logger.info("✅ Deployment completed successfully (use --promote to switch traffic)")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed with exception: {str(e)}")
            self.log_step("deployment", False, f"Exception: {str(e)}")
            return False
        finally:
            # Generate report
            report = self.generate_deployment_report()
            report_file = f"deployment_report_{self.deployment_id}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info("=" * 50)
            logger.info(f"📊 Deployment Report: {report_file}")
            logger.info(f"   Duration: {report['duration_seconds']:.1f}s")
            logger.info(f"   Success Rate: {report['success_rate']:.1f}%")
            logger.info(f"   Steps: {report['successful_steps']}/{report['total_steps']}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Blue/Green Deployment for Mech-Exo')
    parser.add_argument('--image', required=True,
                       help='Docker image to deploy (e.g., mech-exo:v0.5.1-rc)')
    parser.add_argument('--namespace', default='default',
                       help='Kubernetes namespace (default: default)')
    parser.add_argument('--promote', action='store_true',
                       help='Promote deployment to production (switch traffic)')
    parser.add_argument('--cleanup', choices=['blue', 'green'],
                       help='Cleanup specified environment (blue or green)')
    parser.add_argument('--rollback', action='store_true',
                       help='Rollback to previous deployment')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (no actual deployment)')
    parser.add_argument('--health-timeout', type=int, default=300,
                       help='Health check timeout in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Create deployment config
    config = DeploymentConfig(
        image=args.image,
        namespace=args.namespace,
        health_timeout=args.health_timeout
    )
    
    # Create deployer
    deployer = BlueGreenDeployer(config, dry_run=args.dry_run)
    
    # Handle different operations
    if args.cleanup:
        logger.info(f"🧹 Cleaning up {args.cleanup} environment...")
        success = deployer.cleanup_old_environment(args.cleanup)
    elif args.rollback:
        logger.info("🔄 Rolling back deployment...")
        success = deployer.rollback()
    else:
        # Normal deployment
        success = deployer.deploy(promote=args.promote)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()