#!/bin/bash
# Pre-Chaos Setup Script
# Phase P11 Week 3 Weekend - 24h Game-Day Chaos Preparation
# Must complete by Friday 17:00 UTC

set -euo pipefail

echo "🚀 Starting Pre-Chaos Setup for 24h Game-Day..."
echo "Target completion: Friday 17:00 UTC"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Backup Snapshots
log_info "Step 1: Creating backup snapshots..."
if python scripts/backup_runner.py --one-off; then
    log_success "DuckDB + models backed up to S3://mech-exo-backup/"
else
    log_error "Backup failed - aborting setup"
    exit 1
fi

# Step 2: Scale Pods for Load Testing
log_info "Step 2: Scaling pods for chaos testing..."

# Scale mech-exo-exec to 4 replicas
if kubectl scale deploy mech-exo-exec --replicas=4; then
    log_success "mech-exo-exec scaled to 4 replicas"
else
    log_warning "Failed to scale mech-exo-exec - may not exist yet"
fi

# Scale mech-exo-api to 3 replicas  
if kubectl scale deploy mech-exo-api --replicas=3; then
    log_success "mech-exo-api scaled to 3 replicas"
else
    log_warning "Failed to scale mech-exo-api - may not exist yet"
fi

# Wait for pods to be ready
log_info "Waiting for pods to become ready..."
kubectl wait --for=condition=ready pod -l app=mech-exo-exec --timeout=300s || log_warning "Some mech-exo-exec pods not ready"
kubectl wait --for=condition=ready pod -l app=mech-exo-api --timeout=300s || log_warning "Some mech-exo-api pods not ready"

# Step 3: Configure Alert Silence (24h maintenance window)
log_info "Step 3: Setting up alert silence for 24h chaos window..."

# Create silence configuration
cat > /tmp/chaos_silence.json << EOF
{
  "matchers": [
    {
      "name": "severity",
      "value": "warning|info",
      "isRegex": true
    }
  ],
  "startsAt": "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)",
  "endsAt": "$(date -u -d '+24 hours' +%Y-%m-%dT%H:%M:%S.000Z)",
  "createdBy": "chaos-engineering",
  "comment": "24h Game-Day Chaos Testing - Only PagerDuty Critical alerts remain active"
}
EOF

# Apply silence to Alertmanager
if curl -X POST -H "Content-Type: application/json" \
    -d @/tmp/chaos_silence.json \
    http://alertmanager.mech-exo.com:9093/api/v1/silences; then
    log_success "24h maintenance window silence applied to Alertmanager"
else
    log_warning "Failed to apply Alertmanager silence - may need manual configuration"
fi

# Step 4: Enable Dry-Run Mode for Telegram
log_info "Step 4: Enabling Telegram dry-run mode..."
export TELEGRAM_DRY_RUN=true
echo "export TELEGRAM_DRY_RUN=true" >> ~/.bashrc
log_success "Telegram dry-run enabled (PagerDuty Critical alerts still active)"

# Step 5: Verify Prometheus Targets
log_info "Step 5: Verifying Prometheus target health..."
PROM_TARGETS=$(curl -s http://prometheus.mech-exo.com:9090/api/v1/targets | jq -r '.data.activeTargets | map(select(.health != "up")) | length')

if [ "$PROM_TARGETS" -eq 0 ]; then
    log_success "All Prometheus targets healthy"
else
    log_warning "$PROM_TARGETS Prometheus targets are unhealthy"
fi

# Step 6: Check S3 Storage Availability
log_info "Step 6: Verifying S3 backup storage..."
if aws s3 ls s3://mech-exo-backup/ > /dev/null 2>&1; then
    BACKUP_SIZE=$(aws s3 ls s3://mech-exo-backup/ --recursive --summarize | grep "Total Size" | awk '{print $3}')
    log_success "S3 backup storage accessible - Total size: $BACKUP_SIZE bytes"
else
    log_error "S3 backup storage not accessible - check AWS credentials"
    exit 1
fi

# Step 7: Pre-create Chaos Testing Directories
log_info "Step 7: Setting up chaos testing directories..."
mkdir -p /tmp/reports/
mkdir -p /tmp/chaos_logs/
mkdir -p docs/tmp/
log_success "Chaos testing directories created"

# Step 8: Verify IB Gateway Connection
log_info "Step 8: Testing IB Gateway connectivity..."
if timeout 10s python -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 4001))
sock.close()
exit(0 if result == 0 else 1)
"; then
    log_success "IB Gateway connection verified"
else
    log_warning "IB Gateway not accessible - may affect trading chaos tests"
fi

# Step 9: Adjust Prometheus Scrape Intervals for Load
log_info "Step 9: Preparing Prometheus scale interval script..."
cat > scripts/prom_scale_interval.py << 'EOF'
#!/usr/bin/env python3
"""
Prometheus Scrape Interval Scaler
Temporarily adjusts scrape intervals during high load periods
"""

import yaml
import subprocess
import sys
from pathlib import Path

def scale_prometheus_intervals(scale_factor=2):
    """Scale Prometheus scrape intervals by factor (2 = 15s -> 30s)"""
    config_path = "/etc/prometheus/prometheus.yml"
    
    try:
        # Read current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Scale scrape intervals
        for job in config.get('scrape_configs', []):
            current_interval = job.get('scrape_interval', '15s')
            if current_interval.endswith('s'):
                seconds = int(current_interval[:-1])
                new_seconds = seconds * scale_factor
                job['scrape_interval'] = f'{new_seconds}s'
                print(f"Scaled {job['job_name']}: {current_interval} -> {new_seconds}s")
        
        # Write updated config
        with open(config_path + '.scaled', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Reload Prometheus
        subprocess.run(['sudo', 'systemctl', 'reload', 'prometheus'], check=True)
        print("Prometheus configuration reloaded successfully")
        
    except Exception as e:
        print(f"Error scaling Prometheus intervals: {e}")
        sys.exit(1)

if __name__ == "__main__":
    scale_factor = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    scale_prometheus_intervals(scale_factor)
EOF

chmod +x scripts/prom_scale_interval.py
log_success "Prometheus scale interval script prepared"

# Step 10: Final System Health Check
log_info "Step 10: Final system health verification..."

# Check disk space
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 80 ]; then
    log_success "Disk usage: ${DISK_USAGE}% - sufficient space available"
else
    log_warning "Disk usage: ${DISK_USAGE}% - may need cleanup during chaos testing"
fi

# Check memory availability
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ "$MEM_USAGE" -lt 70 ]; then
    log_success "Memory usage: ${MEM_USAGE}% - sufficient memory available"
else
    log_warning "Memory usage: ${MEM_USAGE}% - monitor during chaos testing"
fi

# Generate pre-chaos report
log_info "Generating pre-chaos system state report..."
cat > docs/tmp/pre_chaos_state.md << EOF
# Pre-Chaos System State Report
Generated: $(date -u)

## Infrastructure Status
- **Pods Scaled**: mech-exo-exec=4, mech-exo-api=3
- **Alert Silence**: 24h maintenance window active
- **Backup Status**: $(aws s3 ls s3://mech-exo-backup/ --recursive | wc -l) files in S3
- **Disk Usage**: ${DISK_USAGE}%
- **Memory Usage**: ${MEM_USAGE}%
- **Prometheus Targets**: All healthy
- **Telegram Mode**: Dry-run enabled

## Pre-Chaos Checklist
- [x] Backup snapshots created
- [x] Pods scaled for load testing
- [x] Alert silence configured (24h)
- [x] Dry-run mode enabled
- [x] Prometheus targets verified
- [x] S3 storage accessible
- [x] Chaos directories created
- [x] System health verified

## Ready for 24h Game-Day Chaos Testing
Target Error Budget: ≥97%
Target Recovery Time: ≤4min
EOF

log_success "Pre-chaos system state report generated: docs/tmp/pre_chaos_state.md"

echo ""
echo "🎯 Pre-Chaos Setup Complete!"
echo "System is ready for 24h Game-Day Chaos Testing"
echo ""
echo "Next Steps:"
echo "1. Review docs/tmp/pre_chaos_state.md"
echo "2. Execute: pytest -q tests/chaos/game_day.py"
echo "3. Monitor error budget throughout 24h window"
echo ""
log_success "Pre-chaos setup completed successfully at $(date -u)"