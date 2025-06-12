# Prefect Deployment Guide: Canary Performance Flow

This guide covers deploying and managing the **Daily Canary Performance Tracker** flow using Prefect.

## 📋 Overview

The canary performance flow (`dags/canary_perf_flow.py`) runs daily at **23:30 UTC** to:
- Track canary vs base allocation performance
- Calculate rolling 30-day Sharpe ratios
- Execute auto-disable logic with hysteresis protection
- Send Telegram alerts for auto-disable events
- Update health endpoint cache

## 🚀 Deployment Steps

### 1. Prerequisites

```bash
# Install Prefect and dependencies
pip install prefect>=2.14.0
pip install prefect-dask  # Optional: for distributed execution

# Set Prefect API URL (if using Prefect Cloud)
prefect config set PREFECT_API_URL="https://api.prefect.cloud/api/accounts/[ACCOUNT_ID]/workspaces/[WORKSPACE_ID]"

# Or for local Prefect server
prefect config set PREFECT_API_URL="http://localhost:4200/api"
```

### 2. Environment Setup

```bash
# Required environment variables
export PYTHONPATH="/path/to/mech_exo"
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_telegram_chat_id"

# Optional: dry-run mode for testing
export TELEGRAM_DRY_RUN="false"  # Set to "true" for testing
```

### 3. Deploy the Flow

#### Option A: Command Line Deployment

```bash
# Navigate to project directory
cd /path/to/mech_exo

# Deploy flow with schedule
prefect deployment create dags/canary_perf_flow.py:canary_performance_flow \
    --name "daily-canary-tracker" \
    --description "Daily canary vs base performance tracking with auto-disable" \
    --tag "production" \
    --tag "canary" \
    --tag "performance" \
    --cron "30 23 * * *" \
    --timezone "UTC"
```

#### Option B: Python Deployment Script

```python
# deploy_canary_flow.py
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from dags.canary_perf_flow import canary_performance_flow

deployment = Deployment.build_from_flow(
    flow=canary_performance_flow,
    name="daily-canary-tracker",
    description="Daily canary vs base performance tracking with auto-disable",
    tags=["production", "canary", "performance"],
    schedule=CronSchedule(cron="30 23 * * *", timezone="UTC"),
    work_pool_name="default-agent-pool",
    parameters={
        "window_days": 30  # 30-day rolling Sharpe window
    }
)

if __name__ == "__main__":
    deployment.apply()
    print("✅ Canary performance flow deployed successfully")
```

### 4. Start Prefect Agent

```bash
# Start agent to execute scheduled flows
prefect agent start --pool "default-agent-pool"

# Or with specific tags
prefect agent start --pool "default-agent-pool" --tag "canary"
```

## 🔧 Configuration Management

### Flow Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_date` | `None` | Date to process (YYYY-MM-DD), defaults to today |
| `window_days` | `30` | Rolling window for Sharpe calculation |

### Deployment Configuration

```yaml
# deployment.yaml
name: daily-canary-tracker
description: Daily canary vs base performance tracking
schedule:
  cron: "30 23 * * *"
  timezone: UTC
tags:
  - production
  - canary
  - performance
parameters:
  window_days: 30
work_pool_name: default-agent-pool
```

## 🏥 Monitoring & Health Checks

### Flow Run Monitoring

```bash
# View recent flow runs
prefect flow-run ls --flow-name "Daily Canary Performance Tracker" --limit 10

# Get specific flow run details
prefect flow-run inspect <flow-run-id>

# View logs for failed runs
prefect flow-run logs <flow-run-id>
```

### Health Check Script

```python
#!/usr/bin/env python3
"""Check canary flow health and recent performance"""

from prefect.client.orchestration import get_client
from datetime import datetime, timedelta
import asyncio

async def check_flow_health():
    async with get_client() as client:
        # Get recent flow runs
        flow_runs = await client.read_flow_runs(
            flow_filter={"name": {"any_": ["Daily Canary Performance Tracker"]}},
            limit=5,
            sort="START_TIME_DESC"
        )
        
        print(f"📊 Recent Canary Flow Runs ({len(flow_runs)} found)")
        print("-" * 50)
        
        for run in flow_runs:
            start_time = run.start_time.strftime("%Y-%m-%d %H:%M:%S") if run.start_time else "Not started"
            duration = str(run.total_run_time).split('.')[0] if run.total_run_time else "N/A"
            
            print(f"🎯 {run.name}")
            print(f"   Status: {run.state.type}")
            print(f"   Start: {start_time}")
            print(f"   Duration: {duration}")
            print()

# Run health check
asyncio.run(check_flow_health())
```

## 🚨 Alert Configuration

### Telegram Setup

1. **Create Telegram Bot**:
   ```
   Message @BotFather on Telegram
   /newbot
   Choose bot name and username
   Save the bot token
   ```

2. **Get Chat ID**:
   ```
   Message @userinfobot on Telegram
   /start
   Note your chat ID
   ```

3. **Test Alerts**:
   ```bash
   # Test alert with dry-run
   export TELEGRAM_DRY_RUN="true"
   python scripts/test_canary_thresholds.py
   
   # Test live alert (be careful!)
   export TELEGRAM_DRY_RUN="false"
   python -c "
   from mech_exo.utils.alerts import AlertManager, Alert, AlertType, AlertLevel
   from datetime import datetime
   
   alert = Alert(
       alert_type=AlertType.SYSTEM_INFO,
       level=AlertLevel.INFO,
       title='🧪 Prefect Deployment Test',
       message='Canary flow deployment and alert system working correctly.',
       timestamp=datetime.now()
   )
   
   alert_manager = AlertManager()
   success = alert_manager.send_alert(alert, channels=['telegram'])
   print(f'Alert sent: {success}')
   "
   ```

## 🐛 Troubleshooting

### Common Issues

**Flow not starting:**
```bash
# Check deployment status
prefect deployment ls

# Check agent status
prefect agent ls

# Ensure agent is running with correct work pool
prefect agent start --pool "default-agent-pool"
```

**Database connection errors:**
```bash
# Check database file permissions
ls -la data/mech_exo.duckdb

# Test database connection
python -c "
from mech_exo.datasource.storage import DataStorage
storage = DataStorage()
count = storage.conn.execute('SELECT COUNT(*) FROM canary_performance').fetchone()[0]
print(f'Performance records: {count}')
storage.close()
"
```

**Missing dependencies:**
```bash
# Install missing packages
pip install -r requirements.txt

# Verify imports
python -c "
from mech_exo.execution.allocation import get_allocation_config
from mech_exo.reporting.pnl import store_daily_performance
print('✅ All imports successful')
"
```

### Flow Debugging

**Manual execution:**
```bash
# Run flow manually for specific date
cd dags
python canary_perf_flow.py 2024-01-15

# Run with debug logging
PYTHONPATH=/path/to/mech_exo python -m prefect.logging.configuration --level DEBUG canary_perf_flow.py
```

**Step-by-step debugging:**
```python
# debug_canary_steps.py
import sys
from pathlib import Path
from datetime import date

sys.path.append(str(Path(__file__).parent.parent))

from dags.canary_perf_flow import (
    pull_fills_today, calc_canary_pnl, 
    update_canary_performance, compute_rolling_sharpe_metrics,
    check_auto_disable_logic
)

# Debug each step individually
target_date = date.today()

print("🔍 Step 1: Pull fills")
fills_result = pull_fills_today(target_date)
print(f"Result: {fills_result}")

print("🔍 Step 2: Calculate P&L")
pnl_result = calc_canary_pnl(fills_result)
print(f"Result: {pnl_result}")

# Continue with other steps...
```

## 📈 Performance Optimization

### Resource Configuration

```python
# For high-volume trading environments
deployment = Deployment.build_from_flow(
    flow=canary_performance_flow,
    name="daily-canary-tracker-optimized",
    infrastructure=ProcessRunner(
        env={
            "PREFECT_LOGGING_LEVEL": "INFO",
            "PYTHONPATH": "/path/to/mech_exo",
            "OMP_NUM_THREADS": "2"  # Limit CPU usage
        }
    ),
    schedule=CronSchedule(cron="30 23 * * *", timezone="UTC")
)
```

### Database Optimization

```sql
-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_canary_performance_date ON canary_performance(date);
CREATE INDEX IF NOT EXISTS idx_fills_filled_at ON fills(filled_at);
CREATE INDEX IF NOT EXISTS idx_fills_tag ON fills(tag);
```

## 🔄 Backup & Recovery

### Configuration Backup

```bash
# Backup allocation config
cp config/allocation.yml config/allocation.yml.backup.$(date +%Y%m%d)

# Backup database
cp data/mech_exo.duckdb data/mech_exo.duckdb.backup.$(date +%Y%m%d)
```

### Recovery Procedures

```bash
# Restore from backup
cp config/allocation.yml.backup.20240115 config/allocation.yml

# Rebuild missing performance data
python -c "
from mech_exo.reporting.pnl import store_daily_performance
from datetime import date, timedelta

# Rebuild last 30 days
for i in range(30):
    target_date = date.today() - timedelta(days=i)
    success = store_daily_performance(target_date)
    print(f'{target_date}: {success}')
"
```

## 📊 Deployment Checklist

- [ ] ✅ Prefect server/cloud configured
- [ ] ✅ Environment variables set (TELEGRAM_BOT_TOKEN, etc.)
- [ ] ✅ Dependencies installed (prefect>=2.14.0)
- [ ] ✅ Database accessible and properly structured
- [ ] ✅ Flow deployed with correct schedule (23:30 UTC)
- [ ] ✅ Agent running on target environment
- [ ] ✅ Telegram alerts tested and working
- [ ] ✅ Health checks configured
- [ ] ✅ Backup procedures in place
- [ ] ✅ Monitoring dashboards configured

---

**Next Steps:**
1. Deploy flow to staging environment
2. Run manual test execution
3. Monitor for 1 week before production
4. Configure production alerts and monitoring