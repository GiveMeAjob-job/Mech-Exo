# On-Call Run-Book

This document provides operational procedures for on-call engineers managing the Mech-Exo trading system.

## Table of Contents

- [System Overview](#system-overview)
- [Daily Stop-Loss (-0.8%)](#daily-stop-loss--08)
- [Monthly Stop-Loss (-3%)](#monthly-stop-loss--3)
- [Rollback Drill Procedure](#rollback-drill-procedure)
- [Kill-Switch Operations](#kill-switch-operations)
- [Risk Management Dashboard](#risk-management-dashboard)
- [Alert Escalation](#alert-escalation)
- [Common Troubleshooting](#common-troubleshooting)
- [Emergency Contacts](#emergency-contacts)

## System Overview

### Architecture
- **Trading Engine**: Python-based quantitative trading system
- **Risk Management**: Multi-layered risk controls with automatic kill-switches
- **Monitoring**: Real-time dashboard + Telegram alerts
- **Data Storage**: DuckDB for analytics, SQLite for execution data

### Key Components
- **Kill-Switch**: Emergency trading halt mechanism
- **Intraday Sentinel**: 5-minute monitoring with -0.8% daily threshold
- **Monthly Guard**: End-of-day monitoring with -3% monthly threshold
- **Alert System**: Telegram notifications with escalation

---

## Daily Stop-Loss (-0.8%)

### Overview
Automated intraday monitoring that triggers kill-switch when daily P&L falls to -0.8% or below.

### Symptoms
- 🔴 **Critical Alert**: "CRITICAL DAY LOSS: -X.XX%"
- Dashboard shows red PnL indicator
- Trading automatically disabled

### Immediate Actions
1. **Verify Alert**:
   ```bash
   exo kill status
   exo dashboard health
   ```

2. **Check Position Status**:
   ```bash
   exo positions summary
   exo risk current
   ```

3. **Review Market Conditions**:
   - Check major indices (SPY, QQQ, VIX)
   - Verify no unusual market events
   - Confirm execution data integrity

### Investigation Steps
1. **Analyze Loss Sources**:
   ```bash
   exo fills today --summary
   exo positions --sort-by pnl
   ```

2. **Check Risk Metrics**:
   ```bash
   exo risk violations
   exo scoring factor-health
   ```

3. **Verify Data Quality**:
   ```bash
   exo data validate --date today
   exo flows status intraday-pnl-sentinel
   ```

### Recovery Procedures
1. **If Loss is Valid**:
   - Document the cause in incident log
   - Wait for next trading day or manual override
   - Consider position adjustments for next session

2. **If Data Issue**:
   ```bash
   # Fix data and recalculate
   exo data refresh --source [fills|prices]
   exo pnl recalculate --date today
   ```

3. **Re-enable Trading** (only after approval):
   ```bash
   exo kill on --reason "Issue resolved - [brief description]"
   ```

### Escalation
- **< 15 minutes**: Self-investigate
- **15-30 minutes**: Alert Lead Engineer
- **> 30 minutes**: Escalate to CTO

---

## Monthly Stop-Loss (-3%)

### Overview
End-of-day monitoring that triggers kill-switch when month-to-date P&L falls to -3% or below.

### Symptoms
- 🛑 **Critical Alert**: "MONTHLY STOP-LOSS TRIGGERED: -X.XX%"
- Red stop-sign emoji in Telegram
- Monthly dashboard badge turns red
- Trading automatically disabled

### Immediate Actions
1. **Verify Monthly Calculation**:
   ```bash
   exo monthly status
   exo monthly summary --verbose
   ```

2. **Check Historical Context**:
   ```bash
   exo monthly history --months 3
   exo nav series --days 30
   ```

3. **Validate Data Integrity**:
   ```bash
   exo flows status monthly-drawdown-guard
   exo data validate --range month-to-date
   ```

### Investigation Steps
1. **Analyze Monthly Performance**:
   ```bash
   exo monthly breakdown --by-day
   exo monthly breakdown --by-strategy
   ```

2. **Review Risk Attribution**:
   ```bash
   exo risk monthly-attribution
   exo scoring monthly-performance
   ```

3. **Check System Health**:
   ```bash
   exo health full-check
   exo flows runs --flow monthly-drawdown-guard --count 5
   ```

### Recovery Procedures

#### If Loss is Valid (Genuine Market Loss)
1. **Document Incident**:
   - Create incident report with root cause analysis
   - Review strategy performance and risk management
   - Consider strategy adjustments or position limits

2. **Strategic Review**:
   - Schedule emergency strategy review meeting
   - Analyze factor performance and model drift
   - Consider temporary strategy modifications

3. **Re-enable Trading** (requires CTO approval):
   ```bash
   # Only after approval and risk review
   exo kill on --reason "Monthly review complete - CTO approved"
   ```

#### If Data/Calculation Issue
1. **Identify Root Cause**:
   ```bash
   exo monthly debug --verbose
   exo nav validate --range month-to-date
   ```

2. **Fix Data Issue**:
   ```bash
   exo data repair --type [fills|nav|daily-metrics]
   exo monthly recalculate --force
   ```

3. **Verify Fix**:
   ```bash
   exo monthly status
   exo test monthly-guard --dry-run
   ```

4. **Re-enable Trading**:
   ```bash
   exo kill on --reason "Data issue resolved - calculation corrected"
   ```

### Monthly Stop-Loss Configuration
Location: `config/risk.yml`
```yaml
monthly_stop:
  enabled: true
  threshold_pct: -3.0
  min_history_days: 10
```

### Escalation Timeline
- **Immediate**: Alert on-call engineer
- **< 30 minutes**: Notify Lead Engineer
- **< 1 hour**: Escalate to CTO
- **> 1 hour**: Activate incident response team

---

## Rollback Drill Procedure

### Overview
Quarterly rollback drills test the complete kill-switch backup/restore cycle to ensure operational readiness during emergencies.

### Drill Frequency
- **Recommended**: Every 90 days
- **Maximum Interval**: 120 days
- **Dashboard Monitoring**: Color-coded badge shows days since last drill
  - 🟢 Green: < 90 days (current)
  - 🟡 Yellow: 90-120 days (due soon) 
  - 🔴 Red: > 120 days (overdue)

### Manual Drill Execution

#### Prerequisites
- Ensure no active trading or critical operations
- Coordinate with team (avoid market hours if possible)
- Have dashboard access for monitoring

#### Dry-Run Drill (Safe Testing)
```bash
# Test drill without affecting live trading
exo drill rollback --dry-run --wait 60

# Quick test for CI/development
exo drill rollback --dry-run --wait 1
```

#### Live Drill (Quarterly Requirement)
```bash
# Full operational drill with actual kill-switch toggle
exo drill rollback --wait 120
```

### Drill Process Overview

The drill executes a 4-step test cycle:

1. **Step A: Backup Configuration**
   - Creates timestamped backup of `killswitch.yml`
   - Verifies backup integrity before proceeding

2. **Step B: Disable Trading**
   - Executes `exo kill off --reason "ROLLBACK_DRILL"`
   - Verifies `trading_enabled=false` in config

3. **Step C: Wait Period**
   - Maintains disabled state for specified duration
   - Default: 120 seconds (configurable via `--wait`)

4. **Step D: Restore Trading**
   - Executes `exo kill on --reason "ROLLBACK_DRILL_COMPLETE"`
   - Verifies `trading_enabled=true` in config

### Automated Drill Execution

#### Prefect Flow
```bash
# Manual flow execution
prefect flow run rollback-drill --param dry_run=false

# Scheduled quarterly execution
prefect deployment create dags/drill_flow.py:rollback_drill_flow
```

#### Parameters
- `dry_run`: Safety mode (default: true)
- `wait_seconds`: Wait duration (default: 120)
- `interval_days`: Frequency check (default: 90)
- `skip_frequency_check`: Force drill regardless of schedule

### Drill Reports

#### Report Generation
- Automatically generates `drill_YYYYMMDD_HHMM.md` report
- Contains step-by-step execution log with timestamps
- Includes pass/fail status and performance metrics
- Reports stored in project root directory

#### Report Delivery
- 📱 **Telegram**: Automatic alert with report attachment
- 📊 **Dashboard**: Status badge updates with last drill date
- 📁 **Artifacts**: CI uploads reports for team access

#### Report Contents
- Executive summary (PASS/FAIL)
- Detailed step execution times
- Error analysis (if applicable)
- Recommendations for next steps
- Placeholder for dashboard screenshot

### Troubleshooting Drill Issues

#### Drill Lock Error
```bash
# Error: Another drill is already running
ls -la .drill_lock

# If stale lock exists, remove manually
rm .drill_lock
```

#### Backup Creation Failed
```bash
# Check file permissions
ls -la config/killswitch.yml

# Verify disk space
df -h .

# Check config directory write access
touch config/test_write && rm config/test_write
```

#### Kill-Switch Command Failed
```bash
# Test kill-switch CLI directly
exo kill status
exo kill off --reason "test"
exo kill on --reason "test complete"

# Check configuration syntax
python -c "import yaml; yaml.safe_load(open('config/killswitch.yml'))"
```

#### Emergency Restore
If drill fails and trading remains disabled:

```bash
# Check for backup files
ls -la config/killswitch_backup_*.yml

# Manual restore from most recent backup
cp config/killswitch_backup_YYYYMMDD_HHMMSS.yml config/killswitch.yml

# Verify restoration
exo kill status
```

### Drill Success Criteria

#### Passing Drill
- ✅ All 4 steps complete successfully
- ✅ Trading disabled and re-enabled properly
- ✅ Configuration backed up and restored
- ✅ Report generated with no errors
- ✅ Telegram alert sent successfully

#### Failing Drill
- ❌ Any step fails or times out
- ❌ Configuration corruption
- ❌ Kill-switch commands fail
- ❌ Emergency restore required

### Post-Drill Actions

#### Successful Drill
1. Review report for performance metrics
2. Update drill schedule (next +90 days)
3. File report in operations log
4. Clean up backup files (automatic)

#### Failed Drill
1. **Immediate**: Ensure trading is restored
2. **Within 24h**: Root cause analysis
3. **Within 48h**: Fix underlying issues
4. **Within 7d**: Re-run drill to verify fixes

### Drill Scheduling Reminders

#### Automated Alerts
- Daily 09:00 UTC check if last drill > 90 days
- Dashboard badge shows overdue status
- Prefect flow includes frequency checking

#### Manual Scheduling
```bash
# Check last drill date
exo drill status

# Check if drill is due
python -c "
from dags.drill_flow import get_last_drill_info
info = get_last_drill_info()
print(f'Last drill: {info}')
"
```

### CI Integration

#### Automated Testing
- Every push/PR triggers drill smoke test
- Tests both success and failure scenarios
- Validates report generation
- Completes in < 4 minutes

#### Artifact Access
- Drill reports uploaded as CI artifacts
- Available for 30 days post-execution
- Downloadable from GitHub Actions

### Drill Metrics Tracking

The system maintains comprehensive drill logs in the `drill_log` database table:

- Execution timestamps and duration
- Pass/fail status and error details
- Dry-run vs live drill classification
- Report file paths and metadata

Access via dashboard or direct SQL queries for trend analysis.

---

## Kill-Switch Operations

### Status Commands
```bash
# Check current status
exo kill status

# View history
exo kill history --days 7

# Health check
exo dashboard health
```

### Manual Operations
```bash
# Disable trading
exo kill off --reason "Manual halt - [reason]"

# Enable trading  
exo kill on --reason "Issue resolved - [description]"

# Dry run (test mode)
exo kill off --reason "Test" --dry-run
```

### Configuration
Location: `config/killswitch.yml`
- Contains current state, reason, and history
- Modified by CLI commands and automated systems

---

## Risk Management Dashboard

### Overview
Real-time risk monitoring dashboard providing VaR tracking, position analysis, and alert management.

### Dashboard Access
- **Live Dashboard**: `/dashboard/risk-live`
- **Health Check**: `/health`
- **API Endpoint**: `/api/risk/current`

### Key Components

#### Risk Heatmap
- Visual representation of portfolio risk by sector/strategy
- Color coding: 🟢 Green (low risk) → 🟡 Yellow (medium) → 🔴 Red (high)
- Updates every 30 seconds during market hours

#### VaR Timeline
- Real-time Value at Risk tracking
- 95% and 99% confidence intervals
- Historical trend analysis
- Alert thresholds visualization

#### Position Breakdown
- Risk contribution by individual positions
- Concentration analysis
- Long/short exposure breakdown

### Risk Thresholds

#### VaR Limits
- **95% VaR**: Daily limit $2,000,000
- **99% VaR**: Daily limit $3,000,000
- **Position Concentration**: Max 15% per single name

#### Alert Levels
- **🟢 Normal**: VaR < 70% of limit
- **🟡 Warning**: VaR 70-90% of limit  
- **🔴 Critical**: VaR > 90% of limit

### Emergency Procedures

#### High Risk Alert (VaR > 90% limit)
1. **Immediate Actions**:
   ```bash
   # Check current risk metrics
   python -c "from mech_exo.reporting.query import RiskQueryEngine; print(RiskQueryEngine().get_live_risk_metrics())"
   
   # View dashboard
   open http://localhost:8050/dashboard/risk-live
   ```

2. **Risk Assessment**:
   - Identify highest risk contributors
   - Check for unusual market conditions
   - Verify position data accuracy

3. **Response Options**:
   - Reduce high-risk positions
   - Contact risk manager immediately
   - Consider emergency position exit

#### Critical Risk Alert (VaR > 100% limit)
1. **IMMEDIATE STOP**: Halt all new trading
   ```bash
   exo kill off --reason "VaR limit breach - emergency halt"
   ```

2. **Emergency Position Review**:
   - Assess current exposures
   - Calculate potential loss scenarios
   - Plan risk reduction strategy

3. **Escalation**: Contact CTO immediately

### Testing & Monitoring

#### End-to-End Tests
```bash
# Run comprehensive risk system test
python scripts/e2e_risk_test.py

# Test dashboard components
pytest tests/test_risk_panel.py -v

# Quick smoke test
python -c "
from mech_exo.reporting.dash_layout.risk_live import RiskLiveLayout
layout = RiskLiveLayout()
print('✅ Risk dashboard components loaded successfully')
"
```

#### Data Freshness Checks
```bash
# Verify risk data is current (< 5 minutes old)
python -c "
from mech_exo.reporting.query import RiskQueryEngine
from datetime import datetime, timedelta
engine = RiskQueryEngine()
last_update = engine.get_last_update_time()
age = datetime.now() - last_update
print(f'Data age: {age}')
assert age < timedelta(minutes=5), 'Risk data is stale!'
print('✅ Risk data is fresh')
"
```

### Environment Configuration

#### Required Environment Variables
```bash
# Telegram Alerts (Critical for risk notifications)
export TELEGRAM_BOT_TOKEN="1234567890:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
export TELEGRAM_CHAT_ID="-1001234567890"

# AWS Configuration (for S3 report storage)
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

# Database Connection
export DATABASE_URL="sqlite:///data/trading.db"

# Risk Dashboard
export DASH_HOST="0.0.0.0"
export DASH_PORT="8050"
```

#### Configuration Files
- `config/risk.yml`: Risk limits and thresholds
- `config/killswitch.yml`: Trading halt configuration
- `config/alerts.yml`: Alert routing and escalation

### Troubleshooting

#### Dashboard Won't Load
```bash
# Check dashboard service
curl -f http://localhost:8050/health || echo "Dashboard down!"

# Restart dashboard
python -m mech_exo.dashboard.app --host 0.0.0.0 --port 8050

# Check for memory issues
ps aux | grep python | grep dashboard
```

#### Stale Risk Data
```bash
# Check data pipeline status
python -c "
from mech_exo.reporting.query import RiskQueryEngine
engine = RiskQueryEngine()
print('Database status:', engine.check_connection())
print('Last update:', engine.get_last_update_time())
"

# Refresh risk calculations
python -c "
from mech_exo.reporting.query import RiskQueryEngine
engine = RiskQueryEngine()
engine.refresh_risk_metrics()
print('✅ Risk metrics refreshed')
"
```

#### Alert Delivery Failures
```bash
# Test Telegram connectivity
python scripts/test_telegram_alerts.py

# Manual alert test
python -c "
from mech_exo.utils.alerts import send_risk_alert
result = send_risk_alert({
    'type': 'TEST',
    'message': 'Risk system test alert',
    'severity': 'INFO'
})
print('Alert result:', result)
"
```

#### Memory Issues (JavaScript heap out of memory)
```bash
# Increase Node.js memory for large reports
export NODE_OPTIONS="--max-old-space-size=4096"

# Alternative: Use smaller batch sizes
python -c "
import os
os.environ['RISK_BATCH_SIZE'] = '100'  # Reduce from default 500
print('Reduced batch size for risk calculations')
"
```

---

## Alert Escalation

### Severity Levels

#### 🔴 **CRITICAL** (Immediate Response)
- Monthly stop-loss triggered (-3%)
- Daily stop-loss triggered (-0.8%)
- System errors affecting trading
- **Response Time**: < 15 minutes

#### 🟡 **WARNING** (Monitor Closely)
- Approaching daily threshold (-0.4% to -0.8%)
- Risk limit violations
- Data quality issues
- **Response Time**: < 30 minutes

#### 🟢 **INFO** (Awareness)
- Daily summaries
- System status updates
- Scheduled maintenance
- **Response Time**: Next business day

### Escalation Chain

1. **On-Call Engineer** (Primary)
   - First responder for all alerts
   - Authority to investigate and perform basic fixes
   - Must escalate if unable to resolve within SLA

2. **Lead Engineer** (Secondary)
   - Escalated for complex technical issues
   - Authority to modify system parameters
   - Decision maker for strategy adjustments

3. **CTO** (Final Authority)
   - Escalated for business-critical decisions
   - Authority to override risk limits
   - Final approval for monthly stop-loss recovery

### Communication Channels
- **Primary**: Telegram alerts to on-call channel
- **Secondary**: Email for documentation
- **Emergency**: Phone for urgent escalations

---

## Common Troubleshooting

### Dashboard Not Loading
```bash
# Check dashboard service
exo dashboard status
exo dashboard restart

# Check database connections
exo health database
```

### Missing Data
```bash
# Validate data sources
exo data validate --all-sources
exo data refresh --source [fills|ohlc|fundamentals]

# Check Prefect flows
exo flows status --all
exo flows restart --flow data-pipeline
```

### Alert System Issues
```bash
# Test alert system
exo test alerts --channel telegram
exo test alerts --dry-run

# Check alert configuration
exo alerts config
exo alerts history --hours 24
```

### Performance Issues
```bash
# Check system resources
exo health system
exo health database --verbose

# Monitor flows
exo flows monitor --live
```

---

## Emergency Contacts

### Primary On-Call
- **Engineer**: [On-call rotation]
- **Telegram**: @mech_exo_oncall
- **Phone**: [Emergency number]

### Secondary Escalation
- **Lead Engineer**: [Name]
- **Email**: [email@company.com]
- **Phone**: [Phone number]

### Executive Escalation
- **CTO**: [Name]
- **Email**: [email@company.com]
- **Phone**: [Emergency phone]

### External Contacts
- **Broker Support**: [IBKR support number]
- **Data Provider**: [Provider support]
- **Infrastructure**: [Cloud provider support]

---

## Maintenance Windows

### Daily
- **23:10 UTC**: Monthly guard flow execution
- **00:00 UTC**: Daily summary generation
- **06:00 UTC**: System health checks

### Weekly
- **Sunday 06:00 UTC**: Full system maintenance
- **Sunday 07:00 UTC**: Backup verification

### Monthly
- **First Sunday**: Strategy performance review
- **First Monday**: Risk limit review

---

*Last Updated: [Current Date]*
*Version: 1.0*
*Owner: Trading Operations Team*