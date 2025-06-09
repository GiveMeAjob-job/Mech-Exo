# Prefect Nightly Backtest Deployment Guide

This guide explains how to deploy the nightly backtest flow using Prefect for automated strategy monitoring.

## 🚀 Quick Start

### 1. Install Prefect

```bash
pip install prefect>=2.14.0
```

### 2. Start Prefect Server (Local Development)

```bash
prefect server start
```

The Prefect UI will be available at http://localhost:4200

### 3. Deploy the Nightly Backtest Flow

```python
from dags.backtest_flow import create_nightly_backtest_deployment

# Create deployment with 03:30 EST schedule (08:30 UTC)
deployment_id = create_nightly_backtest_deployment()
print(f"Deployment created: {deployment_id}")
```

### 4. Run Manual Backtest (Testing)

```python
from dags.backtest_flow import run_manual_backtest

# Test the flow manually
result = run_manual_backtest(lookback="365D", symbols=["SPY", "QQQ", "IWM"])
print(f"Manual backtest result: {result}")
```

## 📊 Flow Components

### 1. **Signal Generation** (`generate_recent_signals`)
- Generates trading signals for specified lookback period
- Supports symbols list or defaults to diversified ETF portfolio
- Uses simple buy-and-hold strategy (can be enhanced with actual strategy signals)

### 2. **Backtest Execution** (`run_recent_backtest`)
- Runs vectorbt backtest with realistic fees and slippage
- Calculates comprehensive performance metrics
- Handles both net and gross performance calculations

### 3. **Metrics Storage** (`store_backtest_metrics`)
- Creates `backtest_metrics` table in DuckDB if not exists
- Stores JSON-serializable metrics for health endpoint consumption
- Includes performance, risk, and cost metrics

### 4. **Tearsheet Generation** (`generate_tearsheet_artifact`)
- Exports interactive HTML tearsheet
- Creates Prefect artifact for UI display
- Saves tearsheet file for external access

### 5. **Alert Monitoring** (`check_backtest_alerts`)
- Monitors Sharpe ratio and max drawdown thresholds
- Sends Slack alerts via existing AlertManager
- Configurable via environment variables

## ⚙️ Configuration

### Environment Variables

```bash
# Alert thresholds
export ALERT_SHARPE_MIN=0.5        # Minimum acceptable Sharpe ratio
export ALERT_MAX_DD_PCT=20.0       # Maximum acceptable drawdown (%)

# Slack alerting (optional)
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Backtest Configuration

The flow uses `config/backtest.yml` for backtesting parameters:

```yaml
# Trading costs
commission_per_share: 0.005
slippage_pct: 0.001
spread_cost_pct: 0.0005

# Risk parameters
max_portfolio_var: 0.02
max_single_position_var: 0.005

# Execution settings
execution_delay: 1
fill_probability: 0.98
```

## 📈 Monitoring & Health Checks

### Health Endpoint Integration

The `/healthz` endpoint now includes backtest metrics:

```bash
curl -H "Accept: application/json" http://localhost:8050/healthz
```

Response includes:
```json
{
  "status": "operational",
  "risk_ok": true,
  "timestamp": "2025-06-08T19:30:00Z",
  "fills_today": 12,
  "backtest_sharpe": 1.8,
  "backtest_cagr": 0.15,
  "backtest_max_dd": -0.08,
  "backtest_date": "2025-06-08T03:30:00Z",
  "backtest_trades": 48
}
```

### DuckDB Metrics Table

```sql
-- Query recent backtest results
SELECT 
    backtest_date,
    period_start,
    period_end,
    cagr_net,
    sharpe_net,
    max_drawdown,
    total_trades
FROM backtest_metrics 
ORDER BY backtest_date DESC 
LIMIT 10;
```

## 🔄 Scheduling

### Cron Schedule: 03:30 EST Daily

```python
# UTC time (EST + 5 hours during standard time)
schedule=CronSchedule(cron="30 8 * * *")  # 08:30 UTC = 03:30 EST
```

### Custom Schedule

```python
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

deployment = Deployment.build_from_flow(
    flow=nightly_backtest_flow,
    name="custom-backtest",
    schedule=CronSchedule(cron="0 10 * * *"),  # 10:00 AM UTC daily
    parameters={"lookback": "365D", "symbols": ["SPY", "QQQ", "IWM", "EFA"]}
)
```

## 🚨 Alerting

### Automatic Alerts

Alerts are sent when:
- **Sharpe Ratio** < 0.5 (configurable via `ALERT_SHARPE_MIN`)
- **Max Drawdown** > 20% (configurable via `ALERT_MAX_DD_PCT`)
- **Flow Failure** (any unhandled exception)

### Alert Message Format

```
🚨 **Backtest Alert: Low Sharpe Ratio**

**Period**: 2024-01-01 to 2024-12-31
**Sharpe Ratio**: 0.3 (threshold: 0.5)
**CAGR**: 8.50%
**Max Drawdown**: -12.50%
**Total Trades**: 48

Strategy performance may need review.
```

## 🏗️ Production Deployment

### 1. Prefect Cloud Setup

```bash
# Login to Prefect Cloud
prefect cloud login

# Create work pool
prefect work-pool create --type process mech-exo-pool
```

### 2. Server Deployment

```bash
# Create deployment
prefect deploy dags/backtest_flow.py:nightly_backtest_flow

# Start worker
prefect worker start --pool mech-exo-pool
```

### 3. Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["prefect", "worker", "start", "--pool", "mech-exo-pool"]
```

## 📝 Logs & Debugging

### Flow Logs

```bash
# View recent flow runs
prefect flow-run ls --limit 10

# View specific flow run logs
prefect flow-run logs <flow-run-id>
```

### Manual Testing

```python
# Test individual tasks
from dags.backtest_flow import generate_recent_signals, run_recent_backtest

signals = generate_recent_signals(lookback="30D", symbols=["SPY"])
result = run_recent_backtest(signals, lookback="30D")
print(f"Test result: {result['metrics']}")
```

## 🔧 Troubleshooting

### Common Issues

1. **vectorbt Import Error**
   ```bash
   pip install vectorbt>=0.27.0
   ```

2. **DuckDB Connection Issues**
   - Check `data/mech_exo.duckdb` file permissions
   - Verify DataStorage configuration

3. **Alert Delivery Failures**
   - Verify Slack webhook URL
   - Check AlertManager configuration

4. **Flow Scheduling Issues**
   - Confirm Prefect server is running
   - Check work pool status
   - Verify timezone configuration

### Debug Mode

```python
# Run with debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

from dags.backtest_flow import nightly_backtest_flow
result = nightly_backtest_flow(lookback="30D", symbols=["SPY"])
```

## 📊 Metrics & Performance

### Key Metrics Tracked

- **Performance**: Total Return, CAGR, Sharpe Ratio, Sortino Ratio
- **Risk**: Max Drawdown, Volatility, Calmar Ratio
- **Trading**: Win Rate, Profit Factor, Trade Count, Average Duration
- **Costs**: Total Fees, Cost Drag, Fee per Trade

### Historical Analysis

```sql
-- Performance trend over time
SELECT 
    DATE_TRUNC('month', backtest_date) as month,
    AVG(sharpe_net) as avg_sharpe,
    AVG(cagr_net) as avg_cagr,
    AVG(max_drawdown) as avg_max_dd
FROM backtest_metrics 
GROUP BY month 
ORDER BY month DESC;
```

This automated backtesting system provides continuous strategy monitoring with comprehensive alerting and reporting capabilities.