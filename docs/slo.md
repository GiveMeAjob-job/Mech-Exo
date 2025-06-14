# Service Level Objectives (SLO) - Risk Management System

## Overview

This document defines the Service Level Objectives (SLOs) for the Mech-Exo risk management system, implemented as part of Phase P11 Week 2. SLOs provide measurable targets for system reliability and help balance feature velocity with operational stability.

## SLO Definitions

### 1. Availability SLO

**Target**: 99% availability over any 24-hour period

- **SLI (Service Level Indicator)**: `risk_ops_ok{env="prod"}` metric
- **Measurement Window**: 24 hours (rolling)
- **Error Budget**: 1% = 14.4 minutes of downtime per day

**Formula**:
```
Availability = (uptime_minutes / total_minutes) * 100
Error Budget Remaining = 100 - (downtime_minutes * 100 / 1440)
```

**Prometheus Query**:
```promql
# Availability percentage
avg_over_time(risk_ops_ok{env="prod"}[24h]) * 100

# Error budget remaining
100 - (sum_over_time((1 - risk_ops_ok{env="prod"})[24h:1m]) * 100 / 1440)
```

### 2. Response Time SLO

**Target**: 95% of requests complete within 200ms

- **SLI**: HTTP request duration 95th percentile
- **Measurement Window**: 5 minutes
- **Threshold**: 200 milliseconds

**Prometheus Query**:
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="risk-api"}[5m])) * 1000
```

### 3. Error Rate SLO

**Target**: <1% error rate over any 5-minute period

- **SLI**: Ratio of 5xx responses to total responses
- **Measurement Window**: 5 minutes
- **Threshold**: 1%

**Prometheus Query**:
```promql
rate(http_requests_total{job="risk-api",status=~"5.."}[5m]) / rate(http_requests_total{job="risk-api"}[5m]) * 100
```

## Error Budget Policy

### Error Budget Calculation

Error budgets represent the acceptable amount of unreliability. They balance innovation velocity with operational stability.

**24-hour Error Budget**:
- **100% budget**: 1440 minutes (24 hours)
- **99% SLO**: 14.4 minutes downtime allowance
- **Remaining budget**: `100 - (actual_downtime_minutes * 100 / 1440)`

### Burn Rate Thresholds

| Burn Rate | Time to Exhaustion | Alerting Level | Action Required |
|-----------|-------------------|----------------|-----------------|
| 0.1%/hour | 1000 hours | Info | Monitor |
| 0.5%/hour | 200 hours | Warning | Investigate |
| 2%/hour | 50 hours | Critical | Immediate action |
| 10%/hour | 10 hours | Pager | Emergency response |

### Error Budget Policies

1. **Green (>99.5% budget remaining)**:
   - Normal operations
   - All deployments allowed
   - Feature development continues

2. **Yellow (99-99.5% budget remaining)**:
   - Increased monitoring
   - Non-critical deployments postponed
   - Focus on reliability improvements

3. **Red (<99% budget remaining)**:
   - Emergency response mode
   - All non-critical deployments halted
   - Mandatory reliability work
   - Daily error budget review meetings

## Alerting Configuration

### Critical Alerts (Pager)

1. **ErrorBudgetBurnHigh**:
   - Trigger: Error budget < 98% for 5 minutes
   - Severity: `pager`
   - Team: `trading_ops`

2. **RiskSystemDown**:
   - Trigger: `risk_ops_ok == 0` for 1 minute
   - Severity: `pager`
   - Team: `trading_ops`

3. **VaRLimitBreach**:
   - Trigger: VaR utilization > 95% for 2 minutes
   - Severity: `pager`
   - Team: `risk_management`

### Warning Alerts

1. **ErrorBudgetLow**:
   - Trigger: Error budget < 99% for 10 minutes
   - Severity: `warning`
   - Team: `trading_ops`

2. **RiskAPISlowResponse**:
   - Trigger: 95th percentile latency > 200ms for 3 minutes
   - Severity: `warning`
   - Team: `platform_engineering`

## Monitoring and Dashboards

### Grafana Dashboards

**Primary Dashboard**: Risk Control Dashboard
- URL: `http://grafana.mech-exo.com/d/risk-control`
- Panels:
  - SLO Error Budget (with red/yellow/green thresholds)
  - 24-hour Availability
  - Downtime Minutes
  - Error Budget Burn Rate

**SLO-specific panels**:
```json
{
  "title": "SLO Error Budget",
  "type": "stat",
  "thresholds": {
    "steps": [
      {"color": "red", "value": 98},
      {"color": "yellow", "value": 99},
      {"color": "green", "value": 99.5}
    ]
  }
}
```

### Key Metrics

| Metric | Query | Unit | Good/Bad Threshold |
|--------|-------|------|-------------------|
| Error Budget | `risk:error_budget_remaining` | % | >99% / <98% |
| Availability | `risk:availability_24h` | % | >99% / <98% |
| Downtime | `risk:downtime_minutes_24h` | minutes | <14.4 / >28.8 |
| Response Time | `risk:response_time_95p_5m` | ms | <200 / >500 |

## Incident Response

### Error Budget Exhaustion Response

**When error budget < 98%**:

1. **Immediate (0-15 minutes)**:
   - Page on-call engineer
   - Create incident channel
   - Stop all non-critical deployments

2. **Short-term (15-60 minutes)**:
   - Identify root cause
   - Implement immediate fixes
   - Escalate to service owner

3. **Medium-term (1-24 hours)**:
   - Post-incident review
   - Create reliability tasks
   - Update runbooks

### System Down Response

**When `risk_ops_ok == 0`**:

1. **Check health endpoints**:
   ```bash
   curl -f http://risk-api:8050/healthz
   curl -f http://risk-api:8050/riskz
   ```

2. **Verify infrastructure**:
   ```bash
   kubectl get pods -l app=risk-api
   kubectl logs -l app=risk-api --tail=100
   ```

3. **Emergency procedures**:
   - Activate kill-switch if necessary
   - Notify trading desk
   - Escalate to CTO if trading halt required

## Tuning and Optimization

### SLO Adjustment Process

SLOs should be reviewed quarterly and adjusted based on:

1. **Business requirements changes**
2. **Historical performance data**
3. **Customer feedback and impact**
4. **Technology improvements**

### Optimization Strategies

1. **Preventive**:
   - Chaos engineering testing
   - Regular disaster recovery drills
   - Proactive capacity planning

2. **Detective**:
   - Enhanced monitoring and alerting
   - Distributed tracing implementation
   - Performance profiling

3. **Corrective**:
   - Automated rollback mechanisms
   - Circuit breaker patterns
   - Graceful degradation

## Implementation Details

### Prometheus Configuration

**Recording Rules** (`prometheus/risk_rules.yml`):
```yaml
groups:
  - name: risk_slo_rules
    rules:
      - record: risk:error_budget_remaining
        expr: 100 - (sum_over_time((1 - risk_ops_ok{env="prod"})[24h:1m]) * 100 / 1440)
```

**Alerting Rules** (`prometheus/risk_alerts.yml`):
```yaml
groups:
  - name: risk_slo_alerts
    rules:
      - alert: ErrorBudgetBurnHigh
        expr: risk:error_budget_remaining < 98
        for: 5m
        labels:
          severity: pager
```

### Metrics Collection

**Risk Exporter** (`prometheus/risk_exporter.py`):
```python
self.ops_ok_gauge = Gauge(
    'risk_ops_ok',
    'System operational status (1=OK, 0=Down)',
    ['env']
)
```

## Testing and Validation

### SLO Testing

1. **Unit Tests**:
   ```bash
   pytest tests/slo/test_error_budget.py
   ```

2. **Integration Tests**:
   ```bash
   pytest tests/integration/test_slo_alerts.py
   ```

3. **Load Tests**:
   ```bash
   pytest tests/load/test_slo_performance.py
   ```

### Validation Commands

```bash
# Check error budget calculation
curl -s http://prometheus:9090/api/v1/query -d 'query=risk:error_budget_remaining'

# Verify alerting rules
promtool check rules prometheus/risk_alerts.yml

# Test Grafana dashboard
curl -f http://grafana:3000/api/health
```

## References

- [Google SRE Book - SLI, SLO, and Error Budgets](https://sre.google/sre-book/service-level-objectives/)
- [Prometheus Recording Rules](https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/)
- [Grafana Alerting](https://grafana.com/docs/grafana/latest/alerting/)
- [Mech-Exo Runbooks](./runbooks/)

---

**Document Version**: 1.0  
**Last Updated**: 2025-06-13  
**Owner**: Platform Engineering Team  
**Review Cycle**: Quarterly