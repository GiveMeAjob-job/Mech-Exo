# Go-Live Checklist - Risk Control System v0.5.1

## 🎯 Phase P11 Week 1 - Pilot Live Readiness

### Critical Phase P11 Milestones
- [ ] **Capital Expansion**: Canary accounts expanded to 30% allocation (target: $275k total)
- [ ] **Universe Expansion**: Trading universe increased to 500 securities (validated)
- [ ] **After-Hours Monitoring**: 16:00-20:00 ET monitoring active with /riskz ah_loss_pct field
- [ ] **Live Connection Test**: IB Gateway dry-run with 100% zero-quantity order success
- [ ] **Telegram Integration**: Real-time alerts working for all risk thresholds

### Phase P11 Validation Commands
```bash
# Validate capital expansion (should show 30%+ allocation)
python mech_exo/capital/manager.py validate

# Verify 500-security universe
cd datasource && python universe_loader.py validate --file test_universe_500.json

# Test after-hours monitoring
python dags/after_hours_pnl_flow.py test

# Validate live connections
python scripts/live_dry_run.py --orders 5

# Check risk dashboard for new fields
curl -s http://localhost:8050/riskz | jq '.ah_loss_pct'
```

### Phase P11 Success Criteria
- [ ] **Capital Utilization**: All canary accounts showing `capital_ok: true`
- [ ] **Universe Coverage**: Data quality score ≥ 85% for 500 securities
- [ ] **After-Hours Data**: `/riskz` endpoint includes `ah_loss_pct` field
- [ ] **Live Connectivity**: Dry-run success rate ≥ 95% with IB Gateway
- [ ] **Alert Delivery**: Telegram alerts delivered within 30 seconds

---

## Pre-Deployment Verification (20 Critical Items)

### 🔐 1. Secrets & Environment Variables
- [ ] **TELEGRAM_BOT_TOKEN** set in GitHub Secrets
- [ ] **TELEGRAM_CHAT_ID** set in GitHub Secrets  
- [ ] **AWS_ACCESS_KEY_ID** & **AWS_SECRET_ACCESS_KEY** configured (if using S3)
- [ ] **NODE_OPTIONS="--max-old-space-size=4096"** in CI environment
- [ ] Kubernetes secrets deployed: `kubectl get secret mech-exo-secrets`
- [ ] ConfigMap deployed: `kubectl get configmap mech-exo-config`

**Validation Commands:**
```bash
# Check GitHub secrets
gh secret list

# Validate secret format in CI
echo $TELEGRAM_BOT_TOKEN | md5sum  # Should show hash, not empty

# Test Kubernetes secrets
kubectl get secret mech-exo-secrets -o yaml | grep TELEGRAM_BOT_TOKEN
```

### 📊 2. CI/CD Pipeline Health
- [ ] **risk_master.yml** workflow passing on latest commit
- [ ] All GitHub Actions badges showing green
- [ ] No failing tests in test suite
- [ ] Security scanning (if enabled) shows no critical issues
- [ ] Dependency vulnerabilities resolved

**Validation Commands:**
```bash
# Check latest workflow runs
gh run list --workflow=risk_master.yml --limit=5

# Run local tests
python scripts/e2e_risk_test.py
pytest tests/test_risk_panel.py -v
```

### 🚀 3. Deployment Infrastructure
- [ ] Kubernetes cluster accessible: `kubectl cluster-info`
- [ ] Namespace created: `kubectl get namespace`
- [ ] Docker image built and tagged: `docker images | grep mech-exo`
- [ ] Image registry access verified
- [ ] Blue/green deployment script tested

**Validation Commands:**
```bash
# Test blue-green deployment
python scripts/deploy_blue_green.py --image mech-exo:v0.5.1-rc --dry-run

# Verify cluster resources
kubectl get nodes
kubectl get storageclass
```

### 📈 4. Monitoring & Observability  
- [ ] **Grafana** accessible and dashboards imported
- [ ] **Prometheus** scraping risk metrics exporter
- [ ] **Alertmanager** configured with PagerDuty integration
- [ ] Risk dashboard JSON validated: `jq . deploy/grafana/risk_dash.json`
- [ ] Alert rules syntax valid: `promtool check rules deploy/prometheus/risk_alerts.yml`

**Validation Commands:**
```bash
# Test Prometheus exporter
python prometheus/risk_exporter.py --help

# Validate Grafana dashboard
curl -f http://grafana:3000/api/health

# Check alert rules
promtool check rules deploy/prometheus/risk_alerts.yml
```

### 🔔 5. Alert System Integration
- [ ] **Telegram bot** created and token obtained
- [ ] **PagerDuty** service configured with integration key
- [ ] Test alert sent successfully: `python scripts/test_telegram_alerts.py`
- [ ] Alert escalation paths documented
- [ ] On-call rotation configured

**Validation Commands:**
```bash
# Test Telegram integration
python scripts/test_telegram_alerts.py

# Test staging alerts
python scripts/staging_dry_run.py --skip-telegram
```

### 🛡️ 6. Security & Access Control
- [ ] **GitHub PAT** scoped correctly (read:packages, repo)
- [ ] **S3 IAM policy** limited to required buckets only
- [ ] **Kubernetes RBAC** configured for least privilege
- [ ] No hardcoded secrets in code: `git secrets --scan`
- [ ] SSL/TLS certificates valid for ingress

### 🗄️ 7. Data & Storage
- [ ] **Database schema** migrations applied
- [ ] **S3 buckets** created with correct permissions
- [ ] **Backup strategy** implemented and tested
- [ ] Data retention policies configured
- [ ] Historical data migration completed (if applicable)

### 🧪 8. Testing & Quality Assurance
- [ ] **Unit tests** passing: `pytest tests/ -v`
- [ ] **Integration tests** passing: `python scripts/e2e_risk_test.py`
- [ ] **Load testing** completed (if applicable)
- [ ] **Security scanning** passed
- [ ] **Code coverage** meets requirements (>80%)

### 📚 9. Documentation
- [ ] **README-OPS.md** updated with production procedures
- [ ] **Runbook** links working in alerts
- [ ] **API documentation** current
- [ ] **Deployment guide** validated
- [ ] **Incident response** procedures documented

### ⚙️ 10. System Configuration
- [ ] **Risk limits** configured correctly in config/risk_limits.yml
- [ ] **Kill-switch** mechanism tested: `exo kill status`
- [ ] **Drill procedures** validated: `python scripts/rollback_drill.py --dry-run`
- [ ] **Resource limits** set appropriately in K8s
- [ ] **Auto-scaling** configured (if applicable)

---

## Pre-Go-Live Testing (Staging Validation)

### Staging Environment Checklist
- [ ] **Staging deployment** successful via blue-green script
- [ ] **Health checks** all passing (`/healthz`, `/riskz`, `/metrics`)
- [ ] **Risk calculations** accurate with test data
- [ ] **Kill-switch drill** completes successfully
- [ ] **Alert delivery** working to test channels

**Commands:**
```bash
# Deploy to staging
python scripts/deploy_blue_green.py --image mech-exo:v0.5.1-rc --namespace staging --dry-run

# Run staging tests  
python scripts/staging_dry_run.py --staging-url http://staging.mech-exo.com

# Test kill-switch
exo kill off --reason "Go-live testing" && sleep 30 && exo kill on --reason "Test complete"
```

---

## Production Deployment Steps

### 1. Final Preparation (T-1 Hour)
- [ ] **All checklist items** above verified ✅
- [ ] **Deployment window** communicated to stakeholders
- [ ] **On-call engineer** available for monitoring
- [ ] **Rollback plan** reviewed and ready
- [ ] **Monitoring dashboards** open and ready

### 2. Deployment Execution (T-0)
```bash
# Step 1: Deploy green environment
python scripts/deploy_blue_green.py --image mech-exo:v0.5.1 --namespace production

# Step 2: Validate health checks
curl -f http://production.mech-exo.com/healthz
curl -f http://production.mech-exo.com/riskz

# Step 3: Promote to production (switch traffic)
python scripts/deploy_blue_green.py --image mech-exo:v0.5.1 --namespace production --promote

# Step 4: Monitor for 15 minutes
watch -n 30 'curl -s http://production.mech-exo.com/healthz && echo "OK"'
```

### 3. Post-Deployment Validation (T+15 min)
- [ ] **All endpoints** responding correctly
- [ ] **Risk metrics** updating in real-time
- [ ] **Alerts** functioning (send test alert)
- [ ] **Grafana dashboards** showing live data
- [ ] **No error logs** in application logs

---

## Emergency Procedures

### Immediate Rollback (if issues detected)
```bash
# Quick rollback to blue environment
python scripts/deploy_blue_green.py --rollback

# Or manual kubectl rollback
kubectl patch service mech-exo -n production -p '{"spec":{"selector":{"version":"blue"}}}'
```

### Escalation Chain
1. **On-call Engineer** (immediate response)
2. **Lead Engineer** (within 15 minutes)
3. **CTO** (within 30 minutes)
4. **Emergency Contact**: [phone number]

---

## Success Criteria

### Technical Success
- [ ] System uptime > 99.5% in first 24 hours
- [ ] All critical alerts functioning
- [ ] Response times < 200ms for health endpoints
- [ ] No data loss or corruption
- [ ] All risk calculations accurate

### Business Success  
- [ ] Trading operations uninterrupted
- [ ] Risk monitoring fully operational
- [ ] Stakeholder confidence maintained
- [ ] No critical incidents in first week

---

## Post-Go-Live Actions (Week 1)

### Monitoring & Optimization
- [ ] **Monitor error rates** daily for first week
- [ ] **Review alert noise** and tune thresholds if needed
- [ ] **Performance optimization** based on production load
- [ ] **User feedback** collection and review
- [ ] **Capacity planning** for future growth

### Documentation Updates
- [ ] **Update production URLs** in documentation
- [ ] **Record lessons learned** from deployment
- [ ] **Update monitoring runbooks** with production specifics
- [ ] **Create post-deployment report**

---

## Version Information

- **Release Version**: v0.5.1
- **Release Date**: {{ RELEASE_DATE }}
- **Release Manager**: {{ RELEASE_MANAGER }}
- **Git Tag**: `git tag v0.5.1 && git push --tags`

---

## Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| **Release Manager** | | | |
| **Lead Engineer** | | | |
| **Security Review** | | | |
| **Operations Team** | | | |
| **Business Stakeholder** | | | |

---

*Checklist Version: 1.0*  
*Last Updated: 2024-12-13*  
*Owner: Trading Operations Team*