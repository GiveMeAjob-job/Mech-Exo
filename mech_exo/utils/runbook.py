"""
On-Call Runbook & Alert Escalation

Comprehensive runbook covering common operational incidents and resolution steps.
Includes escalation chains and quiet hours management.
"""

import logging
import os
import json
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels"""
    CRITICAL = "critical"      # System down, trading stopped
    HIGH = "high"             # Significant impact, immediate attention
    MEDIUM = "medium"         # Moderate impact, can wait for business hours
    LOW = "low"              # Minor issue, non-urgent


class EscalationLevel(Enum):
    """Alert escalation levels"""
    TELEGRAM = "telegram"     # First level: Telegram notification
    EMAIL = "email"          # Second level: Email alerts
    PHONE = "phone"          # Third level: Phone call (placeholder)


@dataclass
class RunbookEntry:
    """Single runbook entry for an incident type"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    symptoms: List[str]
    diagnosis_steps: List[str]
    resolution_steps: List[str]
    escalation_threshold_minutes: int
    contact_on_call: bool = True
    auto_remediation: Optional[str] = None


@dataclass
class EscalationRule:
    """Escalation rule configuration"""
    level: EscalationLevel
    delay_minutes: int
    enabled: bool = True
    quiet_hours_override: bool = False  # If True, ignores quiet hours


class OnCallRunbook:
    """On-call runbook and escalation manager"""
    
    def __init__(self):
        """Initialize runbook with common incidents"""
        self.runbook_entries = self._load_runbook_entries()
        self.escalation_rules = self._load_escalation_rules()
        self.quiet_hours = self._load_quiet_hours()
        
    def _load_runbook_entries(self) -> Dict[str, RunbookEntry]:
        """Load the 10 most common operational incidents"""
        
        entries = {
            "system_down": RunbookEntry(
                incident_id="system_down",
                title="🚨 System Down / Total Outage",
                description="The entire Mech-Exo system is unresponsive or completely offline",
                severity=IncidentSeverity.CRITICAL,
                symptoms=[
                    "Dashboard returns 500/503 errors",
                    "Health endpoint fails completely",
                    "No recent fills in database",
                    "Prefect flows failing to start",
                    "Database connection errors"
                ],
                diagnosis_steps=[
                    "Check system status: curl -H 'Accept: application/json' http://localhost:8050/healthz",
                    "Verify database connectivity: sqlite3 data/mech_exo.duckdb '.tables'",
                    "Check Prefect agent status: prefect agent ls",
                    "Review recent logs: tail -100 /var/log/mech-exo.log",
                    "Check system resources: top, df -h, free -m"
                ],
                resolution_steps=[
                    "1. Stop all services: systemctl stop mech-exo-*",
                    "2. Check for conflicting processes: ps aux | grep mech-exo",
                    "3. Verify database integrity: python -c 'from mech_exo.datasource.storage import DataStorage; DataStorage().health_check()'",
                    "4. Restart core services: systemctl start mech-exo-core",
                    "5. Start Prefect agent: prefect agent start --pool default",
                    "6. Verify health endpoint: curl http://localhost:8050/healthz",
                    "7. Monitor for 10 minutes to ensure stability",
                    "8. If still failing, escalate to senior engineer"
                ],
                escalation_threshold_minutes=5,
                contact_on_call=True
            ),
            
            "database_corruption": RunbookEntry(
                incident_id="database_corruption",
                title="💾 Database Corruption or Connectivity Issues",
                description="Database errors preventing normal operations",
                severity=IncidentSeverity.CRITICAL,
                symptoms=[
                    "SQLite database locked errors",
                    "Constraint violation errors",
                    "Missing table errors",
                    "Data integrity failures",
                    "Backup restoration needed"
                ],
                diagnosis_steps=[
                    "Check database file permissions: ls -la data/mech_exo.duckdb",
                    "Verify database integrity: sqlite3 data/mech_exo.duckdb 'PRAGMA integrity_check;'",
                    "Check for lock files: ls -la data/*.db-*",
                    "Review recent database errors in logs",
                    "Test read/write operations: python -c 'from mech_exo.datasource.storage import DataStorage; ds = DataStorage(); print(ds.health_check())'"
                ],
                resolution_steps=[
                    "1. Stop all database-writing services",
                    "2. Create emergency backup: cp data/mech_exo.duckdb data/mech_exo.backup.$(date +%Y%m%d_%H%M%S).duckdb",
                    "3. Run integrity check: sqlite3 data/mech_exo.duckdb 'PRAGMA integrity_check;'",
                    "4. If corrupted, restore from most recent backup in data/backups/",
                    "5. Replay missing transactions from logs if possible",
                    "6. Restart services and verify data consistency",
                    "7. Monitor closely for recurrence"
                ],
                escalation_threshold_minutes=10,
                contact_on_call=True
            ),
            
            "trading_halted": RunbookEntry(
                incident_id="trading_halted",
                title="⏹️ Trading Halted / No Orders Executing",
                description="Orders are not being executed despite signals being generated",
                severity=IncidentSeverity.HIGH,
                symptoms=[
                    "No fills recorded in last 2+ hours during market hours",
                    "Orders stuck in 'pending' status",
                    "Broker connectivity errors",
                    "Risk limits exceeded and blocked",
                    "Canary performance disabled trading"
                ],
                diagnosis_steps=[
                    "Check recent fills: mech-exo fills --days 1",
                    "Review order status: python -c 'from mech_exo.execution.order_router import OrderRouter; print(OrderRouter().get_pending_orders())'",
                    "Check risk breaches: mech-exo risk --status",
                    "Verify broker connection: python -c 'from mech_exo.execution.broker_adapter import get_broker; print(get_broker().health_check())'",
                    "Check canary status: grep canary_enabled data/mech_exo.duckdb"
                ],
                resolution_steps=[
                    "1. Check if market is open and trading enabled",
                    "2. Review and clear risk limit breaches if appropriate",
                    "3. Reset stuck orders: mech-exo orders --reset-pending",
                    "4. Re-enable canary if performance recovered: UPDATE canary_config SET enabled = 1",
                    "5. Test with small order: place manual test order",
                    "6. Monitor execution for next 30 minutes",
                    "7. If persists, check broker API status and escalate"
                ],
                escalation_threshold_minutes=15,
                contact_on_call=True
            ),
            
            "risk_breach": RunbookEntry(
                incident_id="risk_breach",
                title="⚠️ Risk Limits Breached / Stop Loss Triggered",
                description="Risk management system has triggered and halted trading",
                severity=IncidentSeverity.HIGH,
                symptoms=[
                    "Risk checker blocking new orders",
                    "Stop loss alerts triggered",
                    "Consecutive breach counter high",
                    "Drawdown exceeding thresholds",
                    "Position size limits exceeded"
                ],
                diagnosis_steps=[
                    "Check current risk status: mech-exo risk --detailed",
                    "Review recent P&L: mech-exo equity --days 7",
                    "Check drawdown metrics: grep drawdown logs/trading.log",
                    "Verify position sizes: mech-exo positions --summary",
                    "Review stop loss triggers: grep 'stop_loss' logs/risk.log"
                ],
                resolution_steps=[
                    "1. Assess if breach is legitimate market movement or system error",
                    "2. Review portfolio exposure and concentrated positions",
                    "3. If legitimate: wait for market conditions to improve",
                    "4. If false positive: adjust risk parameters temporarily",
                    "5. Reset breach counter if appropriate: mech-exo risk --reset-breaches",
                    "6. Gradually re-enable trading with reduced position sizes",
                    "7. Monitor closely and document incident"
                ],
                escalation_threshold_minutes=30,
                contact_on_call=True
            ),
            
            "data_pipeline_failure": RunbookEntry(
                incident_id="data_pipeline_failure",
                title="📊 Data Pipeline Failure / Stale Data",
                description="Data ingestion or processing pipeline has failed",
                severity=IncidentSeverity.MEDIUM,
                symptoms=[
                    "No new OHLC data in last 6+ hours",
                    "News feed not updating",
                    "Factor calculations stale",
                    "Data pipeline Prefect flow failing",
                    "External API errors"
                ],
                diagnosis_steps=[
                    "Check last data update: SELECT MAX(date) FROM ohlc_data",
                    "Review data pipeline logs: prefect flow-run logs data-pipeline",
                    "Test external APIs: python -c 'from mech_exo.datasource.ohlc import OHLCSource; print(OHLCSource().health_check())'",
                    "Check API rate limits and quotas",
                    "Verify network connectivity to data sources"
                ],
                resolution_steps=[
                    "1. Restart data pipeline flow: prefect deployment run data-pipeline",
                    "2. Check and refresh API keys if expired",
                    "3. Implement backfill for missing data period",
                    "4. Switch to backup data source if primary failing",
                    "5. Update data quality monitoring thresholds",
                    "6. Document data gap and impact on trading decisions"
                ],
                escalation_threshold_minutes=60,
                contact_on_call=False
            ),
            
            "ml_model_degradation": RunbookEntry(
                incident_id="ml_model_degradation",
                title="🤖 ML Model Performance Degradation",
                description="ML models showing poor performance or prediction quality",
                severity=IncidentSeverity.MEDIUM,
                symptoms=[
                    "ML Sharpe ratio below threshold",
                    "High prediction errors",
                    "Model drift alerts",
                    "Unusual feature distributions",
                    "Canary A/B test failing"
                ],
                diagnosis_steps=[
                    "Check ML weight status: mech-exo ml-weight --status",
                    "Review model performance: mech-exo scoring --analysis",
                    "Check feature drift: python -c 'from mech_exo.scoring.factors import check_feature_drift; print(check_feature_drift())'",
                    "Compare A/B test results: mech-exo ab-test --summary",
                    "Review recent prediction accuracy"
                ],
                resolution_steps=[
                    "1. Reduce ML weight to conservative level: mech-exo ml-weight --set 0.1",
                    "2. Retrain model with recent data if drift detected",
                    "3. Validate model on out-of-sample data",
                    "4. Gradually increase ML weight if performance improves",
                    "5. Consider rolling back to previous model version",
                    "6. Update model monitoring thresholds"
                ],
                escalation_threshold_minutes=120,
                contact_on_call=False
            ),
            
            "high_latency": RunbookEntry(
                incident_id="high_latency",
                title="🐌 High Latency / Performance Degradation",
                description="System response times significantly slower than normal",
                severity=IncidentSeverity.MEDIUM,
                symptoms=[
                    "Dashboard loading slowly",
                    "Order execution delays",
                    "Database query timeouts",
                    "High CPU or memory usage",
                    "Network connectivity issues"
                ],
                diagnosis_steps=[
                    "Check system resources: top, htop, iostat",
                    "Monitor database performance: EXPLAIN QUERY PLAN for slow queries",
                    "Check network latency: ping external APIs",
                    "Review application logs for timeouts",
                    "Check for memory leaks: ps aux --sort=-%mem"
                ],
                resolution_steps=[
                    "1. Identify resource bottlenecks (CPU, memory, disk, network)",
                    "2. Kill any runaway processes consuming resources",
                    "3. Restart services with high memory usage",
                    "4. Optimize slow database queries with indexes",
                    "5. Scale horizontally if needed (add more workers)",
                    "6. Implement query caching for frequent operations"
                ],
                escalation_threshold_minutes=45,
                contact_on_call=False
            ),
            
            "disk_space_full": RunbookEntry(
                incident_id="disk_space_full",
                title="💽 Disk Space Full / Storage Issues",
                description="Server running out of disk space affecting operations",
                severity=IncidentSeverity.HIGH,
                symptoms=[
                    "Cannot write new log files",
                    "Database write failures",
                    "Backup creation failing",
                    "Temporary file creation errors",
                    "System warnings about disk space"
                ],
                diagnosis_steps=[
                    "Check disk usage: df -h",
                    "Find largest files: du -h /path/to/mech-exo | sort -rh | head -20",
                    "Check log file sizes: ls -lah logs/",
                    "Review database and backup sizes",
                    "Check for core dumps or temp files"
                ],
                resolution_steps=[
                    "1. Archive and compress old log files: gzip logs/*.log.old",
                    "2. Delete old backup files (keep last 7 days)",
                    "3. Clean up temporary files in /tmp",
                    "4. Rotate current log files: logrotate -f /etc/logrotate.conf",
                    "5. Move large files to external storage if needed",
                    "6. Set up automated cleanup scripts for future"
                ],
                escalation_threshold_minutes=20,
                contact_on_call=True
            ),
            
            "prefect_flow_failures": RunbookEntry(
                incident_id="prefect_flow_failures",
                title="🔄 Prefect Flow Failures / Orchestration Issues",
                description="Workflow orchestration system having issues",
                severity=IncidentSeverity.MEDIUM,
                symptoms=[
                    "Multiple flows failing consistently",
                    "Flows not starting at scheduled times",
                    "Agent disconnected from server",
                    "Flow runs stuck in pending state",
                    "Resource pool issues"
                ],
                diagnosis_steps=[
                    "Check agent status: prefect agent ls",
                    "Review failed flows: prefect flow-run ls --state Failed",
                    "Check Prefect server connectivity: prefect profile show",
                    "Review flow logs: prefect flow-run logs <flow-run-id>",
                    "Check resource pool availability"
                ],
                resolution_steps=[
                    "1. Restart Prefect agent: prefect agent start --pool default",
                    "2. Retry failed flow runs: prefect flow-run retry <flow-run-id>",
                    "3. Check and update flow deployments if needed",
                    "4. Clear stuck runs from queue",
                    "5. Verify network connectivity to Prefect Cloud/Server",
                    "6. Update agent configuration if needed"
                ],
                escalation_threshold_minutes=30,
                contact_on_call=False
            ),
            
            "api_rate_limits": RunbookEntry(
                incident_id="api_rate_limits",
                title="🚦 External API Rate Limits / Access Issues",
                description="External data provider APIs hitting rate limits or access issues",
                severity=IncidentSeverity.LOW,
                symptoms=[
                    "HTTP 429 errors from data APIs",
                    "Authentication failures",
                    "Quota exceeded messages",
                    "Data updates falling behind",
                    "Empty responses from APIs"
                ],
                diagnosis_steps=[
                    "Check API response logs for error codes",
                    "Review current API usage quotas",
                    "Test API credentials: curl with auth headers",
                    "Check rate limit headers in responses",
                    "Verify API subscription status"
                ],
                resolution_steps=[
                    "1. Implement exponential backoff for retries",
                    "2. Switch to backup API provider if available",
                    "3. Reduce polling frequency temporarily",
                    "4. Cache API responses more aggressively",
                    "5. Contact API provider about quota increases",
                    "6. Implement request queuing and throttling"
                ],
                escalation_threshold_minutes=90,
                contact_on_call=False,
                auto_remediation="enable_api_fallback"
            )
        }
        
        return entries
    
    def _load_escalation_rules(self) -> List[EscalationRule]:
        """Load escalation rules configuration"""
        return [
            EscalationRule(
                level=EscalationLevel.TELEGRAM,
                delay_minutes=0,  # Immediate
                enabled=True,
                quiet_hours_override=False
            ),
            EscalationRule(
                level=EscalationLevel.EMAIL,
                delay_minutes=15,  # 15 minutes after initial alert
                enabled=True,
                quiet_hours_override=False
            ),
            EscalationRule(
                level=EscalationLevel.PHONE,
                delay_minutes=30,  # 30 minutes after initial alert
                enabled=False,  # Placeholder - not implemented
                quiet_hours_override=True  # Phone calls override quiet hours
            )
        ]
    
    def _load_quiet_hours(self) -> Dict[str, time]:
        """Load quiet hours configuration (22:00-06:00 local)"""
        return {
            'start': time(22, 0),  # 22:00
            'end': time(6, 0)      # 06:00
        }
    
    def get_runbook_entry(self, incident_id: str) -> Optional[RunbookEntry]:
        """Get specific runbook entry"""
        return self.runbook_entries.get(incident_id)
    
    def list_incidents(self) -> List[RunbookEntry]:
        """List all runbook entries sorted by severity"""
        entries = list(self.runbook_entries.values())
        severity_order = {
            IncidentSeverity.CRITICAL: 0,
            IncidentSeverity.HIGH: 1,
            IncidentSeverity.MEDIUM: 2,
            IncidentSeverity.LOW: 3
        }
        return sorted(entries, key=lambda x: severity_order[x.severity])
    
    def should_escalate(self, incident_start_time: datetime, severity: IncidentSeverity) -> List[EscalationLevel]:
        """Determine which escalation levels should be triggered"""
        current_time = datetime.now()
        elapsed_minutes = (current_time - incident_start_time).total_seconds() / 60
        
        # Check if we're in quiet hours
        in_quiet_hours = self._is_quiet_hours(current_time.time())
        
        escalations_due = []
        
        for rule in self.escalation_rules:
            if not rule.enabled:
                continue
                
            # Skip if in quiet hours and rule doesn't override
            if in_quiet_hours and not rule.quiet_hours_override:
                continue
                
            # Check if enough time has passed for this escalation level
            if elapsed_minutes >= rule.delay_minutes:
                escalations_due.append(rule.level)
        
        return escalations_due
    
    def _is_quiet_hours(self, current_time: time) -> bool:
        """Check if current time is within quiet hours"""
        start = self.quiet_hours['start']
        end = self.quiet_hours['end']
        
        if start <= end:
            # Same day range (e.g., 22:00-23:59)
            return start <= current_time <= end
        else:
            # Crosses midnight (e.g., 22:00-06:00)
            return current_time >= start or current_time <= end
    
    def format_runbook_entry(self, entry: RunbookEntry) -> str:
        """Format runbook entry as readable text"""
        
        text = f"""
{entry.title}
{'=' * len(entry.title)}

**Severity:** {entry.severity.value.upper()}
**Description:** {entry.description}

**Symptoms:**
{chr(10).join(f"• {symptom}" for symptom in entry.symptoms)}

**Diagnosis Steps:**
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(entry.diagnosis_steps))}

**Resolution Steps:**
{chr(10).join(step for step in entry.resolution_steps)}

**Escalation:** Contact on-call after {entry.escalation_threshold_minutes} minutes
**Auto-remediation:** {entry.auto_remediation or 'None available'}

---
"""
        
        return text
    
    def export_runbook(self, output_path: str) -> bool:
        """Export complete runbook to markdown file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Mech-Exo On-Call Runbook\n\n")
                f.write("*Comprehensive incident response guide for operational issues*\n\n")
                f.write("## Quick Reference\n\n")
                
                # Table of contents
                f.write("| Incident Type | Severity | Escalation Time |\n")
                f.write("|---------------|----------|----------------|\n")
                
                for entry in self.list_incidents():
                    f.write(f"| [{entry.title}](#{entry.incident_id.replace('_', '-')}) | {entry.severity.value} | {entry.escalation_threshold_minutes}min |\n")
                
                f.write("\n## Escalation Rules\n\n")
                f.write("**Quiet Hours:** 22:00-06:00 local time\n\n")
                f.write("1. **Telegram** (immediate) - Always sent\n")
                f.write("2. **Email** (15min delay) - Suppressed during quiet hours\n")
                f.write("3. **Phone** (30min delay) - Override quiet hours for critical issues\n\n")
                
                f.write("## Incident Procedures\n\n")
                
                # Write detailed entries
                for entry in self.list_incidents():
                    f.write(self.format_runbook_entry(entry))
                
                f.write("\n## Emergency Contacts\n\n")
                f.write("- **Primary On-Call:** Set via environment variable `ONCALL_PRIMARY`\n")
                f.write("- **Secondary On-Call:** Set via environment variable `ONCALL_SECONDARY`\n")
                f.write("- **Escalation Manager:** Set via environment variable `ONCALL_MANAGER`\n\n")
                
                f.write("## Quick Commands\n\n")
                f.write("```bash\n")
                f.write("# System health check\n")
                f.write("curl -H 'Accept: application/json' http://localhost:8050/healthz\n\n")
                f.write("# Emergency rollback\n")
                f.write("mech-exo rollback --flow daily_flow --target '2025-06-15T09:00:00'\n\n")
                f.write("# Risk status\n")
                f.write("mech-exo risk --status\n\n")
                f.write("# Recent fills\n")
                f.write("mech-exo fills --days 1\n\n")
                f.write("# View logs\n")
                f.write("tail -f logs/mech-exo.log\n")
                f.write("```\n")
                
            logger.info(f"Runbook exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export runbook: {e}")
            return False
    
    def trigger_escalation(self, incident_id: str, incident_start_time: datetime, 
                          escalation_level: EscalationLevel, details: str = "") -> bool:
        """Trigger escalation for an incident"""
        try:
            entry = self.get_runbook_entry(incident_id)
            if not entry:
                logger.error(f"Unknown incident ID: {incident_id}")
                return False
            
            # Format escalation message
            elapsed_minutes = (datetime.now() - incident_start_time).total_seconds() / 60
            
            message = f"""
🚨 ESCALATION: {entry.title}

**Severity:** {entry.severity.value.upper()}
**Duration:** {elapsed_minutes:.0f} minutes
**Level:** {escalation_level.value.upper()}

**Details:** {details}

**Next Steps:**
{chr(10).join(entry.resolution_steps[:3])}

See full runbook entry: {incident_id}
"""
            
            if escalation_level == EscalationLevel.TELEGRAM:
                return self._send_telegram_escalation(message)
            elif escalation_level == EscalationLevel.EMAIL:
                return self._send_email_escalation(entry.title, message)
            elif escalation_level == EscalationLevel.PHONE:
                return self._send_phone_escalation(entry.title, message)
            
            return False
            
        except Exception as e:
            logger.error(f"Escalation failed: {e}")
            return False
    
    def _send_telegram_escalation(self, message: str) -> bool:
        """Send Telegram escalation (integrate with alerts system)"""
        try:
            from ..utils.alerts import AlertManager, Alert, AlertType, AlertLevel
            
            alert_manager = AlertManager()
            
            alert = Alert(
                alert_type=AlertType.SYSTEM_ALERT,
                level=AlertLevel.CRITICAL,
                title="🚨 On-Call Escalation",
                message=message,
                timestamp=datetime.now(),
                data={'escalation_level': 'telegram', 'runbook': True}
            )
            
            return alert_manager.send_alert(alert, channels=['telegram'])
            
        except Exception as e:
            logger.error(f"Telegram escalation failed: {e}")
            return False
    
    def _send_email_escalation(self, subject: str, message: str) -> bool:
        """Send email escalation (placeholder)"""
        logger.info(f"EMAIL ESCALATION: {subject}")
        logger.info(message)
        # TODO: Implement actual email sending
        return True
    
    def _send_phone_escalation(self, subject: str, message: str) -> bool:
        """Send phone escalation (placeholder)"""
        logger.critical(f"PHONE ESCALATION: {subject}")
        logger.critical(message)
        # TODO: Implement actual phone calling service
        return True


def get_runbook() -> OnCallRunbook:
    """Get runbook instance"""
    return OnCallRunbook()


def cli_runbook_export(output_path: str = "oncall_runbook.md") -> bool:
    """CLI command to export runbook"""
    runbook = get_runbook()
    return runbook.export_runbook(output_path)


def cli_incident_lookup(incident_id: str) -> Optional[str]:
    """CLI command to lookup incident by ID"""
    runbook = get_runbook()
    entry = runbook.get_runbook_entry(incident_id)
    
    if entry:
        return runbook.format_runbook_entry(entry)
    else:
        available_ids = list(runbook.runbook_entries.keys())
        return f"Incident '{incident_id}' not found. Available: {', '.join(available_ids)}"


if __name__ == "__main__":
    # Export runbook
    runbook = OnCallRunbook()
    success = runbook.export_runbook("oncall_runbook.md")
    print(f"Runbook export: {'Success' if success else 'Failed'}")
    
    # List all incidents
    print("\nAvailable Incidents:")
    for entry in runbook.list_incidents():
        print(f"- {entry.incident_id}: {entry.title} ({entry.severity.value})")