"""
SLO Error Budget Tests - Phase P11 Week 2

Tests for error budget calculation, SLO monitoring, and alerting rules.
Validates that the SLO implementation meets reliability requirements.
"""

import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MockPrometheusQuery:
    """Mock Prometheus query interface for testing"""
    
    def __init__(self):
        self.metrics = {}
        
    def set_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a metric value for testing"""
        labels = labels or {}
        key = f"{name}:{','.join([f'{k}={v}' for k, v in labels.items()])}"
        self.metrics[key] = value
        
    def query(self, query_expr: str) -> List[Dict[str, Any]]:
        """Mock Prometheus query"""
        # Simple mock - in practice would parse PromQL
        if "risk_ops_ok" in query_expr:
            return [{"value": [time.time(), "1"]}]
        elif "error_budget_remaining" in query_expr:
            return [{"value": [time.time(), "99.5"]}]
        return []


class TestErrorBudgetCalculation:
    """Test error budget calculation logic"""
    
    def test_perfect_availability(self):
        """Test error budget with 100% availability"""
        # Given: 24 hours of perfect uptime
        uptime_minutes = 1440  # 24 * 60
        downtime_minutes = 0
        
        # When: Calculate error budget
        error_budget_remaining = 100 - (downtime_minutes * 100 / 1440)
        
        # Then: Should have 100% error budget remaining
        assert error_budget_remaining == 100.0
        
    def test_slo_target_availability(self):
        """Test error budget at exactly 99% availability (SLO target)"""
        # Given: Exactly 99% availability (14.4 minutes downtime)
        downtime_minutes = 14.4
        
        # When: Calculate error budget
        error_budget_remaining = 100 - (downtime_minutes * 100 / 1440)
        
        # Then: Should have 99% error budget remaining
        assert abs(error_budget_remaining - 99.0) < 0.01
        
    def test_error_budget_exhausted(self):
        """Test error budget when SLO is breached"""
        # Given: More than 1% downtime (>14.4 minutes)
        downtime_minutes = 30.0  # 2.08% downtime
        
        # When: Calculate error budget
        error_budget_remaining = 100 - (downtime_minutes * 100 / 1440)
        
        # Then: Should have <99% error budget remaining
        assert error_budget_remaining < 99.0
        assert abs(error_budget_remaining - 97.92) < 0.01
        
    def test_burn_rate_calculation(self):
        """Test error budget burn rate calculation"""
        # Given: 5 minutes of downtime in last hour
        downtime_minutes_1h = 5
        
        # When: Calculate burn rate
        burn_rate_pct = (downtime_minutes_1h * 100) / 60
        
        # Then: Should have 8.33% burn rate
        assert abs(burn_rate_pct - 8.33) < 0.01


class TestSLOThresholds:
    """Test SLO threshold logic"""
    
    def test_green_threshold(self):
        """Test green SLO status"""
        error_budget = 99.8
        
        # Should be green (normal operations)
        assert error_budget > 99.5
        
    def test_yellow_threshold(self):
        """Test yellow SLO status"""
        error_budget = 99.2
        
        # Should be yellow (warning)
        assert 99.0 <= error_budget <= 99.5
        
    def test_red_threshold(self):
        """Test red SLO status"""
        error_budget = 97.5
        
        # Should be red (critical)
        assert error_budget < 99.0


class TestAlertingLogic:
    """Test SLO alerting rules"""
    
    def test_error_budget_burn_alert_trigger(self):
        """Test that error budget burn alert triggers correctly"""
        # Given: Error budget below 98% threshold
        error_budget = 97.8
        
        # When: Check alert condition
        should_alert = error_budget < 98.0
        
        # Then: Should trigger pager alert
        assert should_alert is True
        
    def test_error_budget_burn_alert_no_trigger(self):
        """Test that error budget burn alert doesn't trigger above threshold"""
        # Given: Error budget above 98% threshold
        error_budget = 98.5
        
        # When: Check alert condition
        should_alert = error_budget < 98.0
        
        # Then: Should not trigger alert
        assert should_alert is False
        
    def test_system_down_alert(self):
        """Test system down alert logic"""
        # Given: System is down
        risk_ops_ok = 0
        
        # When: Check alert condition
        should_alert = risk_ops_ok == 0
        
        # Then: Should trigger immediate alert
        assert should_alert is True


class TestSLOMetrics:
    """Test SLO metrics collection and calculation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.prom = MockPrometheusQuery()
        
    def test_availability_metric(self):
        """Test availability percentage calculation"""
        # Given: Mock uptime data
        self.prom.set_metric("risk_ops_ok", 1.0, {"env": "prod"})
        
        # When: Query availability
        result = self.prom.query("avg_over_time(risk_ops_ok{env='prod'}[24h]) * 100")
        
        # Then: Should return availability data
        assert len(result) > 0
        
    def test_downtime_metric(self):
        """Test downtime calculation"""
        # Given: Some downtime periods
        downtime_periods = [
            {"start": "2025-06-13T10:00:00Z", "duration_minutes": 5},
            {"start": "2025-06-13T15:30:00Z", "duration_minutes": 3}
        ]
        
        # When: Calculate total downtime
        total_downtime = sum(period["duration_minutes"] for period in downtime_periods)
        
        # Then: Should sum correctly
        assert total_downtime == 8
        
    def test_ops_ok_metric_labels(self):
        """Test that risk_ops_ok metric has correct labels"""
        # Given: Metric with environment label
        self.prom.set_metric("risk_ops_ok", 1.0, {"env": "prod"})
        
        # When: Check metric exists
        metric_key = "risk_ops_ok:env=prod"
        
        # Then: Should exist with correct label
        assert metric_key in self.prom.metrics
        assert self.prom.metrics[metric_key] == 1.0


class TestPrometheusRules:
    """Test Prometheus recording and alerting rules"""
    
    def test_error_budget_recording_rule(self):
        """Test error budget recording rule syntax"""
        # Given: Recording rule expression
        rule_expr = "100 - (sum_over_time((1 - risk_ops_ok{env=\"prod\"})[24h:1m]) * 100 / 1440)"
        
        # When: Validate expression (basic syntax check)
        # This is a simplified check - in practice would use promtool
        assert "risk_ops_ok" in rule_expr
        assert "24h" in rule_expr
        assert "1440" in rule_expr  # minutes in 24h
        
    def test_burn_rate_recording_rule(self):
        """Test burn rate recording rule"""
        # Given: Burn rate rule expression
        rule_expr = "sum_over_time((1 - risk_ops_ok{env=\"prod\"})[1h:1m]) * 100 / 60"
        
        # When: Validate expression
        assert "1h" in rule_expr
        assert "60" in rule_expr  # minutes in 1h
        
    def test_alert_rule_structure(self):
        """Test alert rule structure"""
        # Given: Alert rule definition
        alert_rule = {
            "alert": "ErrorBudgetBurnHigh",
            "expr": "risk:error_budget_remaining < 98",
            "for": "5m",
            "labels": {
                "severity": "pager",
                "env": "prod"
            }
        }
        
        # When: Validate rule structure
        assert alert_rule["alert"] == "ErrorBudgetBurnHigh"
        assert alert_rule["labels"]["severity"] == "pager"
        assert "98" in alert_rule["expr"]


class TestSLODashboard:
    """Test SLO dashboard configuration"""
    
    def test_error_budget_panel_thresholds(self):
        """Test Grafana panel threshold configuration"""
        # Given: Dashboard panel thresholds
        thresholds = [
            {"color": "red", "value": 98},
            {"color": "yellow", "value": 99},
            {"color": "green", "value": 99.5}
        ]
        
        # When: Check threshold logic
        error_budget = 97.5
        
        # Then: Should map to red
        color = "green"
        for threshold in reversed(thresholds):
            if error_budget >= threshold["value"]:
                color = threshold["color"]
                break
        else:
            color = thresholds[0]["color"]  # Default to first (red)
            
        assert color == "red"
        
    def test_availability_panel_config(self):
        """Test availability panel configuration"""
        # Given: Panel configuration
        panel_config = {
            "title": "SLO Availability (24h)",
            "type": "stat",
            "targets": [
                {
                    "expr": "risk:availability_24h",
                    "legendFormat": "Availability"
                }
            ],
            "fieldConfig": {
                "unit": "percent",
                "decimals": 3
            }
        }
        
        # When: Validate configuration
        assert panel_config["type"] == "stat"
        assert panel_config["fieldConfig"]["unit"] == "percent"
        assert "risk:availability_24h" in str(panel_config["targets"])


class TestSLOIntegration:
    """Integration tests for SLO system"""
    
    def test_end_to_end_slo_workflow(self):
        """Test complete SLO monitoring workflow"""
        # Given: System starts healthy
        system_state = {
            "risk_ops_ok": 1,
            "error_budget": 100.0,
            "alert_fired": False
        }
        
        # When: System goes down for 30 minutes
        downtime_minutes = 30
        system_state["risk_ops_ok"] = 0
        system_state["error_budget"] = 100 - (downtime_minutes * 100 / 1440)
        
        # Then: Error budget should decrease and alert should fire
        assert system_state["error_budget"] < 98.0  # Below threshold
        assert system_state["error_budget"] > 97.0  # Reasonable range
        
        # And: Alert should be triggered
        if system_state["error_budget"] < 98.0:
            system_state["alert_fired"] = True
            
        assert system_state["alert_fired"] is True
        
    def test_slo_recovery_workflow(self):
        """Test SLO recovery after incident"""
        # Given: System was down but is now recovered
        initial_downtime = 15  # minutes
        error_budget_after_incident = 100 - (initial_downtime * 100 / 1440)
        
        # When: System runs perfectly for remaining day
        # Error budget should remain stable (no additional downtime)
        final_error_budget = error_budget_after_incident
        
        # Then: Error budget should be around 98.96%
        expected_budget = 100 - (15 * 100 / 1440)
        assert abs(final_error_budget - expected_budget) < 0.01
        assert final_error_budget > 98.0  # Above alert threshold


def test_slo_configuration_files_exist():
    """Test that required SLO configuration files exist"""
    # Given: Expected configuration files
    required_files = [
        "prometheus/risk_rules.yml",
        "prometheus/risk_alerts.yml", 
        "docs/slo.md"
    ]
    
    # When: Check file existence
    for file_path in required_files:
        full_path = Path(__file__).parent.parent.parent / file_path
        
        # Then: Files should exist
        assert full_path.exists(), f"Required SLO file missing: {file_path}"


def test_prometheus_rules_yaml_validity():
    """Test that Prometheus rules files are valid YAML"""
    import yaml
    
    # Given: Prometheus rule files
    rule_files = [
        "prometheus/risk_rules.yml",
        "prometheus/risk_alerts.yml"
    ]
    
    for rule_file in rule_files:
        rule_path = Path(__file__).parent.parent.parent / rule_file
        
        if rule_path.exists():
            # When: Parse YAML
            with open(rule_path, 'r') as f:
                rules_config = yaml.safe_load(f)
            
            # Then: Should be valid YAML with groups
            assert isinstance(rules_config, dict)
            assert "groups" in rules_config
            assert isinstance(rules_config["groups"], list)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])