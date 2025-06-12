"""
Unified Operations Dashboard

Provides a comprehensive overview of system health, alerts, flow status,
CI badges, and operational metrics for on-call monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

logger = logging.getLogger(__name__)


def create_ops_overview_layout():
    """Create the unified operations dashboard layout"""
    
    return html.Div([
        # Header with last update timestamp
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-tachometer-alt me-2"),
                    "Operations Dashboard"
                ], className="text-primary"),
                html.P("Unified system health and operational monitoring", className="text-muted"),
            ], width=8),
            dbc.Col([
                html.Div(id="ops-last-updated", className="text-end"),
                dbc.Badge("Auto-refresh: 5min", color="info", className="mt-2")
            ], width=4)
        ], className="mb-4"),
        
        # System Status Cards Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("System Health", className="card-title"),
                        html.Div(id="system-health-status"),
                        html.Small(id="health-last-check", className="text-muted")
                    ])
                ], color="light", className="h-100")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Alert Status", className="card-title"),
                        html.Div(id="alert-summary"),
                        html.Small(id="alert-last-time", className="text-muted")
                    ])
                ], color="light", className="h-100")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Flow Status", className="card-title"),
                        html.Div(id="flow-status-summary"),
                        html.Small(id="flow-last-run", className="text-muted")
                    ])
                ], color="light", className="h-100")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("CI Status", className="card-title"),
                        html.Div(id="ci-status-badges"),
                        html.Small(id="ci-last-build", className="text-muted")
                    ])
                ], color="light", className="h-100")
            ], width=3)
        ], className="mb-4"),
        
        # Capital Utilization Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Capital Utilization", className="card-title"),
                        html.Div(id="capital-utilization-display"),
                        html.Small(id="capital-last-check", className="text-muted")
                    ])
                ], color="light", className="h-100")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Account Status", className="card-title"),
                        html.Div(id="account-status-summary"),
                        html.Small(id="account-last-update", className="text-muted")
                    ])
                ], color="light", className="h-100")
            ], width=6)
        ], className="mb-4"),
        
        # Detailed Sections in Accordion
        dbc.Accordion([
            # Recent Alerts Section
            dbc.AccordionItem([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Recent Alerts (24h)"),
                            html.Div(id="recent-alerts-table")
                        ], width=8),
                        dbc.Col([
                            html.H5("Alert Distribution"),
                            dcc.Graph(id="alert-distribution-chart", style={"height": "300px"})
                        ], width=4)
                    ])
                ])
            ], title="🚨 Alert History & Analysis", item_id="alerts"),
            
            # Flow Run Details Section
            dbc.AccordionItem([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Prefect Flow Runs (Last 7 Days)"),
                            html.Div(id="flow-runs-table")
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Flow Success Rate"),
                            dcc.Graph(id="flow-success-chart", style={"height": "300px"})
                        ], width=6),
                        dbc.Col([
                            html.H5("Flow Duration Trends"),
                            dcc.Graph(id="flow-duration-chart", style={"height": "300px"})
                        ], width=6)
                    ], className="mt-3")
                ])
            ], title="🔄 Flow Status & Performance", item_id="flows"),
            
            # System Resources Section
            dbc.AccordionItem([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Resource Monitoring"),
                            html.P("Grafana dashboard embedded below (if available):", className="text-muted"),
                            html.Div(id="grafana-iframe-container")
                        ], width=8),
                        dbc.Col([
                            html.H5("Quick Metrics"),
                            html.Div(id="resource-metrics")
                        ], width=4)
                    ])
                ])
            ], title="📊 System Resources", item_id="resources"),
            
            # Risk & Trading Status Section
            dbc.AccordionItem([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Current Risk Breaches"),
                            html.Div(id="risk-breaches-table")
                        ], width=6),
                        dbc.Col([
                            html.H5("Trading Status"),
                            html.Div(id="trading-status-cards")
                        ], width=6)
                    ])
                ])
            ], title="⚠️ Risk & Trading Status", item_id="risk")
        ], start_collapsed=True, className="mb-4"),
        
        # Action Buttons Row
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button([
                        html.I(className="fas fa-sync me-2"),
                        "Refresh Now"
                    ], id="manual-refresh-btn", color="primary", outline=True),
                    dbc.Button([
                        html.I(className="fas fa-download me-2"),
                        "Export Report"
                    ], id="export-ops-report", color="secondary", outline=True),
                    dbc.Button([
                        html.I(className="fas fa-external-link-alt me-2"),
                        "Open Prefect UI"
                    ], id="open-prefect-ui", color="info", outline=True, external_link=True),
                ])
            ], width="auto"),
            dbc.Col([
                html.Div(id="ops-action-feedback", className="text-end")
            ])
        ], justify="between", className="mt-4")
    ])


def register_ops_overview_callbacks():
    """Register callbacks for the ops overview dashboard"""
    
    @callback(
        [
            Output('ops-last-updated', 'children'),
            Output('system-health-status', 'children'),
            Output('health-last-check', 'children'),
            Output('alert-summary', 'children'),
            Output('alert-last-time', 'children'),
            Output('flow-status-summary', 'children'),
            Output('flow-last-run', 'children'),
            Output('ci-status-badges', 'children'),
            Output('ci-last-build', 'children'),
            Output('capital-utilization-display', 'children'),
            Output('capital-last-check', 'children'),
            Output('account-status-summary', 'children'),
            Output('account-last-update', 'children')
        ],
        [
            Input('interval-ops', 'n_intervals'),
            Input('manual-refresh-btn', 'n_clicks')
        ]
    )
    def update_ops_overview(n_intervals, manual_clicks):
        """Update main ops overview cards"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            last_updated = html.Small(f"Last updated: {current_time}", className="text-muted")
            
            # Get system health
            health_data = _get_system_health()
            health_status = _render_health_status(health_data)
            health_check_time = health_data.get('last_check', 'Unknown')
            
            # Get alert summary
            alert_data = _get_alert_summary()
            alert_summary = _render_alert_summary(alert_data)
            alert_time = alert_data.get('last_alert_time', 'No recent alerts')
            
            # Get flow status
            flow_data = _get_flow_status()
            flow_summary = _render_flow_summary(flow_data)
            flow_time = flow_data.get('last_run_time', 'No recent runs')
            
            # Get CI status
            ci_data = _get_ci_status()
            ci_badges = _render_ci_badges(ci_data)
            ci_time = ci_data.get('last_build_time', 'Unknown')
            
            # Get capital utilization data
            capital_data = _get_capital_utilization()
            capital_display = _render_capital_utilization(capital_data)
            capital_time = capital_data.get('last_check', 'No recent data')
            
            # Get account status
            account_data = _get_account_status()
            account_summary = _render_account_status(account_data)
            account_time = account_data.get('last_update', 'Unknown')
            
            return (
                last_updated,
                health_status,
                health_check_time,
                alert_summary,
                alert_time,
                flow_summary,
                flow_time,
                ci_badges,
                ci_time,
                capital_display,
                capital_time,
                account_summary,
                account_time
            )
            
        except Exception as e:
            logger.error(f"Failed to update ops overview: {e}")
            error_msg = html.Span("⚠️ Error loading data", className="text-danger")
            return (error_msg,) * 13  # Updated to match new output count
    
    @callback(
        Output('recent-alerts-table', 'children'),
        Output('alert-distribution-chart', 'figure'),
        Input('interval-ops', 'n_intervals')
    )
    def update_alerts_section(n_intervals):
        """Update recent alerts table and distribution chart"""
        try:
            alerts_data = _get_recent_alerts(hours=24)
            
            # Create alerts table
            alerts_table = _create_alerts_table(alerts_data)
            
            # Create distribution chart
            distribution_chart = _create_alert_distribution_chart(alerts_data)
            
            return alerts_table, distribution_chart
            
        except Exception as e:
            logger.error(f"Failed to update alerts section: {e}")
            return html.P("Error loading alerts data", className="text-danger"), {}
    
    @callback(
        [
            Output('flow-runs-table', 'children'),
            Output('flow-success-chart', 'figure'),
            Output('flow-duration-chart', 'figure')
        ],
        Input('interval-ops', 'n_intervals')
    )
    def update_flows_section(n_intervals):
        """Update flow runs table and charts"""
        try:
            flows_data = _get_flow_runs(days=7)
            
            # Create flows table
            flows_table = _create_flows_table(flows_data)
            
            # Create success rate chart
            success_chart = _create_flow_success_chart(flows_data)
            
            # Create duration chart
            duration_chart = _create_flow_duration_chart(flows_data)
            
            return flows_table, success_chart, duration_chart
            
        except Exception as e:
            logger.error(f"Failed to update flows section: {e}")
            return html.P("Error loading flows data", className="text-danger"), {}, {}
    
    @callback(
        [
            Output('grafana-iframe-container', 'children'),
            Output('resource-metrics', 'children')
        ],
        Input('interval-ops', 'n_intervals')
    )
    def update_resources_section(n_intervals):
        """Update system resources section"""
        try:
            # Check if Grafana is available
            grafana_url = _get_grafana_url()
            if grafana_url:
                grafana_iframe = html.Iframe(
                    src=grafana_url,
                    style={
                        "width": "100%",
                        "height": "400px",
                        "border": "1px solid #ddd",
                        "border-radius": "5px"
                    }
                )
            else:
                grafana_iframe = dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    "Grafana dashboard not configured. Set GRAFANA_URL environment variable."
                ], color="info")
            
            # Get quick metrics
            metrics = _get_resource_metrics()
            metrics_display = _render_resource_metrics(metrics)
            
            return grafana_iframe, metrics_display
            
        except Exception as e:
            logger.error(f"Failed to update resources section: {e}")
            return html.P("Error loading resources", className="text-danger"), html.P("Error", className="text-danger")
    
    @callback(
        [
            Output('risk-breaches-table', 'children'),
            Output('trading-status-cards', 'children')
        ],
        Input('interval-ops', 'n_intervals')
    )
    def update_risk_section(n_intervals):
        """Update risk and trading status section"""
        try:
            # Get risk breaches
            risk_data = _get_risk_breaches()
            risk_table = _create_risk_breaches_table(risk_data)
            
            # Get trading status
            trading_data = _get_trading_status()
            trading_cards = _render_trading_status(trading_data)
            
            return risk_table, trading_cards
            
        except Exception as e:
            logger.error(f"Failed to update risk section: {e}")
            return html.P("Error loading risk data", className="text-danger"), html.P("Error", className="text-danger")


# Helper functions for data retrieval and rendering

def _get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    try:
        from ..query import get_health_data
        health_data = get_health_data()
        
        return {
            'status': health_data.get('system_status', 'unknown'),
            'risk_ok': health_data.get('risk_ok', False),
            'fills_today': health_data.get('fills_today', 0),
            'canary_enabled': health_data.get('canary_enabled', False),
            'ml_signal': health_data.get('ml_signal', False),
            'last_check': health_data.get('last_updated', 'Unknown'),
            'ops_ok': True  # Will be computed based on various factors
        }
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        return {'status': 'error', 'ops_ok': False, 'error': str(e)}


def _render_health_status(health_data: Dict[str, Any]) -> html.Div:
    """Render system health status card content"""
    status = health_data.get('status', 'unknown')
    ops_ok = health_data.get('ops_ok', False)
    
    if status == 'operational' and ops_ok:
        icon = "fas fa-check-circle text-success"
        status_text = "All Systems Operational"
        badge_color = "success"
    elif status == 'degraded':
        icon = "fas fa-exclamation-triangle text-warning"
        status_text = "System Degraded"
        badge_color = "warning"
    else:
        icon = "fas fa-times-circle text-danger"
        status_text = "System Error"
        badge_color = "danger"
    
    return html.Div([
        html.H4([
            html.I(className=icon + " me-2"),
            status_text
        ]),
        dbc.Badge(f"Fills today: {health_data.get('fills_today', 0)}", color="secondary", className="me-2"),
        dbc.Badge(f"ML: {'ON' if health_data.get('ml_signal') else 'OFF'}", color="info", className="me-2"),
        dbc.Badge(f"Canary: {'ON' if health_data.get('canary_enabled') else 'OFF'}", color="primary")
    ])


def _get_alert_summary() -> Dict[str, Any]:
    """Get alert summary for the last 24 hours"""
    try:
        # This would integrate with your alerting system
        # For now, return mock data
        return {
            'total_alerts': 3,
            'critical': 0,
            'warning': 2,
            'info': 1,
            'last_alert_time': '2 hours ago'
        }
    except Exception as e:
        logger.error(f"Failed to get alert summary: {e}")
        return {'total_alerts': 0, 'error': str(e)}


def _render_alert_summary(alert_data: Dict[str, Any]) -> html.Div:
    """Render alert summary card content"""
    total = alert_data.get('total_alerts', 0)
    critical = alert_data.get('critical', 0)
    warning = alert_data.get('warning', 0)
    info = alert_data.get('info', 0)
    
    if critical > 0:
        icon = "fas fa-exclamation-circle text-danger"
        summary_text = f"{critical} Critical Alert{'s' if critical != 1 else ''}"
    elif warning > 0:
        icon = "fas fa-exclamation-triangle text-warning"
        summary_text = f"{warning} Warning{'s' if warning != 1 else ''}"
    elif total > 0:
        icon = "fas fa-info-circle text-info"
        summary_text = f"{info} Info Alert{'s' if info != 1 else ''}"
    else:
        icon = "fas fa-check-circle text-success"
        summary_text = "No Active Alerts"
    
    return html.Div([
        html.H4([
            html.I(className=icon + " me-2"),
            summary_text
        ]),
        html.P(f"Total last 24h: {total}", className="mb-0")
    ])


def _get_flow_status() -> Dict[str, Any]:
    """Get Prefect flow status summary"""
    try:
        # This would integrate with Prefect API
        # For now, return mock data
        return {
            'total_runs': 12,
            'successful': 10,
            'failed': 1,
            'running': 1,
            'last_run_time': '30 minutes ago',
            'last_failure': 'canary_perf_flow - 6 hours ago'
        }
    except Exception as e:
        logger.error(f"Failed to get flow status: {e}")
        return {'total_runs': 0, 'error': str(e)}


def _render_flow_summary(flow_data: Dict[str, Any]) -> html.Div:
    """Render flow status summary"""
    total = flow_data.get('total_runs', 0)
    successful = flow_data.get('successful', 0)
    failed = flow_data.get('failed', 0)
    running = flow_data.get('running', 0)
    
    success_rate = (successful / total * 100) if total > 0 else 0
    
    if failed > 0:
        icon = "fas fa-exclamation-circle text-warning"
        status_text = f"{failed} Failed Run{'s' if failed != 1 else ''}"
    elif running > 0:
        icon = "fas fa-spinner fa-spin text-primary"
        status_text = f"{running} Running"
    else:
        icon = "fas fa-check-circle text-success"
        status_text = "All Flows OK"
    
    return html.Div([
        html.H4([
            html.I(className=icon + " me-2"),
            status_text
        ]),
        html.P(f"Success rate: {success_rate:.1f}% ({successful}/{total})", className="mb-0")
    ])


def _get_ci_status() -> Dict[str, Any]:
    """Get CI/CD status from GitHub or other CI system"""
    try:
        # This would integrate with GitHub Actions API
        # For now, return mock data
        return {
            'last_build_status': 'passed',
            'last_build_time': '2 hours ago',
            'branch': 'main',
            'commit': 'abc123f',
            'badges': [
                {'name': 'Build', 'status': 'passed', 'url': '#'},
                {'name': 'Tests', 'status': 'passed', 'url': '#'},
                {'name': 'A/B Smoke', 'status': 'passed', 'url': '#'},
                {'name': 'ML Weight CI', 'status': 'passed', 'url': '#'}
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get CI status: {e}")
        return {'last_build_status': 'unknown', 'error': str(e)}


def _render_ci_badges(ci_data: Dict[str, Any]) -> html.Div:
    """Render CI status badges"""
    badges = ci_data.get('badges', [])
    last_status = ci_data.get('last_build_status', 'unknown')
    
    badge_elements = []
    
    # Overall status
    if last_status == 'passed':
        overall_badge = dbc.Badge("✅ All Checks Passed", color="success", className="me-2 mb-1")
    elif last_status == 'failed':
        overall_badge = dbc.Badge("❌ Build Failed", color="danger", className="me-2 mb-1")
    else:
        overall_badge = dbc.Badge("⏳ Building", color="warning", className="me-2 mb-1")
    
    badge_elements.append(overall_badge)
    
    # Individual badges
    for badge in badges:
        color = "success" if badge['status'] == 'passed' else "danger" if badge['status'] == 'failed' else "warning"
        badge_element = dbc.Badge(
            badge['name'],
            color=color,
            className="me-1 mb-1",
            href=badge.get('url', '#'),
            external_link=True
        )
        badge_elements.append(badge_element)
    
    return html.Div(badge_elements)


def _get_grafana_url() -> Optional[str]:
    """Get Grafana dashboard URL if configured"""
    import os
    return os.getenv('GRAFANA_URL')


def _get_resource_metrics() -> Dict[str, Any]:
    """Get basic resource metrics"""
    import psutil
    import os
    
    try:
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_avg': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0,
            'uptime_hours': (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds() / 3600
        }
    except Exception as e:
        logger.error(f"Failed to get resource metrics: {e}")
        return {'error': str(e)}


def _render_resource_metrics(metrics: Dict[str, Any]) -> html.Div:
    """Render resource metrics display"""
    if 'error' in metrics:
        return html.P("Resource data unavailable", className="text-muted")
    
    return html.Div([
        html.P([
            html.Strong("CPU: "), f"{metrics.get('cpu_percent', 0):.1f}%"
        ], className="mb-1"),
        html.P([
            html.Strong("Memory: "), f"{metrics.get('memory_percent', 0):.1f}%"
        ], className="mb-1"),
        html.P([
            html.Strong("Disk: "), f"{metrics.get('disk_percent', 0):.1f}%"
        ], className="mb-1"),
        html.P([
            html.Strong("Load: "), f"{metrics.get('load_avg', 0):.2f}"
        ], className="mb-1"),
        html.P([
            html.Strong("Uptime: "), f"{metrics.get('uptime_hours', 0):.1f}h"
        ], className="mb-0")
    ])


# Mock functions for data that would come from real systems
def _get_recent_alerts(hours: int = 24) -> List[Dict]:
    """Get recent alerts (mock implementation)"""
    return []

def _create_alerts_table(alerts_data: List[Dict]) -> html.Div:
    """Create alerts table"""
    if not alerts_data:
        return html.P("No recent alerts", className="text-muted")
    return html.P("Alerts table would go here", className="text-muted")

def _create_alert_distribution_chart(alerts_data: List[Dict]) -> Dict:
    """Create alert distribution chart"""
    return {}

def _get_flow_runs(days: int = 7) -> List[Dict]:
    """Get flow runs data"""
    return []

def _create_flows_table(flows_data: List[Dict]) -> html.Div:
    """Create flows table"""
    return html.P("Flow runs table would go here", className="text-muted")

def _create_flow_success_chart(flows_data: List[Dict]) -> Dict:
    """Create flow success chart"""
    return {}

def _create_flow_duration_chart(flows_data: List[Dict]) -> Dict:
    """Create flow duration chart"""
    return {}

def _get_risk_breaches() -> List[Dict]:
    """Get current risk breaches"""
    return []

def _create_risk_breaches_table(risk_data: List[Dict]) -> html.Div:
    """Create risk breaches table"""
    return html.P("No active risk breaches", className="text-success")

def _get_trading_status() -> Dict[str, Any]:
    """Get current trading status"""
    return {'market_open': True, 'trading_enabled': True}

def _render_trading_status(trading_data: Dict[str, Any]) -> html.Div:
    """Render trading status cards"""
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H6("Market Status"),
                dbc.Badge("OPEN" if trading_data.get('market_open') else "CLOSED", 
                         color="success" if trading_data.get('market_open') else "secondary")
            ])
        ], className="mb-2"),
        dbc.Card([
            dbc.CardBody([
                html.H6("Trading"),
                dbc.Badge("ENABLED" if trading_data.get('trading_enabled') else "DISABLED",
                         color="success" if trading_data.get('trading_enabled') else "danger")
            ])
        ])
    ])


def _get_capital_utilization() -> Dict[str, Any]:
    """Get capital utilization data"""
    try:
        from ...cli.capital import CapitalManager
        
        manager = CapitalManager()
        config = manager.config
        
        # Get utilization data from configuration
        utilization_data = config.get('utilization', {})
        accounts_data = utilization_data.get('accounts', {})
        
        # Calculate aggregate statistics
        total_accounts = len(config.get('capital_limits', {}).get('accounts', {}))
        accounts_with_data = len(accounts_data)
        
        total_utilization = 0
        total_buying_power = 0
        total_max_capital = 0
        
        status_counts = {'ok': 0, 'warning': 0, 'critical': 0, 'error': 0}
        
        for account_id, account_data in accounts_data.items():
            total_buying_power += account_data.get('buying_power', 0)
            utilization_pct = account_data.get('utilization_pct', 0)
            total_utilization += utilization_pct
            
            # Count statuses
            status = account_data.get('status', 'error')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Get max capital from account config
            account_config = config.get('capital_limits', {}).get('accounts', {}).get(account_id, {})
            total_max_capital += account_config.get('max_capital', 0)
        
        avg_utilization = (total_utilization / accounts_with_data) if accounts_with_data > 0 else 0
        
        return {
            'total_accounts': total_accounts,
            'accounts_with_data': accounts_with_data,
            'avg_utilization_pct': avg_utilization,
            'total_buying_power': total_buying_power,
            'total_max_capital': total_max_capital,
            'status_counts': status_counts,
            'last_check': utilization_data.get('last_check', 'No recent data'),
            'accounts_data': accounts_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get capital utilization: {e}")
        return {
            'total_accounts': 0,
            'accounts_with_data': 0,
            'avg_utilization_pct': 0,
            'total_buying_power': 0,
            'total_max_capital': 0,
            'status_counts': {'ok': 0, 'warning': 0, 'critical': 0, 'error': 0},
            'last_check': 'Error loading data',
            'error': str(e)
        }


def _render_capital_utilization(capital_data: Dict[str, Any]) -> html.Div:
    """Render capital utilization display"""
    if 'error' in capital_data:
        return html.Div([
            html.P("❌ Error loading capital data", className="text-danger"),
            html.Small(capital_data['error'], className="text-muted")
        ])
    
    avg_utilization = capital_data.get('avg_utilization_pct', 0)
    status_counts = capital_data.get('status_counts', {})
    total_accounts = capital_data.get('total_accounts', 0)
    accounts_with_data = capital_data.get('accounts_with_data', 0)
    
    # Determine progress bar color based on utilization
    if avg_utilization >= 90:
        progress_color = "danger"
        icon = "fas fa-exclamation-triangle text-danger"
    elif avg_utilization >= 70:
        progress_color = "warning"
        icon = "fas fa-exclamation-triangle text-warning"
    else:
        progress_color = "success"
        icon = "fas fa-check-circle text-success"
    
    # Status summary
    critical_count = status_counts.get('critical', 0)
    warning_count = status_counts.get('warning', 0)
    
    if critical_count > 0:
        status_text = f"{critical_count} Critical Alert{'s' if critical_count != 1 else ''}"
        status_color = "danger"
    elif warning_count > 0:
        status_text = f"{warning_count} Warning{'s' if warning_count != 1 else ''}"
        status_color = "warning"
    else:
        status_text = "All Accounts OK"
        status_color = "success"
    
    return html.Div([
        html.H5([
            html.I(className=icon + " me-2"),
            status_text
        ]),
        
        # Progress bar for average utilization
        html.Div([
            html.Label(f"Average Utilization: {avg_utilization:.1f}%", className="mb-1"),
            dbc.Progress(
                value=avg_utilization,
                color=progress_color,
                striped=True,
                animated=avg_utilization > 70,
                className="mb-2"
            )
        ]),
        
        # Account summary badges
        html.Div([
            dbc.Badge(f"{accounts_with_data}/{total_accounts} Monitored", color="info", className="me-2"),
            dbc.Badge(f"${capital_data.get('total_buying_power', 0):,.0f} Available", color="secondary", className="me-2"),
            dbc.Badge(f"${capital_data.get('total_max_capital', 0):,.0f} Total Limit", color="primary", className="me-2"),
            _render_greylist_badge()
        ])
    ])


def _get_account_status() -> Dict[str, Any]:
    """Get account status summary"""
    try:
        from ...cli.capital import CapitalManager
        
        manager = CapitalManager()
        config = manager.config
        
        accounts_config = config.get('capital_limits', {}).get('accounts', {})
        utilization_data = config.get('utilization', {})
        accounts_data = utilization_data.get('accounts', {})
        
        account_summaries = []
        
        for account_id, account_config in accounts_config.items():
            if account_config.get('enabled', True):
                account_util = accounts_data.get(account_id, {})
                
                summary = {
                    'account_id': account_id,
                    'max_capital': account_config.get('max_capital', 0),
                    'currency': account_config.get('currency', 'USD'),
                    'enabled': account_config.get('enabled', True),
                    'buying_power': account_util.get('buying_power'),
                    'utilization_pct': account_util.get('utilization_pct', 0),
                    'status': account_util.get('status', 'no_data'),
                    'last_updated': account_util.get('last_updated', 'Never')
                }
                
                account_summaries.append(summary)
        
        return {
            'accounts': account_summaries,
            'total_enabled': len(account_summaries),
            'last_update': utilization_data.get('last_check', 'No recent data')
        }
        
    except Exception as e:
        logger.error(f"Failed to get account status: {e}")
        return {
            'accounts': [],
            'total_enabled': 0,
            'last_update': 'Error loading data',
            'error': str(e)
        }


def _render_account_status(account_data: Dict[str, Any]) -> html.Div:
    """Render account status summary"""
    if 'error' in account_data:
        return html.Div([
            html.P("❌ Error loading account data", className="text-danger"),
            html.Small(account_data['error'], className="text-muted")
        ])
    
    accounts = account_data.get('accounts', [])
    
    if not accounts:
        return html.Div([
            html.P("ℹ️ No accounts configured", className="text-muted"),
            html.Small("Add accounts using the capital CLI", className="text-muted")
        ])
    
    account_cards = []
    
    for account in accounts:
        account_id = account['account_id']
        status = account.get('status', 'no_data')
        utilization_pct = account.get('utilization_pct', 0)
        buying_power = account.get('buying_power')
        max_capital = account.get('max_capital', 0)
        
        # Status icon and color
        if status == 'critical':
            status_icon = "fas fa-exclamation-circle text-danger"
            card_color = "danger"
            status_text = "CRITICAL"
        elif status == 'warning':
            status_icon = "fas fa-exclamation-triangle text-warning"
            card_color = "warning"
            status_text = "WARNING"
        elif status == 'ok':
            status_icon = "fas fa-check-circle text-success"
            card_color = "success"
            status_text = "OK"
        else:
            status_icon = "fas fa-question-circle text-muted"
            card_color = "light"
            status_text = "NO DATA"
        
        account_card = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6([
                        html.I(className=status_icon + " me-2"),
                        account_id
                    ], className="mb-1"),
                    
                    html.Small([
                        html.Strong(status_text),
                        f" • {utilization_pct:.1f}% utilized"
                    ], className="text-muted mb-2"),
                    
                    html.Div([
                        html.Small(f"Available: ${buying_power:,.0f}" if buying_power else "Available: N/A", className="me-3"),
                        html.Small(f"Limit: ${max_capital:,.0f}", className="")
                    ])
                ])
            ])
        ], outline=True, color=card_color, className="mb-2")
        
        account_cards.append(account_card)
    
    return html.Div(account_cards)


def _render_greylist_badge() -> html.Span:
    """Render greylist status badge"""
    try:
        from ...utils.greylist import get_greylist_manager
        
        manager = get_greylist_manager()
        greylist = manager.get_greylist()
        
        if not greylist:
            return dbc.Badge("Greylist: 0 symbols", color="success", className="me-2")
        
        symbol_count = len(greylist)
        color = "warning" if symbol_count > 0 else "success"
        
        # Show first few symbols in tooltip
        if symbol_count <= 3:
            symbol_text = ", ".join(greylist)
        else:
            symbol_text = f"{', '.join(greylist[:3])}, +{symbol_count-3} more"
        
        return dbc.Badge(
            f"Greylist: {symbol_count} symbol{'s' if symbol_count != 1 else ''}",
            color=color,
            className="me-2",
            title=f"Greylisted symbols: {symbol_text}"
        )
        
    except Exception as e:
        logger.error(f"Failed to render greylist badge: {e}")
        return dbc.Badge("Greylist: Error", color="danger", className="me-2")