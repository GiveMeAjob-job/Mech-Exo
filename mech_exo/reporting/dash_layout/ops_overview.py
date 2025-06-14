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
            
            # Risk & Trading Status Section (Enhanced Day 2 Module 4)
            dbc.AccordionItem([
                html.Div([
                    # Live Risk Panel Row
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("📊 Live PnL Monitor", className="card-title"),
                                    html.Div(id="live-pnl-display"),
                                    # Monthly P&L Badge (Day 3 Module 5)
                                    html.Hr(),
                                    html.H6("Monthly P&L", className="text-muted"),
                                    html.Div(id="monthly-pnl-badge"),
                                    # Drill Status Badge (Day 4 Module 4)
                                    html.Hr(),
                                    html.H6("Last Drill", className="text-muted"),
                                    html.Div(id="drill-status-badge"),
                                    # Auto-refresh interval for 1-minute updates as specified
                                    dcc.Interval(
                                        id='interval-pnl',
                                        interval=60*1000,  # 1 minute refresh
                                        n_intervals=0
                                    )
                                ])
                            ], color="light", className="h-100")
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("🚨 Kill-Switch Status", className="card-title"),
                                    html.Div(id="killswitch-status-display")
                                ])
                            ], color="light", className="h-100")
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("⚖️ Risk Health", className="card-title"),
                                    html.Div(id="risk-health-summary")
                                ])
                            ], color="light", className="h-100")
                        ], width=4)
                    ], className="mb-3"),
                    
                    # Existing Risk & Trading Status Row
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
            Output('live-pnl-display', 'children'),
            Output('monthly-pnl-badge', 'children'),
            Output('drill-status-badge', 'children'),
            Output('killswitch-status-display', 'children'),
            Output('risk-health-summary', 'children')
        ],
        [
            Input('interval-pnl', 'n_intervals'),
            Input('interval-ops', 'n_intervals')
        ]
    )
    def update_risk_panel(pnl_intervals, ops_intervals):
        """Update live risk panel (Day 2 Module 4 + Day 3 Module 5 + Day 4 Module 4)"""
        try:
            # Get live PnL data
            pnl_data = _get_live_pnl_data()
            pnl_display = _render_live_pnl_display(pnl_data)
            
            # Get monthly PnL data (Day 3 Module 5)
            monthly_pnl_data = _get_monthly_pnl_data()
            monthly_pnl_badge = _render_monthly_pnl_badge(monthly_pnl_data)
            
            # Get drill status data (Day 4 Module 4)
            drill_data = _get_drill_status_data()
            drill_badge = _render_drill_status_badge(drill_data)
            
            # Get kill-switch status
            killswitch_data = _get_killswitch_data()
            killswitch_display = _render_killswitch_display(killswitch_data)
            
            # Get risk health summary
            risk_health_data = _get_risk_health_data()
            risk_health_display = _render_risk_health_display(risk_health_data)
            
            return pnl_display, monthly_pnl_badge, drill_badge, killswitch_display, risk_health_display
            
        except Exception as e:
            logger.error(f"Failed to update risk panel: {e}")
            error_msg = html.P("Error loading data", className="text-danger")
            return error_msg, error_msg, error_msg, error_msg, error_msg

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


# Day 2 Module 4: Risk Panel Functions

def _get_live_pnl_data() -> Dict[str, Any]:
    """Get live PnL data for risk panel"""
    try:
        from ..query import get_health_data
        health_data = get_health_data()
        
        return {
            'day_loss_pct': health_data.get('day_loss_pct', 0.0),
            'day_loss_ok': health_data.get('day_loss_ok', True),
            'live_nav': health_data.get('live_nav', 0.0),
            'day_start_nav': health_data.get('day_start_nav', 0.0),
            'last_updated': health_data.get('last_updated', 'Unknown')
        }
    except Exception as e:
        logger.error(f"Failed to get live PnL data: {e}")
        return {
            'day_loss_pct': 0.0,
            'day_loss_ok': True,
            'live_nav': 0.0,
            'day_start_nav': 0.0,
            'last_updated': 'Error',
            'error': str(e)
        }


def _render_live_pnl_display(pnl_data: Dict[str, Any]) -> html.Div:
    """Render live PnL display with color-coded indicator"""
    if 'error' in pnl_data:
        return html.Div([
            html.P("❌ PnL data unavailable", className="text-danger"),
            html.Small(pnl_data['error'], className="text-muted")
        ])
    
    day_loss_pct = pnl_data.get('day_loss_pct', 0.0)
    day_loss_ok = pnl_data.get('day_loss_ok', True)
    live_nav = pnl_data.get('live_nav', 0.0)
    day_start_nav = pnl_data.get('day_start_nav', 0.0)
    
    # Color-coded indicator as specified: green (> -0.4%), yellow (-0.4% to -0.8%), red (≤ -0.8%)
    if day_loss_pct > -0.4:
        indicator_color = "success"
        indicator_icon = "fas fa-circle text-success"
        status_text = "HEALTHY"
    elif day_loss_pct >= -0.8:
        indicator_color = "warning"
        indicator_icon = "fas fa-circle text-warning"
        status_text = "WARNING"
    else:
        indicator_color = "danger"
        indicator_icon = "fas fa-circle text-danger"
        status_text = "CRITICAL"
    
    return html.Div([
        # Numeric today-PnL as specified
        html.H4([
            html.I(className=indicator_icon + " me-2"),
            f"{day_loss_pct:+.2f}%"
        ], className=f"text-{indicator_color}"),
        
        # Status badge
        dbc.Badge(status_text, color=indicator_color, className="mb-2"),
        
        # NAV details
        html.Div([
            html.Small(f"Live NAV: ${live_nav:,.2f}", className="d-block"),
            html.Small(f"Start NAV: ${day_start_nav:,.2f}", className="d-block"),
            html.Small(f"Updated: {pnl_data.get('last_updated', 'Unknown')}", className="text-muted")
        ])
    ])


def _get_killswitch_data() -> Dict[str, Any]:
    """Get kill-switch status data"""
    try:
        from ..query import get_health_data
        health_data = get_health_data()
        
        return {
            'trading_enabled': health_data.get('trading_enabled', True),
            'killswitch_reason': health_data.get('killswitch_reason', 'System operational'),
            'last_updated': health_data.get('last_updated', 'Unknown')
        }
    except Exception as e:
        logger.error(f"Failed to get kill-switch data: {e}")
        return {
            'trading_enabled': True,
            'killswitch_reason': 'Status unavailable',
            'last_updated': 'Error',
            'error': str(e)
        }


def _render_killswitch_display(killswitch_data: Dict[str, Any]) -> html.Div:
    """Render kill-switch status display"""
    if 'error' in killswitch_data:
        return html.Div([
            html.P("❌ Kill-switch status unavailable", className="text-danger"),
            html.Small(killswitch_data['error'], className="text-muted")
        ])
    
    trading_enabled = killswitch_data.get('trading_enabled', True)
    reason = killswitch_data.get('killswitch_reason', 'System operational')
    
    if trading_enabled:
        icon = "fas fa-play-circle text-success"
        status_text = "TRADING ENABLED"
        badge_color = "success"
    else:
        icon = "fas fa-stop-circle text-danger"
        status_text = "TRADING DISABLED"
        badge_color = "danger"
    
    return html.Div([
        html.H5([
            html.I(className=icon + " me-2"),
            status_text
        ]),
        
        dbc.Badge(badge_color.upper(), color=badge_color, className="mb-2"),
        
        html.Div([
            html.Small(f"Reason: {reason}", className="d-block"),
            html.Small(f"Updated: {killswitch_data.get('last_updated', 'Unknown')}", className="text-muted")
        ])
    ])


def _get_risk_health_data() -> Dict[str, Any]:
    """Get risk health summary data"""
    try:
        from ..query import get_health_data
        health_data = get_health_data()
        
        return {
            'risk_ok': health_data.get('risk_ok', False),
            'day_loss_ok': health_data.get('day_loss_ok', True),
            'trading_enabled': health_data.get('trading_enabled', True),
            'system_status': health_data.get('system_status', 'unknown'),
            'fills_today': health_data.get('fills_today', 0)
        }
    except Exception as e:
        logger.error(f"Failed to get risk health data: {e}")
        return {
            'risk_ok': False,
            'day_loss_ok': True,
            'trading_enabled': True,
            'system_status': 'error',
            'fills_today': 0,
            'error': str(e)
        }


def _render_risk_health_display(risk_health_data: Dict[str, Any]) -> html.Div:
    """Render risk health summary display"""
    if 'error' in risk_health_data:
        return html.Div([
            html.P("❌ Risk health unavailable", className="text-danger"),
            html.Small(risk_health_data['error'], className="text-muted")
        ])
    
    risk_ok = risk_health_data.get('risk_ok', False)
    day_loss_ok = risk_health_data.get('day_loss_ok', True)
    trading_enabled = risk_health_data.get('trading_enabled', True)
    system_status = risk_health_data.get('system_status', 'unknown')
    fills_today = risk_health_data.get('fills_today', 0)
    
    # Overall health status
    if risk_ok and day_loss_ok and trading_enabled and system_status == 'operational':
        overall_icon = "fas fa-shield-alt text-success"
        overall_status = "ALL SYSTEMS GO"
        overall_color = "success"
    elif not day_loss_ok or not trading_enabled:
        overall_icon = "fas fa-exclamation-triangle text-danger"
        overall_status = "CRITICAL RISK"
        overall_color = "danger"
    else:
        overall_icon = "fas fa-exclamation-triangle text-warning"
        overall_status = "MINOR ISSUES"
        overall_color = "warning"
    
    return html.Div([
        html.H5([
            html.I(className=overall_icon + " me-2"),
            overall_status
        ]),
        
        dbc.Badge(overall_color.upper(), color=overall_color, className="mb-2"),
        
        # Health indicators
        html.Div([
            html.Div([
                html.I(className="fas fa-check-circle text-success me-1" if risk_ok else "fas fa-times-circle text-danger me-1"),
                html.Small("Risk Limits OK" if risk_ok else "Risk Violation")
            ], className="d-block"),
            
            html.Div([
                html.I(className="fas fa-check-circle text-success me-1" if day_loss_ok else "fas fa-times-circle text-danger me-1"),
                html.Small("PnL Within Limits" if day_loss_ok else "PnL Breach")
            ], className="d-block"),
            
            html.Div([
                html.I(className="fas fa-check-circle text-success me-1" if trading_enabled else "fas fa-times-circle text-danger me-1"),
                html.Small("Trading Active" if trading_enabled else "Trading Halted")
            ], className="d-block"),
            
            html.Small(f"Fills today: {fills_today}", className="text-muted mt-1 d-block")
        ])
    ])


# Day 3 Module 5: Monthly P&L Badge Functions

def _get_drill_status_data() -> Dict[str, Any]:
    """Get drill status data for dashboard badge"""
    try:
        from ...dags.drill_flow import get_last_drill_info
        
        # Get last drill information
        last_drill = get_last_drill_info()
        
        if last_drill:
            drill_date = last_drill['drill_date']
            days_ago = last_drill['days_ago']
            passed = last_drill['passed']
            dry_run = last_drill['dry_run']
            
            # Color rules: green < 90 days, yellow 90-120, red > 120
            if days_ago < 90:
                badge_color = 'success'
                status = 'CURRENT'
            elif days_ago <= 120:
                badge_color = 'warning'
                status = 'DUE SOON'
            else:
                badge_color = 'danger'
                status = 'OVERDUE'
            
            # Override color if last drill failed
            if not passed:
                badge_color = 'danger'
                status = 'FAILED'
            
            return {
                'drill_date': drill_date,
                'days_ago': days_ago,
                'passed': passed,
                'dry_run': dry_run,
                'badge_color': badge_color,
                'status': status,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            # No drills found
            return {
                'drill_date': None,
                'days_ago': None,
                'passed': False,
                'dry_run': True,
                'badge_color': 'danger',
                'status': 'NO DRILLS',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
    except Exception as e:
        logger.error(f"Failed to get drill status data: {e}")
        return {
            'drill_date': None,
            'days_ago': None,
            'passed': False,
            'dry_run': True,
            'badge_color': 'danger',
            'status': 'ERROR',
            'last_updated': 'Error',
            'error': str(e)
        }


def _render_drill_status_badge(drill_data: Dict[str, Any]) -> html.Div:
    """Render drill status badge with color-coded indicator"""
    if 'error' in drill_data:
        return html.Div([
            dbc.Badge("Last Drill: Error", color="danger", className="me-2"),
            html.Small(drill_data['error'], className="text-muted")
        ])
    
    drill_date = drill_data.get('drill_date')
    days_ago = drill_data.get('days_ago')
    passed = drill_data.get('passed', False)
    dry_run = drill_data.get('dry_run', True)
    status = drill_data.get('status', 'UNKNOWN')
    badge_color = drill_data.get('badge_color', 'secondary')
    
    if drill_date:
        # Format date display
        try:
            drill_date_obj = date.fromisoformat(drill_date)
            date_display = drill_date_obj.strftime('%Y-%m-%d')
        except:
            date_display = str(drill_date)
        
        # Create badge text
        if days_ago == 0:
            time_text = "today"
        elif days_ago == 1:
            time_text = "1 day ago"
        else:
            time_text = f"{days_ago} days ago"
        
        # Add status indicators
        status_indicators = []
        if not passed:
            status_indicators.append("❌")
        elif dry_run:
            status_indicators.append("🧪")
        else:
            status_indicators.append("✅")
        
        badge_text = f"{''.join(status_indicators)} {date_display} ({time_text})"
        
        return html.Div([
            dbc.Badge([
                html.I(className="fas fa-sync-alt me-1"),
                badge_text
            ], color=badge_color, className="me-2", 
            title=f"Last rollback drill: {status}"),
            
            html.Small(f"Status: {status}", className="text-muted")
        ])
    else:
        return html.Div([
            dbc.Badge([
                html.I(className="fas fa-exclamation-triangle me-1"),
                "No Drills Found"
            ], color="danger", className="me-2",
            title="No rollback drills have been executed"),
            
            html.Small("Status: NO DRILLS", className="text-muted")
        ])


def _get_monthly_pnl_data() -> Dict[str, Any]:
    """Get monthly P&L data for dashboard badge"""
    try:
        from ...utils.monthly_loss_guard import get_mtd_summary
        
        # Get MTD summary
        mtd_summary = get_mtd_summary()
        
        return {
            'mtd_pct': mtd_summary.get('mtd_pct', 0.0),
            'threshold_pct': mtd_summary.get('threshold_pct', -3.0),
            'threshold_breached': mtd_summary.get('threshold_breached', False),
            'status': mtd_summary.get('status', 'UNKNOWN'),
            'status_color': mtd_summary.get('status_color', 'secondary'),
            'month_year': mtd_summary.get('month_year', 'Unknown'),
            'last_updated': mtd_summary.get('last_updated', 'Unknown')
        }
    except Exception as e:
        logger.error(f"Failed to get monthly PnL data: {e}")
        return {
            'mtd_pct': 0.0,
            'threshold_pct': -3.0,
            'threshold_breached': False,
            'status': 'ERROR',
            'status_color': 'danger',
            'month_year': 'Unknown',
            'last_updated': 'Error',
            'error': str(e)
        }


def _render_monthly_pnl_badge(monthly_data: Dict[str, Any]) -> html.Div:
    """Render monthly P&L badge with color-coded indicator"""
    if 'error' in monthly_data:
        return html.Div([
            dbc.Badge("Monthly P&L: Error", color="danger", className="me-2"),
            html.Small(monthly_data['error'], className="text-muted")
        ])
    
    mtd_pct = monthly_data.get('mtd_pct', 0.0)
    status = monthly_data.get('status', 'UNKNOWN')
    month_year = monthly_data.get('month_year', 'Unknown')
    
    # Color rules as specified:
    # Green > 0%, Yellow 0 … -2%, Red ≤ -3%
    if mtd_pct > 0:
        badge_color = "success"
        indicator_icon = "fas fa-arrow-up"
    elif mtd_pct >= -2.0:
        badge_color = "warning"
        indicator_icon = "fas fa-minus"
    else:
        badge_color = "danger"
        indicator_icon = "fas fa-arrow-down"
    
    # Show red specifically for threshold breach (≤ -3%)
    if mtd_pct <= -3.0:
        badge_color = "danger"
        indicator_icon = "fas fa-exclamation-triangle"
    
    return html.Div([
        dbc.Badge([
            html.I(className=indicator_icon + " me-1"),
            f"{mtd_pct:+.1f}%"
        ], color=badge_color, className="me-2", 
        title=f"Month-to-Date return for {month_year}"),
        
        html.Small(f"MTD {month_year}", className="text-muted")
    ])