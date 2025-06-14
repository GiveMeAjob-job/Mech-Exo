"""
Unified Risk Control Panel - Day 5 Module 1

Single dashboard tab consolidating all risk monitoring widgets:
- Today PnL lamp (traffic light indicator)
- Monthly PnL lamp (month-to-date status)  
- Kill-Switch toggle (trading enabled/disabled)
- Canary status badge (A/B testing status)
- Drill badge (days since last drill)
- Capital utilization bar (account usage)
"""

import logging
from datetime import datetime, date
from typing import Dict, Any, Optional

import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import requests

logger = logging.getLogger(__name__)


def create_risk_control_layout():
    """Create the unified Risk Control dashboard layout"""
    
    return html.Div([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-shield-alt me-2 text-danger"),
                    "⚡ Risk Control"
                ], className="text-primary mb-0"),
                html.P("Unified risk monitoring and control center", className="text-muted mb-3"),
            ], width=8),
            dbc.Col([
                html.Div([
                    dbc.Badge("LIVE", color="success", className="me-2"),
                    dbc.Badge("AUTO-REFRESH", color="info", className="me-2"),
                    html.Small(id="risk-last-updated", className="text-muted")
                ], className="text-end")
            ], width=4)
        ], className="mb-4"),
        
        # Main Risk Control Grid
        dbc.Row([
            # Left Column - PnL Lamps
            dbc.Col([
                # Today PnL Lamp
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-traffic-light me-2"),
                            "Today P&L"
                        ], className="card-title text-center"),
                        html.Div(id="today-pnl-lamp", className="text-center mb-3"),
                        html.Div(id="today-pnl-details", className="text-center"),
                    ])
                ], className="h-100", color="light", outline=True),
            ], width=12, lg=6, className="mb-3"),
            
            # Monthly PnL Lamp  
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-calendar-alt me-2"),
                            "Monthly P&L"
                        ], className="card-title text-center"),
                        html.Div(id="monthly-pnl-lamp", className="text-center mb-3"),
                        html.Div(id="monthly-pnl-details", className="text-center"),
                    ])
                ], className="h-100", color="light", outline=True),
            ], width=12, lg=6, className="mb-3"),
        ]),
        
        # Middle Row - Controls
        dbc.Row([
            # Kill-Switch Toggle
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-power-off me-2"),
                            "Kill-Switch"
                        ], className="card-title text-center"),
                        html.Div(id="killswitch-toggle", className="text-center mb-3"),
                        html.Div(id="killswitch-status", className="text-center"),
                    ])
                ], className="h-100", color="light", outline=True),
            ], width=12, lg=4, className="mb-3"),
            
            # Canary Status Badge
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-flask me-2"),
                            "Canary Status"
                        ], className="card-title text-center"),
                        html.Div(id="canary-status-badge", className="text-center mb-3"),
                        html.Div(id="canary-details", className="text-center"),
                    ])
                ], className="h-100", color="light", outline=True),
            ], width=12, lg=4, className="mb-3"),
            
            # Drill Badge
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-sync-alt me-2"),
                            "Last Drill"
                        ], className="card-title text-center"),
                        html.Div(id="drill-status-badge-unified", className="text-center mb-3"),
                        html.Div(id="drill-details", className="text-center"),
                    ])
                ], className="h-100", color="light", outline=True),
            ], width=12, lg=4, className="mb-3"),
        ]),
        
        # Bottom Row - Capital Utilization
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-chart-bar me-2"),
                            "Capital Utilization"
                        ], className="card-title"),
                        html.Div(id="capital-utilization-unified", className="mt-3"),
                    ])
                ], className="h-100", color="light", outline=True),
            ], width=12, className="mb-3"),
        ]),
        
        # Action Buttons
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button([
                        html.I(className="fas fa-sync me-2"),
                        "Force Refresh"
                    ], id="risk-refresh-btn", color="primary", outline=True),
                    dbc.Button([
                        html.I(className="fas fa-download me-2"),
                        "Export Status"
                    ], id="risk-export-btn", color="secondary", outline=True),
                    dbc.Button([
                        html.I(className="fas fa-cog me-2"),
                        "Settings"
                    ], id="risk-settings-btn", color="info", outline=True),
                ])
            ], width="auto"),
            dbc.Col([
                html.Div(id="risk-action-feedback", className="text-end")
            ])
        ], justify="between", className="mt-4"),
        
        # Auto-refresh interval
        dcc.Interval(
            id='interval-risk-control',
            interval=60*1000,  # 60 second refresh
            n_intervals=0
        ),
        
        # Hidden store for risk data
        dcc.Store(id='risk-data-store'),
    ])


def register_risk_control_callbacks():
    """Register callbacks for the Risk Control panel"""
    
    @callback(
        [
            Output('risk-data-store', 'data'),
            Output('risk-last-updated', 'children')
        ],
        [
            Input('interval-risk-control', 'n_intervals'),
            Input('risk-refresh-btn', 'n_clicks')
        ]
    )
    def fetch_risk_data(n_intervals, refresh_clicks):
        """Fetch risk data from /riskz API endpoint"""
        try:
            # Get risk summary from API
            risk_data = _get_risk_summary_api()
            
            # Add fetch timestamp
            risk_data['last_updated'] = datetime.now().isoformat()
            
            # Format last updated display
            last_updated = html.Small(
                f"Updated: {datetime.now().strftime('%H:%M:%S')}",
                className="text-muted"
            )
            
            return risk_data, last_updated
            
        except Exception as e:
            logger.error(f"Failed to fetch risk data: {e}")
            
            # Return error state
            error_data = {
                'today_pnl': 0.0,
                'month_pnl': 0.0,
                'kill': True,  # Safe default
                'canary': 'error',
                'drill_days': 999,
                'capital_pct': 0,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
            
            error_updated = html.Small(
                f"Error: {datetime.now().strftime('%H:%M:%S')}",
                className="text-danger"
            )
            
            return error_data, error_updated
    
    @callback(
        [
            Output('today-pnl-lamp', 'children'),
            Output('today-pnl-details', 'children'),
            Output('monthly-pnl-lamp', 'children'),
            Output('monthly-pnl-details', 'children'),
            Output('killswitch-toggle', 'children'),
            Output('killswitch-status', 'children'),
            Output('canary-status-badge', 'children'),
            Output('canary-details', 'children'),
            Output('drill-status-badge-unified', 'children'),
            Output('drill-details', 'children'),
            Output('capital-utilization-unified', 'children')
        ],
        Input('risk-data-store', 'data')
    )
    def update_risk_widgets(risk_data):
        """Update all risk control widgets based on fetched data"""
        try:
            if not risk_data:
                # Return loading state
                loading = html.Div([
                    dbc.Spinner(size="sm", color="primary"),
                    html.Small(" Loading...", className="ms-2")
                ])
                return [loading] * 11
            
            # Check for error state
            if 'error' in risk_data:
                error_widget = html.Div([
                    html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                    html.Small("Error loading data", className="text-danger")
                ])
                return [error_widget] * 11
            
            # Extract data
            today_pnl = risk_data.get('today_pnl', 0.0)
            month_pnl = risk_data.get('month_pnl', 0.0)
            kill_switch = risk_data.get('kill', False)
            canary_status = risk_data.get('canary', 'unknown')
            drill_days = risk_data.get('drill_days', 999)
            capital_pct = risk_data.get('capital_pct', 0)
            
            # Render widgets
            today_lamp = _render_pnl_lamp(today_pnl, is_daily=True)
            today_details = _render_pnl_details(today_pnl, is_daily=True)
            
            monthly_lamp = _render_pnl_lamp(month_pnl, is_daily=False)
            monthly_details = _render_pnl_details(month_pnl, is_daily=False)
            
            killswitch_toggle = _render_killswitch_toggle(kill_switch)
            killswitch_status = _render_killswitch_status(kill_switch)
            
            canary_badge = _render_canary_badge(canary_status)
            canary_details = _render_canary_details(canary_status)
            
            drill_badge = _render_drill_badge(drill_days)
            drill_details = _render_drill_details(drill_days)
            
            capital_bar = _render_capital_bar(capital_pct)
            
            return (
                today_lamp, today_details,
                monthly_lamp, monthly_details,
                killswitch_toggle, killswitch_status,
                canary_badge, canary_details,
                drill_badge, drill_details,
                capital_bar
            )
            
        except Exception as e:
            logger.error(f"Failed to update risk widgets: {e}")
            error_widget = html.Div([
                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                html.Small(f"Widget error: {e}", className="text-danger")
            ])
            return [error_widget] * 11


# Widget rendering functions

def _render_pnl_lamp(pnl_pct: float, is_daily: bool = True) -> html.Div:
    """Render PnL traffic light lamp"""
    # Color rules for daily: green > -0.4%, yellow -0.4% to -0.8%, red ≤ -0.8%
    # Color rules for monthly: green > 0%, yellow 0 to -2%, red ≤ -3%
    
    if is_daily:
        if pnl_pct > -0.4:
            color = "success"
            icon = "fas fa-circle"
        elif pnl_pct >= -0.8:
            color = "warning"
            icon = "fas fa-circle"
        else:
            color = "danger"
            icon = "fas fa-circle"
    else:  # Monthly
        if pnl_pct > 0:
            color = "success"
            icon = "fas fa-circle"
        elif pnl_pct >= -2.0:
            color = "warning"
            icon = "fas fa-circle"
        else:
            color = "danger"
            icon = "fas fa-circle"
    
    return html.Div([
        html.I(
            className=f"{icon} fa-3x text-{color}",
            style={"filter": "drop-shadow(0 0 10px rgba(0,0,0,0.3))"}
        )
    ])


def _render_pnl_details(pnl_pct: float, is_daily: bool = True) -> html.Div:
    """Render PnL details below lamp"""
    period = "Today" if is_daily else "MTD"
    
    return html.Div([
        html.H4(f"{pnl_pct:+.2f}%", className="mb-1"),
        html.Small(f"{period} P&L", className="text-muted")
    ])


def _render_killswitch_toggle(kill_switch: bool) -> html.Div:
    """Render kill-switch toggle button"""
    if kill_switch:
        # Trading disabled
        return html.Div([
            dbc.Button([
                html.I(className="fas fa-stop me-2"),
                "DISABLED"
            ], color="danger", size="lg", disabled=True, className="w-100")
        ])
    else:
        # Trading enabled
        return html.Div([
            dbc.Button([
                html.I(className="fas fa-play me-2"),
                "ENABLED"
            ], color="success", size="lg", id="killswitch-enable-btn", className="w-100")
        ])


def _render_killswitch_status(kill_switch: bool) -> html.Div:
    """Render kill-switch status details"""
    if kill_switch:
        return html.Div([
            dbc.Badge("TRADING HALTED", color="danger", className="mb-1"),
            html.Br(),
            html.Small("Trading is disabled", className="text-danger")
        ])
    else:
        return html.Div([
            dbc.Badge("TRADING ACTIVE", color="success", className="mb-1"),
            html.Br(),
            html.Small("System operational", className="text-success")
        ])


def _render_canary_badge(canary_status: str) -> html.Div:
    """Render canary status badge"""
    if canary_status.lower() in ['on', 'enabled', 'active']:
        color = "success"
        icon = "fas fa-check-circle"
        text = "ACTIVE"
    elif canary_status.lower() in ['off', 'disabled', 'inactive']:
        color = "secondary"
        icon = "fas fa-pause-circle"
        text = "INACTIVE"
    else:
        color = "warning"
        icon = "fas fa-question-circle"
        text = "UNKNOWN"
    
    return html.Div([
        html.I(className=f"{icon} fa-2x text-{color}")
    ])


def _render_canary_details(canary_status: str) -> html.Div:
    """Render canary status details"""
    if canary_status.lower() in ['on', 'enabled', 'active']:
        return html.Div([
            dbc.Badge("A/B TESTING", color="success", className="mb-1"),
            html.Br(),
            html.Small("Canary enabled", className="text-success")
        ])
    else:
        return html.Div([
            dbc.Badge("BASELINE", color="secondary", className="mb-1"),
            html.Br(),
            html.Small("Canary disabled", className="text-muted")
        ])


def _render_drill_badge(drill_days: int) -> html.Div:
    """Render drill status badge"""
    # Color rules: green < 90 days, yellow 90-120, red > 120
    if drill_days < 90:
        color = "success"
        icon = "fas fa-check-circle"
    elif drill_days <= 120:
        color = "warning" 
        icon = "fas fa-exclamation-triangle"
    else:
        color = "danger"
        icon = "fas fa-times-circle"
    
    return html.Div([
        html.I(className=f"{icon} fa-2x text-{color}")
    ])


def _render_drill_details(drill_days: int) -> html.Div:
    """Render drill status details"""
    if drill_days < 90:
        status = "CURRENT"
        color = "success"
    elif drill_days <= 120:
        status = "DUE SOON"
        color = "warning"
    else:
        status = "OVERDUE"
        color = "danger"
    
    return html.Div([
        dbc.Badge(status, color=color, className="mb-1"),
        html.Br(),
        html.Small(f"{drill_days} days ago", className="text-muted")
    ])


def _render_capital_bar(capital_pct: float) -> html.Div:
    """Render capital utilization progress bar"""
    # Color rules: green < 70%, yellow 70-90%, red > 90%
    if capital_pct < 70:
        color = "success"
        variant = ""
    elif capital_pct <= 90:
        color = "warning"
        variant = "striped"
    else:
        color = "danger"
        variant = "striped animated"
    
    return html.Div([
        html.Div([
            html.Span(f"Utilization: {capital_pct:.1f}%", className="fw-bold"),
            html.Span(f" ({_get_utilization_status(capital_pct)})", className="text-muted ms-2")
        ], className="d-flex justify-content-between mb-2"),
        
        dbc.Progress(
            value=capital_pct,
            color=color,
            striped=bool(variant),
            animated="animated" in variant if variant else False,
            style={"height": "20px"}
        )
    ])


def _get_utilization_status(capital_pct: float) -> str:
    """Get utilization status text"""
    if capital_pct < 70:
        return "Healthy"
    elif capital_pct <= 90:
        return "Elevated"
    else:
        return "Critical"


def _get_risk_summary_api() -> Dict[str, Any]:
    """Get risk summary from /riskz API endpoint"""
    try:
        # Try to fetch from API endpoint
        response = requests.get("http://localhost:8000/riskz", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"API returned status {response.status_code}")
            return _get_risk_summary_fallback()
            
    except requests.RequestException as e:
        logger.warning(f"API request failed: {e}, using fallback")
        return _get_risk_summary_fallback()


def _get_risk_summary_fallback() -> Dict[str, Any]:
    """Fallback risk summary when API is unavailable"""
    try:
        # Import individual data sources
        from ..query import get_health_data
        from ...dags.drill_flow import get_last_drill_info
        from ...cli.capital import CapitalManager
        
        # Get health data
        health_data = get_health_data()
        
        # Get drill info
        drill_info = get_last_drill_info()
        drill_days = drill_info['days_ago'] if drill_info else 999
        
        # Get capital data
        try:
            capital_manager = CapitalManager()
            capital_config = capital_manager.config
            utilization_data = capital_config.get('utilization', {})
            accounts_data = utilization_data.get('accounts', {})
            
            # Calculate average utilization
            if accounts_data:
                total_utilization = sum(acc.get('utilization_pct', 0) for acc in accounts_data.values())
                avg_utilization = total_utilization / len(accounts_data)
            else:
                avg_utilization = 0
                
        except Exception as e:
            logger.warning(f"Failed to get capital data: {e}")
            avg_utilization = 0
        
        return {
            'today_pnl': health_data.get('day_loss_pct', 0.0),
            'month_pnl': health_data.get('mtd_pnl_pct', 0.0),
            'kill': not health_data.get('trading_enabled', True),
            'canary': 'on' if health_data.get('canary_enabled', False) else 'off',
            'drill_days': drill_days,
            'capital_pct': avg_utilization
        }
        
    except Exception as e:
        logger.error(f"Fallback data fetch failed: {e}")
        return {
            'today_pnl': 0.0,
            'month_pnl': 0.0,
            'kill': True,  # Safe default
            'canary': 'unknown',
            'drill_days': 999,
            'capital_pct': 0,
            'error': str(e)
        }