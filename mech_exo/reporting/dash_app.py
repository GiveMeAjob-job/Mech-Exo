"""
Dash application for real-time trading dashboard
Interactive monitoring of portfolio, positions, and risk metrics
"""

import logging
import os
from typing import Optional

import dash
from dash import dcc, html, Input, Output, callback
from flask import Flask, request
from flask_httpauth import HTTPBasicAuth

from .query import get_health_data
from .dash_layout.equity import create_equity_layout, register_equity_callbacks
from .dash_layout.positions import create_positions_layout, register_positions_callbacks
from .dash_layout.risk import create_risk_layout, register_risk_callbacks
from .dash_layout.factor_health import create_factor_health_layout
from .dash_layout.optuna_monitor import create_optuna_monitor_layout
from .dash_layout.ml_signal import create_ml_signal_tab
from .dash_layout.ab_test import create_ab_test_layout, register_ab_test_callbacks
from .dash_layout.ops_overview import create_ops_overview_layout, register_ops_overview_callbacks

logger = logging.getLogger(__name__)


def create_dash_app(flask_app: Optional[Flask] = None) -> dash.Dash:
    """
    Factory function to create Dash application
    
    Args:
        flask_app: Optional Flask app to use as server
        
    Returns:
        Configured Dash application instance
    """
    
    # Create Flask server if not provided
    if flask_app is None:
        flask_app = Flask(__name__)
        
        # Set up basic authentication
        auth = HTTPBasicAuth()
        
        # Get credentials from environment variables
        dash_user = os.getenv('DASH_USER', 'admin')
        dash_pass = os.getenv('DASH_PASS', 'changeme')
        users = {dash_user: dash_pass}
        
        @auth.get_password
        def get_pw(username):
            return users.get(username)
        
        # Add health check endpoint (no auth required for monitoring)
        @flask_app.route('/healthz')
        def health_check():
            """Health check endpoint for monitoring"""
            try:
                health_data = get_health_data()
                status = health_data.get('system_status', 'error')
                risk_ok = health_data.get('risk_ok', False)
                
                # Return JSON response for programmatic access
                if request.headers.get('Accept', '').startswith('application/json'):
                    # Get current ML weight for health endpoint
                    try:
                        from .query import get_current_ml_weight_info
                        weight_info = get_current_ml_weight_info()
                        ml_weight = weight_info['current_weight']
                    except Exception as e:
                        logger.warning(f"Failed to get ML weight for health endpoint: {e}")
                        ml_weight = 0.30  # Default fallback
                    
                    # Get capital health status
                    capital_ok = _get_capital_health_status()
                    
                    # Calculate ops_ok based on various health indicators
                    ops_ok = (
                        status == 'operational' and
                        risk_ok and
                        capital_ok and
                        health_data.get('fills_today', 0) >= 0  # Basic sanity check
                    )
                    
                    return {
                        'status': status,
                        'risk_ok': risk_ok,
                        'capital_ok': capital_ok,  # Capital utilization health
                        'timestamp': health_data.get('last_updated'),
                        'fills_today': health_data.get('fills_today', 0),
                        'ml_signal': True,  # ML Signal tab is available
                        'ml_weight': ml_weight,  # Current ML weight
                        'canary_sharpe_30d': health_data.get('canary_sharpe_30d', 0.0),
                        'canary_enabled': health_data.get('canary_enabled', True),
                        'ops_ok': ops_ok  # Overall operational health indicator
                    }
                
                # Return simple text for curl/browser
                if status == 'operational':
                    return 'OK', 200
                else:
                    return f'DEGRADED: {status}', 503
                    
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return f'ERROR: {str(e)}', 500
        
        # Protect dashboard routes with authentication
        @flask_app.before_request
        def protect_routes():
            """Require authentication for dashboard routes, allow health check"""
            # Allow health check without authentication
            if request.path == '/healthz':
                return None
            # Require authentication for all other routes
            if not auth.current_user():
                return auth.auth_error_callback()
    
    # Create Dash app
    app = dash.Dash(
        __name__,
        server=flask_app,
        url_base_pathname='/',
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
        ]
    )
    
    # Configure app
    app.title = "Mech-Exo Trading Dashboard"
    app.config.suppress_callback_exceptions = True
    
    # Custom CSS styling
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f8f9fa;
                }
                .main-header {
                    background: linear-gradient(135deg, #007bff, #6610f2);
                    color: white;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    border-radius: 0.5rem;
                }
                .nav-tabs .nav-link {
                    color: #495057;
                    border: none;
                    border-radius: 0.5rem 0.5rem 0 0;
                }
                .nav-tabs .nav-link.active {
                    background-color: #007bff;
                    color: white;
                    border-color: #007bff;
                }
                .tab-content {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 0 0.5rem 0.5rem 0.5rem;
                    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
                }
                .metric-card {
                    background: white;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
                    margin-bottom: 1rem;
                }
                .status-healthy { color: #28a745; }
                .status-warning { color: #ffc107; }
                .status-error { color: #dc3545; }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Build layout
    app.layout = create_layout()
    
    # Register callbacks
    register_callbacks(app)
    register_equity_callbacks()
    register_positions_callbacks()
    register_risk_callbacks()
    register_ab_test_callbacks()
    register_ops_overview_callbacks()
    
    logger.info("Dash application created successfully")
    return app


def create_layout():
    """Create the dashboard layout"""
    
    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.H1([
                    html.I(className="fas fa-chart-line me-2"),
                    "Mech-Exo Trading Dashboard"
                ], className="h3 mb-1"),
                html.P("Real-time portfolio monitoring and risk management", 
                       className="mb-0 opacity-75"),
            ], className="col"),
            html.Div([
                html.Div(id="system-status", className="text-end")
            ], className="col-auto")
        ], className="row main-header"),
        
        # Tab navigation
        html.Div([
            dcc.Tabs(
                id="main-tabs",
                value="equity-tab",
                className="nav nav-tabs",
                children=[
                    dcc.Tab(
                        label="📈 Equity Curve",
                        value="equity-tab",
                        className="nav-link"
                    ),
                    dcc.Tab(
                        label="💼 Positions",
                        value="positions-tab", 
                        className="nav-link"
                    ),
                    dcc.Tab(
                        label="🎯 Risk Heat-map",
                        value="risk-tab",
                        className="nav-link"
                    ),
                    dcc.Tab(
                        label="🧬 Factor Health",
                        value="factor-health-tab",
                        className="nav-link"
                    ),
                    dcc.Tab(
                        label="🔬 Optuna Monitor",
                        value="optuna-monitor-tab",
                        className="nav-link"
                    ),
                    dcc.Tab(
                        label="🤖 ML Signal",
                        value="ml-signal-tab",
                        className="nav-link"
                    ),
                    dcc.Tab(
                        label="🧪 A/B Test",
                        value="ab-test-tab",
                        className="nav-link"
                    ),
                    dcc.Tab(
                        label="🔧 Operations",
                        value="ops-overview-tab",
                        className="nav-link"
                    )
                ]
            )
        ]),
        
        # Tab content
        html.Div([
            html.Div(id="tab-content", className="tab-content")
        ]),
        
        # Auto-refresh intervals
        dcc.Interval(
            id='interval-health',
            interval=10*1000,  # Update every 10 seconds
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-positions',
            interval=30*1000,  # Update every 30 seconds  
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-risk',
            interval=60*1000,  # Update every 60 seconds
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-ops',
            interval=5*60*1000,  # Update every 5 minutes
            n_intervals=0
        )
        
    ], className="container-fluid py-3")


def register_callbacks(app: dash.Dash):
    """Register dashboard callbacks"""
    
    @app.callback(
        Output('system-status', 'children'),
        Input('interval-health', 'n_intervals')
    )
    def update_system_status(n):
        """Update system status indicator"""
        try:
            health_data = get_health_data()
            status = health_data.get('system_status', 'error')
            last_updated = health_data.get('last_updated', 'Unknown')
            
            if status == 'operational':
                icon = "fas fa-check-circle status-healthy"
                text = "System Operational"
            elif status == 'degraded':
                icon = "fas fa-exclamation-triangle status-warning"
                text = "System Degraded"
            else:
                icon = "fas fa-times-circle status-error"
                text = "System Error"
            
            return html.Div([
                html.I(className=icon + " me-1"),
                html.Span(text),
                html.Br(),
                html.Small(f"Updated: {last_updated[:19]}", className="text-muted")
            ])
            
        except Exception as e:
            logger.error(f"Failed to update system status: {e}")
            return html.Div([
                html.I(className="fas fa-times-circle status-error me-1"),
                html.Span("Status Error")
            ])
    
    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'value')
    )
    def render_tab_content(active_tab):
        """Render content for active tab"""
        try:
            if active_tab == 'equity-tab':
                return create_equity_tab()
            elif active_tab == 'positions-tab':
                return create_positions_tab()
            elif active_tab == 'risk-tab':
                return create_risk_tab()
            elif active_tab == 'factor-health-tab':
                return create_factor_health_tab()
            elif active_tab == 'optuna-monitor-tab':
                return create_optuna_monitor_tab()
            elif active_tab == 'ml-signal-tab':
                return create_ml_signal_tab()
            elif active_tab == 'ab-test-tab':
                return create_ab_test_tab()
            elif active_tab == 'ops-overview-tab':
                return create_ops_overview_tab()
            else:
                return html.Div("Tab not found", className="text-danger")
                
        except Exception as e:
            logger.error(f"Failed to render tab content: {e}")
            return html.Div(f"Error loading tab: {str(e)}", className="text-danger")


def create_equity_tab():
    """Create equity curve tab content"""
    return create_equity_layout()


def create_positions_tab():
    """Create positions tab content"""
    return create_positions_layout()


def create_risk_tab():
    """Create risk heat-map tab content"""
    return create_risk_layout()


def create_factor_health_tab():
    """Create factor health tab content"""
    return create_factor_health_layout()


def create_optuna_monitor_tab():
    """Create Optuna monitor tab content"""
    return create_optuna_monitor_layout()


def create_ab_test_tab():
    """Create A/B test tab content"""
    return create_ab_test_layout()


def create_ops_overview_tab():
    """Create operations overview tab content"""
    return create_ops_overview_layout()


def make_dash_app() -> dash.Dash:
    """
    Factory function for creating Dash app instance
    Used by CLI and WSGI servers
    """
    return create_dash_app()


# For gunicorn/WSGI deployment
def create_app():
    """WSGI application factory"""
    # Create Flask app with authentication
    flask_app = Flask(__name__)
    
    # Set up basic authentication
    auth = HTTPBasicAuth()
    
    # Get credentials from environment variables
    dash_user = os.getenv('DASH_USER', 'admin')
    dash_pass = os.getenv('DASH_PASS', 'changeme')
    users = {dash_user: dash_pass}
    
    @auth.get_password
    def get_pw(username):
        return users.get(username)
    
    # Add health check endpoint (no auth required)
    @flask_app.route('/healthz')
    def health_check():
        """Health check endpoint for monitoring"""
        try:
            health_data = get_health_data()
            status = health_data.get('system_status', 'error')
            risk_ok = health_data.get('risk_ok', False)
            
            # Return JSON response for programmatic access
            if request.headers.get('Accept', '').startswith('application/json'):
                # Get current ML weight for health endpoint
                try:
                    from .query import get_current_ml_weight_info
                    weight_info = get_current_ml_weight_info()
                    ml_weight = weight_info['current_weight']
                except Exception as e:
                    logger.warning(f"Failed to get ML weight for health endpoint: {e}")
                    ml_weight = 0.30  # Default fallback
                
                # Calculate ops_ok based on various health indicators
                ops_ok = (
                    status == 'operational' and
                    risk_ok and
                    health_data.get('fills_today', 0) >= 0  # Basic sanity check
                )
                
                return {
                    'status': status,
                    'risk_ok': risk_ok,
                    'timestamp': health_data.get('last_updated'),
                    'fills_today': health_data.get('fills_today', 0),
                    'ml_signal': True,  # ML Signal tab is available
                    'ml_weight': ml_weight,  # Current ML weight
                    'canary_sharpe_30d': health_data.get('canary_sharpe_30d', 0.0),
                    'canary_enabled': health_data.get('canary_enabled', True),
                    'ops_ok': ops_ok  # Overall operational health indicator
                }
            
            # Return simple text for curl/browser
            if status == 'operational':
                return 'OK', 200
            else:
                return f'DEGRADED: {status}', 503
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return f'ERROR: {str(e)}', 500
    
    # Protect dashboard routes with authentication
    @flask_app.before_request
    def protect_routes():
        """Require authentication for dashboard routes, allow health check"""
        # Allow health check without authentication
        if request.path == '/healthz':
            return None
        # Require authentication for all other routes
        if not auth.current_user():
            return auth.auth_error_callback()
    
    # Create Dash app with the authenticated Flask server
    dash_app = create_dash_app(flask_app)
    return flask_app


def _get_capital_health_status() -> bool:
    """
    Get capital health status from database
    
    Returns:
        True if capital utilization is healthy, False otherwise
    """
    try:
        from ..datasource.storage import DataStorage
        
        storage = DataStorage()
        
        # Check if capital_health table exists and get latest status
        result = storage.conn.execute("""
            SELECT capital_ok, last_check
            FROM capital_health 
            WHERE id = 1
            ORDER BY updated_at DESC
            LIMIT 1
        """).fetchone()
        
        storage.close()
        
        if result:
            capital_ok, last_check = result
            # Check if the status is recent (within last 25 hours)
            from datetime import datetime, timedelta
            if last_check:
                try:
                    check_time = datetime.fromisoformat(last_check.replace('Z', '+00:00'))
                    if datetime.now() - check_time < timedelta(hours=25):
                        return bool(capital_ok)
                except:
                    pass
        
        # Default to True if no recent data (don't block trading)
        return True
        
    except Exception as e:
        logger.warning(f"Failed to get capital health status: {e}")
        # Default to True if error (don't block trading due to monitoring issues)
        return True


if __name__ == '__main__':
    # Development server
    app = create_dash_app()
    app.run(
        debug=True,
        host='0.0.0.0',
        port=8050
    )