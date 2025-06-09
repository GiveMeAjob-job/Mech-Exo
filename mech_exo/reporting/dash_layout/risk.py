"""
Risk heat-map tab layout and callbacks for Dash dashboard
Displays risk metrics with color-coded heat-map visualization
"""

import logging
from typing import Dict, Any

import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html, Input, Output, callback

from ..query import get_live_risk, get_health_data

logger = logging.getLogger(__name__)


def create_risk_layout():
    """
    Create the risk heat-map tab layout
    
    Returns:
        Dash layout components for risk heat-map visualization
    """
    
    return html.Div([
        # Header row with title and status badges
        html.Div([
            html.Div([
                html.H4([
                    html.I(className="fas fa-shield-alt me-2"),
                    "Risk Heat-map"
                ], className="mb-0")
            ], className="col"),
            html.Div([
                html.Div([
                    html.Div(id="drift-status-badge", className="mb-2"),
                    html.Div(id="risk-status-badge")
                ], className="text-end")
            ], className="col-auto")
        ], className="row mb-3"),
        
        # Risk summary cards
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-check-circle fa-2x text-success mb-2"),
                        html.H5("Risk OK", className="mb-1"),
                        html.H4(id="risk-ok-count", className="text-success mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-2"),
                        html.H5("Warnings", className="mb-1"),
                        html.H4(id="risk-warning-count", className="text-warning mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-times-circle fa-2x text-danger mb-2"),
                        html.H5("Breaches", className="mb-1"),
                        html.H4(id="risk-breach-count", className="text-danger mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-clock fa-2x text-secondary mb-2"),
                        html.H5("Last Updated", className="mb-1"),
                        html.H6(id="risk-last-updated", className="text-secondary mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3")
        ], className="row mb-4"),
        
        # Main heat-map visualization
        html.Div([
            html.Div([
                html.H5([
                    html.I(className="fas fa-fire me-2"),
                    "Risk Utilization Heat-map"
                ], className="mb-3"),
                html.P("Color coding: Green < 70%, Yellow 70-90%, Red > 90%", 
                       className="text-muted small mb-3"),
                
                html.Div(id="risk-heatmap-container"),
                
                # Data loading indicator
                html.Div([
                    dcc.Loading(
                        id="risk-loading",
                        type="default",
                        children=html.Div(id="risk-loading-output")
                    )
                ], className="text-center mt-3")
                
            ], className="metric-card")
        ], className="row mb-4"),
        
        # Risk details table
        html.Div([
            html.Div([
                html.H5([
                    html.I(className="fas fa-table me-2"),
                    "Risk Metrics Details"
                ], className="mb-3"),
                html.Div(id="risk-details-table")
            ], className="metric-card")
        ], className="row")
        
    ])


def register_risk_callbacks():
    """Register callbacks for risk heat-map tab"""
    
    @callback(
        [Output('risk-heatmap-container', 'children'),
         Output('risk-status-badge', 'children'),
         Output('drift-status-badge', 'children'),
         Output('risk-ok-count', 'children'),
         Output('risk-warning-count', 'children'),
         Output('risk-breach-count', 'children'),
         Output('risk-last-updated', 'children'),
         Output('risk-details-table', 'children'),
         Output('risk-loading-output', 'children')],
        [Input('interval-risk', 'n_intervals')]  # Refresh every 60 seconds
    )
    def update_risk_heatmap(n_intervals):
        """Update risk heat-map and metrics"""
        try:
            # Get risk data
            risk_df = get_live_risk()
            
            # Get health data for drift metrics
            health_data = get_health_data()
            drift_status = health_data.get('drift_status', 'unknown')
            drift_pct = health_data.get('drift_pct', 0.0)
            
            # Create drift badge
            drift_badge = create_drift_status_badge(drift_status, drift_pct)
            
            if risk_df.empty:
                # Return empty state
                empty_heatmap = create_empty_risk_heatmap()
                empty_badge = create_risk_status_badge("NO_DATA")
                empty_table = create_empty_risk_table()
                return (empty_heatmap, empty_badge, drift_badge, "0", "0", "0", "No Data", empty_table, "")
            
            # Create heat-map visualization
            heatmap_chart = create_risk_heatmap(risk_df)
            
            # Calculate status counts
            ok_count = len(risk_df[risk_df['status'] == 'OK'])
            warning_count = len(risk_df[risk_df['status'] == 'WARNING'])
            breach_count = len(risk_df[risk_df['status'] == 'BREACH'])
            
            # Create overall status badge
            overall_status = "BREACH" if breach_count > 0 else "WARNING" if warning_count > 0 else "OK"
            status_badge = create_risk_status_badge(overall_status)
            
            # Create details table
            details_table = create_risk_details_table(risk_df)
            
            # Last updated timestamp
            from datetime import datetime
            last_updated = datetime.now().strftime("%H:%M:%S")
            
            return (heatmap_chart, status_badge, drift_badge, str(ok_count), str(warning_count), 
                   str(breach_count), last_updated, details_table, "")
            
        except Exception as e:
            logger.error(f"Failed to update risk heat-map: {e}")
            error_heatmap = create_error_risk_heatmap(str(e))
            error_badge = create_risk_status_badge("ERROR")
            error_drift_badge = create_drift_status_badge("ERROR", 0.0)
            error_table = create_error_risk_table(str(e))
            return (error_heatmap, error_badge, error_drift_badge, "Error", "Error", "Error", "Error", error_table, "")


def create_risk_heatmap(risk_df: pd.DataFrame) -> dcc.Graph:
    """
    Create risk heat-map visualization
    
    Args:
        risk_df: DataFrame with risk metrics data
        
    Returns:
        Dash Graph component with heat-map
    """
    
    # Prepare data for heat-map
    metrics = risk_df['metric'].tolist()
    utilizations = risk_df['utilization'].tolist()
    values = risk_df['value'].tolist()
    limits = risk_df['limit'].tolist()
    
    # Create heat-map as single row
    z_data = [utilizations]  # Single row with all utilizations
    
    # Create custom hover text
    hover_text = []
    hover_row = []
    for i, metric in enumerate(metrics):
        util_pct = utilizations[i] * 100
        hover_row.append(
            f"<b>{metric}</b><br>" +
            f"Value: {values[i]:.2f}<br>" +
            f"Limit: {limits[i]:.2f}<br>" +
            f"Utilization: {util_pct:.1f}%"
        )
    hover_text.append(hover_row)
    
    # Create annotations for cell text
    annotations = []
    for i, metric in enumerate(metrics):
        util_pct = utilizations[i] * 100
        annotations.append(
            dict(
                x=i, y=0,
                text=f"<b>{metric}</b><br>{util_pct:.0f}%",
                showarrow=False,
                font=dict(color="white", size=10),
                xref="x", yref="y"
            )
        )
    
    # Create heat-map figure
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=metrics,
        y=["Risk Metrics"],
        colorscale=[
            [0, "#2ecc71"],      # Green for < 70%
            [0.7, "#f1c40f"],    # Yellow for 70-90%
            [0.9, "#e67e22"],    # Orange for 90%+
            [1, "#e74c3c"]       # Red for 100%+
        ],
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text,
        colorbar=dict(
            title="Risk Utilization",
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=0.25,
            tickformat=".0%"
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text="Risk Metrics Utilization", x=0.5, font=dict(size=16)),
        xaxis=dict(
            title="Risk Metrics",
            tickangle=45,
            side="bottom"
        ),
        yaxis=dict(
            title="",
            showticklabels=False
        ),
        height=250,
        margin=dict(l=50, r=100, t=60, b=100),
        plot_bgcolor='white',
        paper_bgcolor='white',
        annotations=annotations
    )
    
    return dcc.Graph(
        figure=fig,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'zoom2d']
        }
    )


def create_risk_status_badge(status: str) -> html.Div:
    """Create risk status badge"""
    
    if status == "OK":
        badge_class = "badge bg-success"
        icon_class = "fas fa-check-circle me-1"
        text = "Risk OK"
    elif status == "WARNING":
        badge_class = "badge bg-warning"
        icon_class = "fas fa-exclamation-triangle me-1"
        text = "Risk Warning"
    elif status == "BREACH":
        badge_class = "badge bg-danger"
        icon_class = "fas fa-times-circle me-1"
        text = "Risk Breach"
    elif status == "NO_DATA":
        badge_class = "badge bg-secondary"
        icon_class = "fas fa-question-circle me-1"
        text = "No Risk Data"
    else:  # ERROR
        badge_class = "badge bg-dark"
        icon_class = "fas fa-exclamation-circle me-1"
        text = "Risk Error"
    
    return html.H5([
        html.I(className=icon_class),
        text
    ], className=badge_class)


def create_risk_details_table(risk_df: pd.DataFrame) -> html.Div:
    """Create risk details table"""
    
    table_rows = []
    for _, row in risk_df.iterrows():
        util_pct = row['utilization'] * 100
        
        # Color code the utilization cell
        if row['status'] == 'OK':
            util_class = "text-success"
        elif row['status'] == 'WARNING':
            util_class = "text-warning"
        else:
            util_class = "text-danger"
        
        table_rows.append(
            html.Tr([
                html.Td(row['metric'], className="fw-bold"),
                html.Td(f"{row['value']:.2f}"),
                html.Td(f"{row['limit']:.2f}"),
                html.Td(f"{util_pct:.1f}%", className=util_class),
                html.Td(
                    html.Span(row['status'], className=f"badge bg-{'success' if row['status'] == 'OK' else 'warning' if row['status'] == 'WARNING' else 'danger'}")
                )
            ])
        )
    
    return html.Div([
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Metric"),
                    html.Th("Current Value"),
                    html.Th("Limit"),
                    html.Th("Utilization"),
                    html.Th("Status")
                ])
            ]),
            html.Tbody(table_rows)
        ], className="table table-sm table-striped")
    ])


def create_empty_risk_heatmap() -> html.Div:
    """Create empty state when no risk data is available"""
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-shield-alt fa-3x text-muted mb-3"),
            html.H5("No Risk Data Available", className="text-muted"),
            html.P("Risk metrics will appear here when trading data is available.", 
                   className="text-muted")
        ], className="text-center py-5")
    ])


def create_empty_risk_table() -> html.Div:
    """Create empty risk details table"""
    
    return html.Div([
        html.P("No risk metrics to display", className="text-muted text-center py-3")
    ])


def create_error_risk_heatmap(error_msg: str) -> html.Div:
    """Create error state when risk data loading fails"""
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-exclamation-triangle fa-3x text-danger mb-3"),
            html.H5("Error Loading Risk Data", className="text-danger"),
            html.P(f"Failed to load risk metrics: {error_msg}", 
                   className="text-muted small")
        ], className="text-center py-5")
    ])


def create_error_risk_table(error_msg: str) -> html.Div:
    """Create error risk details table"""
    
    return html.Div([
        html.P(f"Error loading risk details: {error_msg}", 
               className="text-danger text-center py-3")
    ])


def create_drift_status_badge(status: str, drift_pct: float) -> html.Div:
    """
    Create drift status badge with color coding
    
    Args:
        status: Drift status ('OK', 'WARN', 'BREACH', 'unknown', 'ERROR')
        drift_pct: Drift percentage
        
    Returns:
        Dash component with drift badge
    """
    
    if status == "OK":
        badge_class = "badge bg-success"
        icon_class = "fas fa-chart-line me-1"
        text = f"Drift OK ({drift_pct:.1f}%)"
    elif status == "WARN":
        badge_class = "badge bg-warning"
        icon_class = "fas fa-exclamation-triangle me-1"
        text = f"Drift Warning ({drift_pct:.1f}%)"
    elif status == "BREACH":
        badge_class = "badge bg-danger"
        icon_class = "fas fa-times-circle me-1"
        text = f"Drift Breach ({drift_pct:.1f}%)"
    elif status == "unknown":
        badge_class = "badge bg-secondary"
        icon_class = "fas fa-question-circle me-1"
        text = "Drift Unknown"
    else:  # ERROR
        badge_class = "badge bg-dark"
        icon_class = "fas fa-exclamation-circle me-1"
        text = "Drift Error"
    
    return html.H6([
        html.I(className=icon_class),
        text
    ], className=badge_class)