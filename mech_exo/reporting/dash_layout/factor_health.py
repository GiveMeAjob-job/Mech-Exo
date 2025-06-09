"""
Factor Health Dashboard Tab

Displays alpha decay metrics with color-coded visualizations showing
which factors are losing predictive power over time.
"""

import logging
from datetime import datetime
import pandas as pd
import numpy as np

import dash
from dash import html, dcc, dash_table, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px

from ..query import get_factor_decay_latest

logger = logging.getLogger(__name__)


def create_factor_health_layout():
    """
    Create the Factor Health dashboard tab layout
    
    Returns:
        Dash HTML component tree for the Factor Health tab
    """
    return html.Div([
        # Header section
        html.Div([
            html.H2("Factor Health Monitor", className="dashboard-title"),
            html.P([
                "Real-time monitoring of factor alpha decay. ",
                html.Strong("Green"), " factors (>30d half-life) are healthy, ",
                html.Strong("Yellow", style={'color': '#ff9500'}), " factors (10-30d) need monitoring, ",
                html.Strong("Red", style={'color': '#ff0000'}), " factors (<10d) require immediate attention."
            ], className="dashboard-subtitle")
        ], className="header-section"),
        
        # Data storage for caching
        dcc.Store(id='factor-decay-data'),
        
        # Main content row
        html.Div([
            # Left column - Data table
            html.Div([
                html.H4("Factor Decay Metrics", className="section-title"),
                html.Div(id="factor-decay-table-container"),
            ], className="col-lg-8"),
            
            # Right column - Summary stats
            html.Div([
                html.H4("Health Summary", className="section-title"),
                html.Div(id="factor-health-summary"),
                
                html.Br(),
                
                html.H5("Thresholds", className="section-subtitle"),
                html.Ul([
                    html.Li([html.Strong("Healthy: "), "> 30 days half-life"], style={'color': '#28a745'}),
                    html.Li([html.Strong("Warning: "), "10-30 days half-life"], style={'color': '#ffc107'}),
                    html.Li([html.Strong("Critical: "), "< 10 days half-life"], style={'color': '#dc3545'})
                ]),
                
                html.Br(),
                
                html.P([
                    html.Small([
                        "Half-life measures how quickly a factor's predictive power decays. ",
                        "IC (Information Coefficient) shows current correlation with returns."
                    ])
                ], className="text-muted")
                
            ], className="col-lg-4")
        ], className="row"),
        
        html.Hr(),
        
        # Heat-map section
        html.Div([
            html.H4("Factor Health Heat-map", className="section-title"),
            html.Div(id="factor-heatmap-container"),
        ], className="heatmap-section"),
        
        # Auto-refresh interval (every 6 hours)
        dcc.Interval(
            id='factor-health-interval',
            interval=6*60*60*1000,  # 6 hours in milliseconds
            n_intervals=0
        ),
        
        # Last updated timestamp
        html.Div([
            html.P(id="factor-health-last-updated", className="text-muted small")
        ], className="text-center mt-3")
        
    ], className="factor-health-tab")


@callback(
    Output('factor-decay-data', 'data'),
    Input('factor-health-interval', 'n_intervals')
)
def update_factor_decay_data(n_intervals):
    """
    Update factor decay data from database
    
    Args:
        n_intervals: Number of interval triggers
        
    Returns:
        Dictionary with factor decay data
    """
    try:
        logger.info(f"Refreshing factor decay data (interval: {n_intervals})")
        
        # Get latest factor decay metrics
        decay_df = get_factor_decay_latest()
        
        if decay_df.empty:
            logger.warning("No factor decay data available")
            return {
                'data': [],
                'timestamp': datetime.now().isoformat(),
                'status': 'no_data'
            }
        
        # Convert to dictionary for storage
        data_dict = decay_df.to_dict('records')
        
        logger.info(f"Loaded {len(data_dict)} factor decay records")
        
        return {
            'data': data_dict,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Failed to update factor decay data: {e}")
        return {
            'data': [],
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        }


@callback(
    Output('factor-decay-table-container', 'children'),
    Input('factor-decay-data', 'data')
)
def update_factor_decay_table(factor_data):
    """
    Update the factor decay data table
    
    Args:
        factor_data: Factor decay data from store
        
    Returns:
        Dash DataTable component
    """
    try:
        if not factor_data or factor_data['status'] != 'success' or not factor_data['data']:
            return html.Div([
                html.P("No factor decay data available", className="text-muted text-center"),
                html.P("Run the alpha decay flow to generate metrics", className="text-muted text-center small")
            ])
        
        # Convert back to DataFrame
        df = pd.DataFrame(factor_data['data'])
        
        # Prepare table data
        table_data = []
        for _, row in df.iterrows():
            table_data.append({
                'Factor': row['factor'],
                'Half-life (d)': f"{row['half_life']:.2f}" if pd.notna(row['half_life']) else 'N/A',
                'Latest IC': f"{row['latest_ic']:.4f}" if pd.notna(row['latest_ic']) else 'N/A',
                'IC Mean': f"{row['ic_mean']:.4f}" if pd.notna(row['ic_mean']) else 'N/A',
                'IC Trend': f"{row['ic_trend']:.6f}" if pd.notna(row['ic_trend']) else 'N/A',
                'Data Points': int(row['data_points']) if pd.notna(row['data_points']) else 0,
                'Status': row['status'],
                'Color': row['color_status']
            })
        
        # Create DataTable with conditional formatting
        data_table = dash_table.DataTable(
            data=table_data,
            columns=[
                {'name': 'Factor', 'id': 'Factor', 'type': 'text'},
                {'name': 'Half-life (d)', 'id': 'Half-life (d)', 'type': 'numeric'},
                {'name': 'Latest IC', 'id': 'Latest IC', 'type': 'numeric'},
                {'name': 'IC Mean', 'id': 'IC Mean', 'type': 'numeric'},
                {'name': 'IC Trend', 'id': 'IC Trend', 'type': 'numeric'},
                {'name': 'Data Points', 'id': 'Data Points', 'type': 'numeric'},
                {'name': 'Status', 'id': 'Status', 'type': 'text'}
            ],
            sort_action='native',
            sort_by=[{'column_id': 'Half-life (d)', 'direction': 'asc'}],  # Sort by half-life ascending
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'fontFamily': 'Arial, sans-serif'
            },
            style_header={
                'backgroundColor': '#f8f9fa',
                'fontWeight': 'bold',
                'border': '1px solid #dee2e6'
            },
            style_data_conditional=[
                # Color code rows based on half-life
                {
                    'if': {'filter_query': '{Color} = green'},
                    'backgroundColor': '#d4edda',
                    'border': '1px solid #c3e6cb'
                },
                {
                    'if': {'filter_query': '{Color} = yellow'},
                    'backgroundColor': '#fff3cd',
                    'border': '1px solid #ffeaa7'
                },
                {
                    'if': {'filter_query': '{Color} = red'},
                    'backgroundColor': '#f8d7da',
                    'border': '1px solid #f5c6cb'
                },
                {
                    'if': {'filter_query': '{Color} = unknown'},
                    'backgroundColor': '#f8f9fa',
                    'border': '1px solid #dee2e6'
                }
            ],
            # Hide the Color column
            hidden_columns=['Color']
        )
        
        return data_table
        
    except Exception as e:
        logger.error(f"Failed to update factor decay table: {e}")
        return html.Div([
            html.P(f"Error loading factor decay table: {e}", className="text-danger")
        ])


@callback(
    Output('factor-health-summary', 'children'),
    Input('factor-decay-data', 'data')
)
def update_factor_health_summary(factor_data):
    """
    Update the factor health summary statistics
    
    Args:
        factor_data: Factor decay data from store
        
    Returns:
        HTML components with summary statistics
    """
    try:
        if not factor_data or factor_data['status'] != 'success' or not factor_data['data']:
            return html.Div([
                html.P("No summary available", className="text-muted")
            ])
        
        # Convert back to DataFrame
        df = pd.DataFrame(factor_data['data'])
        
        # Calculate summary statistics
        total_factors = len(df)
        
        # Count by health status
        green_count = len(df[df['color_status'] == 'green'])
        yellow_count = len(df[df['color_status'] == 'yellow'])
        red_count = len(df[df['color_status'] == 'red'])
        unknown_count = len(df[df['color_status'] == 'unknown'])
        
        # Calculate average metrics
        avg_half_life = df['half_life'].mean() if not df['half_life'].isna().all() else 0
        avg_ic = df['latest_ic'].mean() if not df['latest_ic'].isna().all() else 0
        
        # Create summary cards
        summary_cards = html.Div([
            # Total factors
            html.Div([
                html.H5(str(total_factors), className="card-value"),
                html.P("Total Factors", className="card-label")
            ], className="summary-card"),
            
            # Healthy factors
            html.Div([
                html.H5(str(green_count), className="card-value text-success"),
                html.P("Healthy", className="card-label")
            ], className="summary-card"),
            
            # Warning factors
            html.Div([
                html.H5(str(yellow_count), className="card-value text-warning"),
                html.P("Warning", className="card-label")
            ], className="summary-card"),
            
            # Critical factors
            html.Div([
                html.H5(str(red_count), className="card-value text-danger"),
                html.P("Critical", className="card-label")
            ], className="summary-card"),
            
        ], className="summary-cards"),
        
        html.Hr(),
        
        # Average metrics
        html.Div([
            html.P([
                html.Strong("Avg Half-life: "), 
                f"{avg_half_life:.1f} days"
            ]),
            html.P([
                html.Strong("Avg IC: "), 
                f"{avg_ic:.3f}"
            ])
        ])
        
        return summary_cards
        
    except Exception as e:
        logger.error(f"Failed to update factor health summary: {e}")
        return html.Div([
            html.P(f"Error loading summary: {e}", className="text-danger")
        ])


@callback(
    Output('factor-heatmap-container', 'children'),
    Input('factor-decay-data', 'data')
)
def update_factor_heatmap(factor_data):
    """
    Update the factor health heat-map
    
    Args:
        factor_data: Factor decay data from store
        
    Returns:
        Plotly graph component with heat-map
    """
    try:
        if not factor_data or factor_data['status'] != 'success' or not factor_data['data']:
            return html.Div([
                html.P("No data available for heat-map", className="text-muted text-center")
            ])
        
        # Convert back to DataFrame
        df = pd.DataFrame(factor_data['data'])
        
        if len(df) == 0:
            return html.Div([
                html.P("No factors to display", className="text-muted text-center")
            ])
        
        # Prepare data for heat-map
        factors = df['factor'].tolist()
        half_lives = df['half_life'].fillna(90).tolist()  # Fill NaN with 90 (max)
        ic_values = df['latest_ic'].fillna(0).tolist()
        
        # Create hover text
        hover_text = []
        for i, row in df.iterrows():
            hover_info = (
                f"Factor: {row['factor']}<br>"
                f"Half-life: {row['half_life']:.1f} days<br>"
                f"Latest IC: {row['latest_ic']:.3f}<br>"
                f"IC Mean: {row['ic_mean']:.3f}<br>"
                f"Status: {row['status']}"
            )
            hover_text.append(hover_info)
        
        # Create heat-map figure
        fig = go.Figure(data=go.Heatmap(
            x=factors,
            y=['Half-life'],
            z=[half_lives],
            text=[f"{hl:.1f}d" for hl in half_lives],
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale=[
                [0.0, '#dc3545'],    # Red for low half-life
                [0.11, '#dc3545'],   # Red up to 10 days  
                [0.33, '#ffc107'],   # Yellow for 10-30 days
                [0.33, '#ffc107'],   # Yellow
                [1.0, '#28a745']     # Green for high half-life
            ],
            colorbar=dict(
                title="Half-life (days)",
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=15
            ),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text
        ))
        
        fig.update_layout(
            title="Factor Health Heat-map",
            xaxis_title="Factors",
            yaxis_title="",
            height=200,
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(size=12)
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return dcc.Graph(
            figure=fig,
            config={'displayModeBar': False}
        )
        
    except Exception as e:
        logger.error(f"Failed to update factor heatmap: {e}")
        return html.Div([
            html.P(f"Error creating heat-map: {e}", className="text-danger")
        ])


@callback(
    Output('factor-health-last-updated', 'children'),
    Input('factor-decay-data', 'data')
)
def update_last_updated_timestamp(factor_data):
    """
    Update the last updated timestamp
    
    Args:
        factor_data: Factor decay data from store
        
    Returns:
        HTML paragraph with timestamp
    """
    try:
        if not factor_data:
            return "Last updated: Never"
        
        timestamp = factor_data.get('timestamp')
        if timestamp:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            return f"Last updated: {formatted_time}"
        else:
            return "Last updated: Unknown"
            
    except Exception as e:
        logger.error(f"Failed to update timestamp: {e}")
        return "Last updated: Error"