"""
Equity curve tab layout and callbacks for Dash dashboard
Displays portfolio performance with cumulative and daily P&L charts
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html, Input, Output, callback

from ..query import get_nav_data

logger = logging.getLogger(__name__)


def create_equity_layout():
    """
    Create the equity curve tab layout
    
    Returns:
        Dash layout components for equity curve visualization
    """
    
    return html.Div([
        # Header row with title and controls
        html.Div([
            html.Div([
                html.H4([
                    html.I(className="fas fa-chart-line me-2"),
                    "Portfolio Equity Curve"
                ], className="mb-0")
            ], className="col"),
            html.Div([
                html.Label("View:", className="form-label me-2"),
                dcc.Dropdown(
                    id='equity-view-dropdown',
                    options=[
                        {'label': '📈 Cumulative P&L', 'value': 'cumulative'},
                        {'label': '📊 Daily P&L', 'value': 'daily'}
                    ],
                    value='cumulative',
                    className="form-select",
                    style={'width': '200px'}
                )
            ], className="col-auto d-flex align-items-center")
        ], className="row mb-3"),
        
        # Performance metrics cards
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-dollar-sign fa-2x text-success mb-2"),
                        html.H5("Total Return", className="mb-1"),
                        html.H4(id="total-return-value", className="text-success mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-bar fa-2x text-info mb-2"),
                        html.H5("Win Rate", className="mb-1"),
                        html.H4(id="win-rate-value", className="text-info mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-arrow-down fa-2x text-warning mb-2"),
                        html.H5("Max Drawdown", className="mb-1"),
                        html.H4(id="max-drawdown-value", className="text-warning mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-calendar fa-2x text-secondary mb-2"),
                        html.H5("Days Active", className="mb-1"),
                        html.H4(id="days-active-value", className="text-secondary mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3")
        ], className="row mb-4"),
        
        # Main chart
        html.Div([
            html.Div([
                dcc.Graph(
                    id='equity-curve-chart',
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                    },
                    style={'height': '500px'}
                )
            ], className="metric-card")
        ], className="row mb-4"),
        
        # Performance statistics table
        html.Div([
            html.Div([
                html.H5([
                    html.I(className="fas fa-table me-2"),
                    "Performance Statistics"
                ], className="mb-3"),
                html.Div(id="performance-stats-table")
            ], className="metric-card")
        ], className="row")
        
    ])


def register_equity_callbacks():
    """Register callbacks for equity curve tab"""
    
    @callback(
        [Output('equity-curve-chart', 'figure'),
         Output('total-return-value', 'children'),
         Output('win-rate-value', 'children'),
         Output('max-drawdown-value', 'children'),
         Output('days-active-value', 'children'),
         Output('performance-stats-table', 'children')],
        [Input('equity-view-dropdown', 'value'),
         Input('interval-health', 'n_intervals')]  # Refresh with health check interval
    )
    def update_equity_curve(view_type, n_intervals):
        """Update equity curve chart and performance metrics"""
        try:
            # Get NAV data (2 years)
            nav_data = get_nav_data(days=730)
            
            if nav_data.empty:
                # Return empty chart and zero metrics
                empty_fig = create_empty_chart(view_type)
                return (empty_fig, "$0.00", "0.0%", "$0.00", "0", 
                       html.P("No trading data available", className="text-muted"))
            
            # Create the chart
            fig = create_equity_chart(nav_data, view_type)
            
            # Calculate performance metrics
            metrics = calculate_performance_metrics(nav_data)
            
            # Format metrics for display
            total_return = f"${metrics['total_return']:,.2f}"
            win_rate = f"{metrics['win_rate']:.1f}%"
            max_drawdown = f"${metrics['max_drawdown']:,.2f}"
            days_active = str(metrics['days_active'])
            
            # Create performance stats table
            stats_table = create_performance_stats_table(metrics)
            
            return (fig, total_return, win_rate, max_drawdown, days_active, stats_table)
            
        except Exception as e:
            logger.error(f"Failed to update equity curve: {e}")
            error_fig = create_error_chart(str(e))
            return (error_fig, "Error", "Error", "Error", "Error", 
                   html.P(f"Error loading data: {str(e)}", className="text-danger"))


def create_equity_chart(nav_data: pd.DataFrame, view_type: str) -> go.Figure:
    """
    Create equity curve chart
    
    Args:
        nav_data: DataFrame with date, daily_pnl, cumulative_pnl columns
        view_type: 'cumulative' or 'daily'
        
    Returns:
        Plotly figure object
    """
    
    fig = go.Figure()
    
    if view_type == 'cumulative':
        # Cumulative P&L line chart
        fig.add_trace(go.Scatter(
            x=nav_data['date'],
            y=nav_data['cumulative_pnl'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(width=2),
            hovertemplate='<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        title = "Portfolio Cumulative P&L"
        yaxis_title = "Cumulative P&L ($)"
        
    else:  # daily
        # Daily P&L bar chart
        colors = ['green' if x >= 0 else 'red' for x in nav_data['daily_pnl']]
        
        fig.add_trace(go.Bar(
            x=nav_data['date'],
            y=nav_data['daily_pnl'],
            name='Daily P&L',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Daily P&L: $%{y:,.2f}<extra></extra>'
        ))
        
        title = "Daily P&L"
        yaxis_title = "Daily P&L ($)"
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=60, r=60, t=60, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Style axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black',
        tickformat='$,.0f'
    )
    
    return fig


def create_empty_chart(view_type: str) -> go.Figure:
    """Create empty chart when no data is available"""
    
    fig = go.Figure()
    
    title = "Portfolio Cumulative P&L" if view_type == 'cumulative' else "Daily P&L"
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="P&L ($)",
        margin=dict(l=60, r=60, t=60, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        annotations=[
            dict(
                text="No trading data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="gray")
            )
        ]
    )
    
    return fig


def create_error_chart(error_msg: str) -> go.Figure:
    """Create error chart when data loading fails"""
    
    fig = go.Figure()
    
    fig.update_layout(
        title=dict(text="Error Loading Data", x=0.5, font=dict(size=16)),
        margin=dict(l=60, r=60, t=60, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        annotations=[
            dict(
                text=f"Error: {error_msg}",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="red")
            )
        ]
    )
    
    return fig


def calculate_performance_metrics(nav_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate performance metrics from NAV data
    
    Args:
        nav_data: DataFrame with date, daily_pnl, cumulative_pnl columns
        
    Returns:
        Dictionary with performance metrics
    """
    
    if nav_data.empty:
        return {
            'total_return': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'days_active': 0,
            'avg_daily_pnl': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'best_day': 0.0,
            'worst_day': 0.0
        }
    
    # Basic metrics
    total_return = nav_data['cumulative_pnl'].iloc[-1]
    days_active = len(nav_data)
    
    # Win rate (percentage of profitable days)
    profitable_days = (nav_data['daily_pnl'] > 0).sum()
    win_rate = (profitable_days / days_active * 100) if days_active > 0 else 0
    
    # Max drawdown calculation
    running_max = nav_data['cumulative_pnl'].expanding().max()
    drawdowns = nav_data['cumulative_pnl'] - running_max
    max_drawdown = drawdowns.min()
    
    # Daily statistics
    avg_daily_pnl = nav_data['daily_pnl'].mean()
    volatility = nav_data['daily_pnl'].std()
    best_day = nav_data['daily_pnl'].max()
    worst_day = nav_data['daily_pnl'].min()
    
    # Sharpe ratio (simplified, assuming 0% risk-free rate)
    sharpe_ratio = (avg_daily_pnl / volatility * (252 ** 0.5)) if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'days_active': days_active,
        'avg_daily_pnl': avg_daily_pnl,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'best_day': best_day,
        'worst_day': worst_day
    }


def create_performance_stats_table(metrics: Dict[str, Any]) -> html.Div:
    """
    Create performance statistics table
    
    Args:
        metrics: Dictionary with performance metrics
        
    Returns:
        HTML table with performance statistics
    """
    
    stats = [
        ("Average Daily P&L", f"${metrics['avg_daily_pnl']:,.2f}"),
        ("Volatility", f"${metrics['volatility']:,.2f}"),
        ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
        ("Best Day", f"${metrics['best_day']:,.2f}"),
        ("Worst Day", f"${metrics['worst_day']:,.2f}")
    ]
    
    table_rows = []
    for stat_name, stat_value in stats:
        table_rows.append(
            html.Tr([
                html.Td(stat_name, className="fw-bold"),
                html.Td(stat_value, className="text-end")
            ])
        )
    
    return html.Div([
        html.Table([
            html.Tbody(table_rows)
        ], className="table table-sm")
    ])