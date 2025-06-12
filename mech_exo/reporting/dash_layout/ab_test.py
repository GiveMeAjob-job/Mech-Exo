"""
A/B Test Dashboard Tab

Displays canary vs base allocation performance with equity curves,
Sharpe ratio comparison, and canary status badge.
"""

import logging
from typing import Dict, Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html, Input, Output, callback
from datetime import datetime

from ..query import get_canary_equity, get_base_equity, get_ab_test_summary

logger = logging.getLogger(__name__)


def create_ab_test_layout():
    """Create A/B test dashboard layout"""
    
    return html.Div([
        # Header section
        html.Div([
            html.Div([
                html.H3([
                    html.I(className="fas fa-flask me-2"),
                    "A/B Test Dashboard"
                ], className="mb-1"),
                html.P("Canary vs Base allocation performance comparison", 
                       className="text-muted mb-0")
            ], className="col"),
            html.Div([
                html.Div(id="ab-status-badge", className="text-end")
            ], className="col-auto")
        ], className="row align-items-center mb-4"),
        
        # Summary metrics cards
        html.Div([
            # Canary metrics card
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("🐤 Canary Allocation", className="card-title"),
                        html.Div(id="canary-metrics", className="mt-2")
                    ], className="card-body")
                ], className="card h-100")
            ], className="col-md-4 mb-3"),
            
            # Base metrics card
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("🏠 Base Allocation", className="card-title"),
                        html.Div(id="base-metrics", className="mt-2")
                    ], className="card-body")
                ], className="card h-100")
            ], className="col-md-4 mb-3"),
            
            # Performance comparison card
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("📊 Performance Comparison", className="card-title"),
                        html.Div(id="comparison-metrics", className="mt-2")
                    ], className="card-body")
                ], className="card h-100")
            ], className="col-md-4 mb-3")
        ], className="row"),
        
        # Charts section
        html.Div([
            # Equity curves chart
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("📈 Equity Curves (Last 180 Days)", className="card-title"),
                        html.P("NAV progression for base vs canary allocations", 
                               className="card-text text-muted"),
                        dcc.Graph(id="equity-curves-chart", style={'height': '400px'})
                    ], className="card-body")
                ], className="card")
            ], className="col-md-8 mb-3"),
            
            # Sharpe ratio chart
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("📊 Rolling Sharpe Difference", className="card-title"),
                        html.P("30-day rolling Sharpe: Canary - Base", 
                               className="card-text text-muted"),
                        dcc.Graph(id="sharpe-diff-chart", style={'height': '400px'})
                    ], className="card-body")
                ], className="card")
            ], className="col-md-4 mb-3")
        ], className="row"),
        
        # Data table section
        html.Div([
            html.Div([
                html.Div([
                    html.H5("📋 Recent Performance Data", className="card-title"),
                    html.Div(id="ab-data-table")
                ], className="card-body")
            ], className="card")
        ], className="col-12 mb-3"),
        
        # Auto-refresh interval (24 hours)
        dcc.Interval(
            id='ab-refresh-interval',
            interval=24*60*60*1000,  # 24 hours in milliseconds
            n_intervals=0
        )
        
    ], className="container-fluid")


def register_ab_test_callbacks():
    """Register callbacks for A/B test dashboard"""
    
    @callback(
        [Output('ab-status-badge', 'children'),
         Output('canary-metrics', 'children'),
         Output('base-metrics', 'children'),
         Output('comparison-metrics', 'children'),
         Output('equity-curves-chart', 'figure'),
         Output('sharpe-diff-chart', 'figure'),
         Output('ab-data-table', 'children')],
        [Input('ab-refresh-interval', 'n_intervals')]
    )
    def update_ab_dashboard(n):
        """Update all A/B dashboard components"""
        try:
            # Get summary data
            summary = get_ab_test_summary(days=180)
            
            # Status badge
            status_badge = create_status_badge(summary)
            
            # Metrics cards
            canary_metrics = create_canary_metrics_card(summary)
            base_metrics = create_base_metrics_card(summary)
            comparison_metrics = create_comparison_metrics_card(summary)
            
            # Charts
            equity_figure = create_equity_curves_chart()
            sharpe_figure = create_sharpe_diff_chart()
            
            # Data table
            data_table = create_ab_data_table()
            
            return (status_badge, canary_metrics, base_metrics, comparison_metrics,
                    equity_figure, sharpe_figure, data_table)
            
        except Exception as e:
            logger.error(f"Failed to update A/B dashboard: {e}")
            
            # Return error state
            error_msg = html.Div([
                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                f"Error loading A/B dashboard: {str(e)}"
            ], className="alert alert-danger")
            
            empty_figure = go.Figure()
            empty_figure.add_annotation(text="Error loading data", 
                                      x=0.5, y=0.5, showarrow=False)
            
            return (error_msg, error_msg, error_msg, error_msg,
                    empty_figure, empty_figure, error_msg)


def create_status_badge(summary: Dict[str, Any]) -> html.Div:
    """Create status badge for canary allocation"""
    
    status_text = summary.get('status_badge', 'UNKNOWN')
    status_color = summary.get('status_color', 'secondary')
    allocation_pct = summary.get('canary_allocation_pct', 0)
    
    # Create badge with tooltip
    badge = html.Span(
        status_text,
        className=f"badge bg-{status_color} fs-6 me-2",
        title=f"Canary allocation: {allocation_pct:.1f}% of each order"
    )
    
    # Add allocation percentage
    allocation_text = html.Small(
        f"{allocation_pct:.1f}% allocation",
        className="text-muted"
    )
    
    # Add last updated timestamp
    last_updated = summary.get('last_updated', '')
    if last_updated:
        try:
            dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            time_text = dt.strftime('%H:%M UTC')
        except:
            time_text = 'Unknown'
    else:
        time_text = 'Unknown'
    
    updated_text = html.Br()
    updated_text2 = html.Small(f"Updated: {time_text}", className="text-muted")
    
    return html.Div([
        badge,
        html.Br(),
        allocation_text,
        updated_text,
        updated_text2
    ])


def create_canary_metrics_card(summary: Dict[str, Any]) -> html.Div:
    """Create canary metrics display"""
    
    current_nav = summary.get('current_canary_nav', 0)
    total_pnl = summary.get('total_canary_pnl', 0)
    sharpe_30d = summary.get('canary_sharpe_30d', 0)
    
    return html.Div([
        html.P([
            html.Strong("Current NAV: "),
            f"${current_nav:,.0f}"
        ], className="mb-1"),
        html.P([
            html.Strong("Total P&L: "),
            html.Span(
                f"${total_pnl:+,.0f}",
                className=f"text-{'success' if total_pnl >= 0 else 'danger'}"
            )
        ], className="mb-1"),
        html.P([
            html.Strong("Sharpe (30d): "),
            html.Span(
                f"{sharpe_30d:.3f}",
                className=f"text-{'success' if sharpe_30d >= 0 else 'danger'}"
            )
        ], className="mb-0")
    ])


def create_base_metrics_card(summary: Dict[str, Any]) -> html.Div:
    """Create base metrics display"""
    
    current_nav = summary.get('current_base_nav', 0)
    total_pnl = summary.get('total_base_pnl', 0)
    sharpe_30d = summary.get('base_sharpe_30d', 0)
    
    return html.Div([
        html.P([
            html.Strong("Current NAV: "),
            f"${current_nav:,.0f}"
        ], className="mb-1"),
        html.P([
            html.Strong("Total P&L: "),
            html.Span(
                f"${total_pnl:+,.0f}",
                className=f"text-{'success' if total_pnl >= 0 else 'danger'}"
            )
        ], className="mb-1"),
        html.P([
            html.Strong("Sharpe (30d): "),
            html.Span(
                f"{sharpe_30d:.3f}",
                className=f"text-{'success' if sharpe_30d >= 0 else 'danger'}"
            )
        ], className="mb-0")
    ])


def create_comparison_metrics_card(summary: Dict[str, Any]) -> html.Div:
    """Create performance comparison display"""
    
    sharpe_diff = summary.get('sharpe_diff', 0)
    canary_outperforming = summary.get('canary_outperforming', False)
    days_analyzed = summary.get('days_analyzed', 0)
    
    # Calculate relative P&L difference
    canary_pnl = summary.get('total_canary_pnl', 0)
    base_pnl = summary.get('total_base_pnl', 0)
    pnl_diff = canary_pnl - base_pnl
    
    outperform_emoji = "🟢" if canary_outperforming else "🔴"
    outperform_text = "Outperforming" if canary_outperforming else "Underperforming"
    
    return html.Div([
        html.P([
            html.Strong("Sharpe Difference: "),
            html.Span(
                f"{sharpe_diff:+.3f}",
                className=f"text-{'success' if sharpe_diff >= 0 else 'danger'}"
            )
        ], className="mb-1"),
        html.P([
            html.Strong("P&L Difference: "),
            html.Span(
                f"${pnl_diff:+,.0f}",
                className=f"text-{'success' if pnl_diff >= 0 else 'danger'}"
            )
        ], className="mb-1"),
        html.P([
            outperform_emoji,
            html.Strong(f" {outperform_text}"),
            html.Br(),
            html.Small(f"({days_analyzed} days analyzed)", className="text-muted")
        ], className="mb-0")
    ])


def create_equity_curves_chart() -> go.Figure:
    """Create equity curves chart for base vs canary"""
    
    try:
        # Get data for both allocations
        canary_data = get_canary_equity(days=180)
        base_data = get_base_equity(days=180)
        
        fig = go.Figure()
        
        if not base_data.empty:
            fig.add_trace(go.Scatter(
                x=base_data['date'],
                y=base_data['base_nav'],
                mode='lines',
                name='Base Allocation',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Base</b><br>Date: %{x}<br>NAV: $%{y:,.0f}<extra></extra>'
            ))
        
        if not canary_data.empty:
            fig.add_trace(go.Scatter(
                x=canary_data['date'],
                y=canary_data['canary_nav'],
                mode='lines',
                name='Canary Allocation',
                line=dict(color='#ff7f0e', width=2),
                hovertemplate='<b>Canary</b><br>Date: %{x}<br>NAV: $%{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Base vs Canary Equity Curves",
            xaxis_title="Date",
            yaxis_title="NAV ($)",
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified',
            template='plotly_white',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Add horizontal line at starting NAV if data available
        if not base_data.empty and not canary_data.empty:
            start_nav = max(base_data['base_nav'].iloc[0], canary_data['canary_nav'].iloc[0])
            fig.add_hline(
                y=start_nav, 
                line_dash="dash", 
                line_color="gray",
                annotation_text="Starting NAV",
                annotation_position="bottom right"
            )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create equity curves chart: {e}")
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading equity data: {str(e)}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(template='plotly_white')
        return fig


def create_sharpe_diff_chart() -> go.Figure:
    """Create Sharpe ratio difference chart"""
    
    try:
        # Get canary data which includes sharpe_diff
        canary_data = get_canary_equity(days=180)
        
        fig = go.Figure()
        
        if not canary_data.empty and 'sharpe_diff' in canary_data.columns:
            # Create bar chart for Sharpe difference
            colors = ['green' if x >= 0 else 'red' for x in canary_data['sharpe_diff']]
            
            fig.add_trace(go.Bar(
                x=canary_data['date'],
                y=canary_data['sharpe_diff'],
                name='Sharpe Difference',
                marker_color=colors,
                hovertemplate='<b>Sharpe Difference</b><br>Date: %{x}<br>Diff: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Rolling 30d Sharpe Difference (Canary - Base)",
            xaxis_title="Date",
            yaxis_title="Sharpe Difference",
            template='plotly_white',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Add horizontal line at zero
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Neutral",
            annotation_position="top right"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create Sharpe difference chart: {e}")
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading Sharpe data: {str(e)}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(template='plotly_white')
        return fig


def create_ab_data_table() -> html.Div:
    """Create data table showing recent A/B performance"""
    
    try:
        # Get recent data (last 30 days)
        canary_data = get_canary_equity(days=30)
        
        if canary_data.empty:
            return html.Div([
                html.P("No recent A/B performance data available", 
                       className="text-muted text-center")
            ])
        
        # Create table data
        table_data = []
        for _, row in canary_data.tail(10).iterrows():  # Last 10 days
            table_data.append(html.Tr([
                html.Td(row['date'].strftime('%Y-%m-%d')),
                html.Td(f"${row['canary_nav']:,.0f}"),
                html.Td(f"${row.get('base_nav', 0):,.0f}"),
                html.Td(f"${row['canary_pnl']:+,.0f}", 
                        className=f"text-{'success' if row['canary_pnl'] >= 0 else 'danger'}"),
                html.Td(f"{row['canary_sharpe_30d']:.3f}",
                        className=f"text-{'success' if row['canary_sharpe_30d'] >= 0 else 'danger'}"),
                html.Td(f"{row.get('sharpe_diff', 0):+.3f}",
                        className=f"text-{'success' if row.get('sharpe_diff', 0) >= 0 else 'danger'}")
            ]))
        
        table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Date"),
                    html.Th("Canary NAV"),
                    html.Th("Base NAV"),
                    html.Th("Canary P&L"),
                    html.Th("Canary Sharpe"),
                    html.Th("Sharpe Diff")
                ])
            ]),
            html.Tbody(table_data)
        ], className="table table-striped table-sm")
        
        return html.Div([
            html.P(f"Showing last {len(table_data)} trading days", 
                   className="text-muted small mb-2"),
            table
        ])
        
    except Exception as e:
        logger.error(f"Failed to create A/B data table: {e}")
        return html.Div([
            html.P(f"Error loading table data: {str(e)}", 
                   className="text-danger")
        ])