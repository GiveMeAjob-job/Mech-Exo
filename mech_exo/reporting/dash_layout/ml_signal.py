"""
ML Signal Dashboard Tab

Displays ML-weighted portfolio performance, prediction accuracy,
and current ML scores with auto-refresh functionality.
"""

import logging
from datetime import datetime

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table, dcc, html, Input, Output, callback

logger = logging.getLogger(__name__)


def create_ml_signal_tab():
    """Create the ML Signal dashboard tab layout."""
    return html.Div([
        # Auto-refresh interval (every hour)
        dcc.Interval(
            id='ml-signal-interval',
            interval=60*60*1000,  # 1 hour in milliseconds
            n_intervals=0
        ),
        
        # Store for caching equity data
        dcc.Store(id='ml-equity-store'),
        
        # Header with ML Weight Badge
        html.Div([
            html.Div([
                html.Div([
                    html.H3("🤖 ML Signal Dashboard", className="mb-2"),
                    html.P(
                        "ML-enhanced portfolio performance and prediction analytics", 
                        className="text-muted mb-0"
                    )
                ], className="col"),
                html.Div([
                    html.Div(id="ml-weight-badge", className="text-end")
                ], className="col-auto")
            ], className="row"),
        ], className="mb-4"),
        
        # Key metrics cards
        html.Div([
            html.Div([
                html.Div([
                    html.H5("ML Alpha", className="card-title"),
                    html.H3(id="ml-alpha-metric", children="--", className="text-success"),
                    html.P("vs Baseline", className="text-muted small")
                ], className="card-body text-center")
            ], className="card", style={"width": "100%"}),
            
            html.Div([
                html.Div([
                    html.H5("Prediction Accuracy", className="card-title"),
                    html.H3(id="ml-accuracy-metric", children="--", className="text-info"),
                    html.P("30-day hit rate", className="text-muted small")
                ], className="card-body text-center")
            ], className="card", style={"width": "100%"}),
            
            html.Div([
                html.Div([
                    html.H5("ML Weight", className="card-title"),
                    html.H3(id="ml-weight-metric", children="--", className="text-warning"),
                    html.P("in final scores", className="text-muted small")
                ], className="card-body text-center")
            ], className="card", style={"width": "100%"}),
            
            html.Div([
                html.Div([
                    html.H5("Active Predictions", className="card-title"),
                    html.H3(id="ml-predictions-metric", children="--", className="text-primary"),
                    html.P("today's symbols", className="text-muted small")
                ], className="card-body text-center")
            ], className="card", style={"width": "100%"})
        ], className="row mb-4", style={"display": "flex", "gap": "1rem"}),
        
        # Live Validation Metrics Card
        html.Div([
            html.Div([
                html.Div([
                    html.H5("🎯 Live Signal Validation", className="card-title"),
                    html.Div([
                        # Performance badge
                        html.Div(id="ml-performance-badge", className="mb-3"),
                        # Mini line chart
                        dcc.Graph(id="ml-live-metrics-chart", style={"height": "250px"})
                    ])
                ], className="card-body")
            ], className="card")
        ], className="mb-4"),
        
        # Charts row
        html.Div([
            # Equity curve (left column, 8/12 width)
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("ML-Weighted Equity Curve", className="card-title"),
                        dcc.Graph(id="ml-equity-chart")
                    ], className="card-body")
                ], className="card")
            ], className="col-md-8"),
            
            # Confusion matrix (right column, 4/12 width)
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("Prediction Accuracy Matrix", className="card-title"),
                        dcc.Graph(id="ml-confusion-matrix")
                    ], className="card-body")
                ], className="card")
            ], className="col-md-4")
        ], className="row mb-4"),
        
        # ML Scores table
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Today's ML Predictions", className="card-title mb-3"),
                    html.Div(id="ml-scores-table-container")
                ], className="card-body")
            ], className="card")
        ], className="mb-4"),
        
        # Footer with last update time
        html.Div([
            html.P(id="ml-last-updated", className="text-muted small text-center")
        ])
    ])


@callback(
    Output('ml-weight-badge', 'children'),
    [Input('ml-signal-interval', 'n_intervals')]
)
def update_ml_weight_badge(n_intervals):
    """Update ML weight badge in header."""
    try:
        from ..query import get_current_ml_weight_info
        
        weight_info = get_current_ml_weight_info()
        
        # Create badge with tooltip
        badge = html.Div([
            html.Span([
                html.I(className="fas fa-robot me-1"),
                "ML Weight"
            ], className="badge-label small text-muted d-block"),
            html.Span(
                weight_info['weight_percentage'],
                className=f"badge bg-{weight_info['badge_color']} fs-6",
                title=weight_info['tooltip_info'],
                **{'data-bs-toggle': 'tooltip', 'data-bs-placement': 'left'}
            )
        ], className="text-center")
        
        return badge
        
    except Exception as e:
        logger.error(f"Failed to update ML weight badge: {e}")
        return html.Div([
            html.Span("ML Weight", className="badge-label small text-muted d-block"),
            html.Span("N/A", className="badge bg-secondary fs-6")
        ], className="text-center")


@callback(
    [Output('ml-equity-store', 'data'),
     Output('ml-alpha-metric', 'children'),
     Output('ml-accuracy-metric', 'children'),
     Output('ml-weight-metric', 'children'),
     Output('ml-predictions-metric', 'children'),
     Output('ml-last-updated', 'children')],
    [Input('ml-signal-interval', 'n_intervals')]
)
def update_ml_metrics(n_intervals):
    """Update ML metrics and cache equity data."""
    try:
        from ..query import get_ml_signal_equity, get_ml_scores_today, get_ml_confusion_matrix
        
        # Get equity data
        equity_data = get_ml_signal_equity(days=365)
        
        # Calculate ML alpha
        if not equity_data.empty and len(equity_data) > 1:
            final_ml = equity_data['ml_weighted_equity'].iloc[-1]
            final_baseline = equity_data['baseline_equity'].iloc[-1]
            ml_alpha = ((final_ml / final_baseline) - 1) * 100
            alpha_display = f"+{ml_alpha:.1f}%" if ml_alpha > 0 else f"{ml_alpha:.1f}%"
        else:
            alpha_display = "--"
        
        # Get prediction accuracy from confusion matrix
        confusion_data = get_ml_confusion_matrix()
        if not confusion_data.empty and 'All' in confusion_data.index:
            total_predictions = confusion_data.loc['All', 'All']
            correct_predictions = (
                confusion_data.loc['Q4', 'Positive'] + confusion_data.loc['Q5', 'Positive'] +
                confusion_data.loc['Q1', 'Negative'] + confusion_data.loc['Q2', 'Negative']
            ) if all(q in confusion_data.index for q in ['Q1', 'Q2', 'Q4', 'Q5']) else 0
            accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            accuracy_display = f"{accuracy:.1f}%"
        else:
            accuracy_display = "--"
        
        # Get today's ML scores
        ml_scores = get_ml_scores_today()
        predictions_count = len(ml_scores) if not ml_scores.empty else 0
        
        # Get actual ML weight from config
        from ..query import get_current_ml_weight_info
        weight_info = get_current_ml_weight_info()
        ml_weight_display = weight_info['weight_percentage']
        
        # Format equity data for storage
        equity_dict = equity_data.to_dict('records') if not equity_data.empty else []
        
        # Last updated timestamp
        last_updated = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return (
            equity_dict,
            alpha_display,
            accuracy_display,
            ml_weight_display,
            f"{predictions_count:,}",
            last_updated
        )
        
    except Exception as e:
        logger.error(f"Failed to update ML metrics: {e}")
        return {}, "--", "--", "--", "--", f"Error updating: {datetime.now().strftime('%H:%M:%S')}"


@callback(
    Output('ml-equity-chart', 'figure'),
    [Input('ml-equity-store', 'data')]
)
def update_equity_chart(equity_data):
    """Update the ML equity curve chart."""
    try:
        if not equity_data:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No equity data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="ML-Weighted Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400
            )
            return fig
        
        df = pd.DataFrame(equity_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Create line chart
        fig = go.Figure()
        
        # ML-weighted equity
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['ml_weighted_equity'],
            mode='lines',
            name='ML-Weighted Portfolio',
            line=dict(color='#28a745', width=3)
        ))
        
        # Baseline equity
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['baseline_equity'],
            mode='lines',
            name='Baseline Portfolio',
            line=dict(color='#6c757d', width=2, dash='dash')
        ))
        
        # S&P 500 benchmark
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['sp500_equity'],
            mode='lines',
            name='S&P 500',
            line=dict(color='#ffc107', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title="ML-Weighted vs Baseline Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400,
            margin=dict(t=60, b=40, l=40, r=40)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create equity chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


@callback(
    Output('ml-confusion-matrix', 'figure'),
    [Input('ml-signal-interval', 'n_intervals')]
)
def update_confusion_matrix(n_intervals):
    """Update the ML prediction confusion matrix."""
    try:
        from ..query import get_ml_confusion_matrix
        
        confusion_data = get_ml_confusion_matrix()
        
        if confusion_data.empty:
            # Return empty heatmap
            fig = go.Figure()
            fig.add_annotation(
                text="No prediction data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Prediction Accuracy",
                height=400
            )
            return fig
        
        # Remove 'All' row/column for cleaner display
        if 'All' in confusion_data.index:
            confusion_data = confusion_data.drop('All')
        if 'All' in confusion_data.columns:
            confusion_data = confusion_data.drop('All', axis=1)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confusion_data.values,
            x=confusion_data.columns,
            y=confusion_data.index,
            colorscale='RdYlGn',
            text=confusion_data.values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoveronlyonhover=False,
            showscale=True
        ))
        
        fig.update_layout(
            title="ML Quintile vs Actual Returns",
            xaxis_title="Actual Return Sign",
            yaxis_title="ML Prediction Quintile",
            height=400,
            margin=dict(t=60, b=40, l=60, r=40)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create confusion matrix: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating matrix: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


@callback(
    [Output('ml-performance-badge', 'children'),
     Output('ml-live-metrics-chart', 'figure')],
    [Input('ml-signal-interval', 'n_intervals')]
)
def update_live_metrics(n_intervals):
    """Update live ML validation metrics."""
    try:
        from ..query import get_latest_ml_live_metrics, get_ml_live_metrics_history
        
        # Get latest metrics for badge
        latest_metrics = get_latest_ml_live_metrics()
        
        # Create performance badge
        hit_rate = latest_metrics.get('hit_rate', 0.0)
        auc = latest_metrics.get('auc', 0.0)
        ic = latest_metrics.get('ic', 0.0)
        date = latest_metrics.get('date')
        
        # Determine badge color based on hit rate (green >0.55 as specified)
        if hit_rate > 0.55:
            badge_color = "success"
            badge_text = "🟢 STRONG"
        elif hit_rate > 0.50:
            badge_color = "warning"
            badge_text = "🟡 NEUTRAL"
        else:
            badge_color = "danger" 
            badge_text = "🔴 WEAK"
        
        performance_badge = html.Div([
            html.Span(badge_text, className=f"badge badge-{badge_color} fs-6 me-2"),
            html.Small([
                f"Hit Rate: {hit_rate:.1%} | AUC: {auc:.3f} | IC: {ic:.3f}",
                html.Br(),
                f"Last Updated: {date or 'N/A'}"
            ], className="text-muted")
        ], className="text-center")
        
        # Get historical data for chart
        history_data = get_ml_live_metrics_history(days=30)
        
        if history_data.empty:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No live metrics data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="30-Day Live Metrics Trend",
                height=250,
                margin=dict(t=40, b=20, l=40, r=20)
            )
            return performance_badge, fig
        
        # Create mini line chart
        fig = go.Figure()
        
        # AUC line
        fig.add_trace(go.Scatter(
            x=history_data['date'],
            y=history_data['auc'],
            mode='lines+markers',
            name='AUC',
            line=dict(color='#007bff', width=2),
            marker=dict(size=4)
        ))
        
        # Hit Rate line
        fig.add_trace(go.Scatter(
            x=history_data['date'], 
            y=history_data['hit_rate'],
            mode='lines+markers',
            name='Hit Rate',
            line=dict(color='#28a745', width=2),
            marker=dict(size=4)
        ))
        
        # Add horizontal reference line at 0.55 (green threshold)
        fig.add_hline(y=0.55, line_dash="dash", line_color="green", 
                      annotation_text="Strong Threshold (55%)", 
                      annotation_position="bottom right")
        
        fig.update_layout(
            title="30-Day AUC & Hit Rate Trend",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            height=250,
            margin=dict(t=40, b=60, l=40, r=20),
            yaxis=dict(range=[0, 1])  # Fix scale 0-1 for consistency
        )
        
        return performance_badge, fig
        
    except Exception as e:
        logger.error(f"Failed to update live metrics: {e}")
        error_badge = html.Div([
            html.Span("❌ ERROR", className="badge badge-secondary fs-6"),
            html.Small(f"Failed to load: {str(e)}", className="text-muted d-block")
        ], className="text-center")
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading metrics: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=250)
        
        return error_badge, fig


@callback(
    Output('ml-scores-table-container', 'children'),
    [Input('ml-signal-interval', 'n_intervals')]
)
def update_ml_scores_table(n_intervals):
    """Update the ML scores table."""
    try:
        from ..query import get_ml_scores_today
        
        ml_scores = get_ml_scores_today()
        
        if ml_scores.empty:
            return html.Div([
                html.P("No ML predictions available for today.", 
                       className="text-muted text-center")
            ])
        
        # Limit to top 20 for display
        display_data = ml_scores.head(20).copy()
        
        # Format for display
        display_data['ml_score'] = display_data['ml_score'].apply(lambda x: f"{x:.4f}")
        
        # Create DataTable
        table = dash_table.DataTable(
            data=display_data.to_dict('records'),
            columns=[
                {"name": "Rank", "id": "ml_rank", "type": "numeric"},
                {"name": "Symbol", "id": "symbol", "type": "text"},
                {"name": "ML Score", "id": "ml_score", "type": "text"},
                {"name": "Algorithm", "id": "algorithm", "type": "text"},
                {"name": "Date", "id": "prediction_date", "type": "text"}
            ],
            style_cell={
                'textAlign': 'center',
                'font_family': 'Arial',
                'font_size': '14px',
                'padding': '8px'
            },
            style_header={
                'backgroundColor': '#f8f9fa',
                'fontWeight': 'bold',
                'border': '1px solid #dee2e6'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 0},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'row_index': 1},
                    'backgroundColor': '#e2f3e4',
                    'color': 'black',
                },
                {
                    'if': {'row_index': 2},
                    'backgroundColor': '#f0f9f1',
                    'color': 'black',
                }
            ],
            page_size=10,
            sort_action="native",
            filter_action="native",
            export_format="csv"
        )
        
        return html.Div([
            table,
            html.P(f"Showing top {len(display_data)} of {len(ml_scores)} predictions", 
                   className="text-muted small mt-2")
        ])
        
    except Exception as e:
        logger.error(f"Failed to create ML scores table: {e}")
        return html.Div([
            html.P(f"Error loading ML scores: {str(e)}", 
                   className="text-danger text-center")
        ])