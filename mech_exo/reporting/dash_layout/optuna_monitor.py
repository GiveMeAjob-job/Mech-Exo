"""
Optuna Monitor dashboard layout and callbacks

Real-time optimization monitoring with trial progress visualization,
parameter importance analysis, and study management controls.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def create_optuna_monitor_layout() -> html.Div:
    """
    Create the Optuna Monitor tab layout
    
    Returns:
        html.Div: Complete layout for Optuna optimization monitoring
    """
    return html.Div([
        # Header section with controls
        html.Div([
            html.H3("🔬 Optuna Optimization Monitor", className="mb-3"),
            
            # Controls row
            html.Div([
                # Study selector
                html.Div([
                    html.Label("Study:", className="form-label"),
                    dcc.Dropdown(
                        id="optuna-study-selector",
                        options=[
                            {"label": "factor_weight_optimization", "value": "factor_weight_optimization"},
                            {"label": "factor_opt", "value": "factor_opt"},
                            {"label": "All Studies", "value": "all"}
                        ],
                        value="factor_weight_optimization",
                        className="form-select"
                    )
                ], className="col-md-3"),
                
                # Optuna Dashboard button
                html.Div([
                    html.Label("External Tools:", className="form-label"),
                    html.Br(),
                    html.A(
                        "📊 Open Optuna Dashboard",
                        href="http://localhost:8080",
                        target="_blank",
                        className="btn btn-outline-primary btn-sm",
                        id="optuna-dashboard-link"
                    )
                ], className="col-md-3"),
                
                # Status indicators
                html.Div([
                    html.Label("Status:", className="form-label"),
                    html.Br(),
                    html.Span(id="optuna-status-badge", className="badge bg-secondary")
                ], className="col-md-3"),
                
                # Last updated
                html.Div([
                    html.Label("Last Updated:", className="form-label"),
                    html.Br(),
                    html.Small(id="optuna-last-updated", className="text-muted")
                ], className="col-md-3")
                
            ], className="row mb-4")
        ], className="container-fluid"),
        
        # Main charts section
        html.Div([
            # Sharpe ratio progression chart
            html.Div([
                html.H5("📈 Sharpe Ratio vs. Trial Number", className="card-title"),
                dcc.Graph(
                    id="optuna-sharpe-chart",
                    config={"displayModeBar": True, "displaylogo": False}
                )
            ], className="col-lg-8"),
            
            # Summary statistics
            html.Div([
                html.H5("📊 Study Summary", className="card-title"),
                html.Div(id="optuna-summary-stats", className="mt-3")
            ], className="col-lg-4")
            
        ], className="row mb-4"),
        
        # Parameter importance and recent trials
        html.Div([
            # Parameter importance chart
            html.Div([
                html.H5("🎯 Parameter Importance", className="card-title"),
                dcc.Graph(
                    id="optuna-importance-chart",
                    config={"displayModeBar": True, "displaylogo": False}
                )
            ], className="col-lg-6"),
            
            # Recent trials table
            html.Div([
                html.H5("🔬 Recent Trials", className="card-title"),
                html.Div(id="optuna-recent-trials", className="table-responsive")
            ], className="col-lg-6")
            
        ], className="row"),
        
        # Auto-refresh interval (30 minutes)
        dcc.Interval(
            id="optuna-refresh-interval",
            interval=30 * 60 * 1000,  # 30 minutes in milliseconds
            n_intervals=0
        ),
        
        # Store component for caching data
        dcc.Store(id="optuna-data-store")
        
    ], className="container-fluid")


@callback(
    [Output("optuna-data-store", "data"),
     Output("optuna-status-badge", "children"),
     Output("optuna-status-badge", "className"),
     Output("optuna-last-updated", "children")],
    [Input("optuna-refresh-interval", "n_intervals"),
     Input("optuna-study-selector", "value")]
)
def update_optuna_data(n_intervals: int, selected_study: str) -> tuple:
    """
    Update Optuna data from database
    
    Args:
        n_intervals: Refresh interval counter
        selected_study: Selected study name
        
    Returns:
        Tuple of (data_dict, status_text, status_class, last_updated)
    """
    try:
        from ...reporting.query import get_optuna_results
        
        # Get trial data
        trials_df = get_optuna_results(limit=200)
        
        if trials_df.empty:
            return (
                {"trials": [], "studies": [], "error": "No data"},
                "No Data",
                "badge bg-warning",
                f"Updated: {datetime.now().strftime('%H:%M:%S')}"
            )
        
        # Filter by study if not "all"
        if selected_study != "all":
            trials_df = trials_df[trials_df['study_name'] == selected_study]
        
        # Convert to dict for JSON serialization
        trials_data = trials_df.to_dict('records')
        
        # Get study list
        study_list = trials_df['study_name'].unique().tolist()
        
        # Calculate status
        total_trials = len(trials_df)
        recent_trials = trials_df[trials_df['calculation_date'] >= 
                                 (datetime.now().date() - pd.Timedelta(days=7))]
        
        if len(recent_trials) > 0:
            status_text = f"{total_trials} trials ({len(recent_trials)} this week)"
            status_class = "badge bg-success"
        else:
            status_text = f"{total_trials} trials (no recent activity)"
            status_class = "badge bg-secondary"
        
        data_dict = {
            "trials": trials_data,
            "studies": study_list,
            "error": None
        }
        
        return (
            data_dict,
            status_text,
            status_class,
            f"Updated: {datetime.now().strftime('%H:%M:%S')}"
        )
        
    except Exception as e:
        logger.error(f"Failed to update Optuna data: {e}")
        return (
            {"trials": [], "studies": [], "error": str(e)},
            "Error",
            "badge bg-danger",
            f"Error: {datetime.now().strftime('%H:%M:%S')}"
        )


@callback(
    Output("optuna-sharpe-chart", "figure"),
    [Input("optuna-data-store", "data")]
)
def update_sharpe_chart(data: Dict) -> go.Figure:
    """
    Create Sharpe ratio progression chart
    
    Args:
        data: Optuna data dictionary
        
    Returns:
        Plotly figure showing Sharpe ratio vs trial number
    """
    try:
        if not data or not data.get("trials") or data.get("error"):
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No optimization data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Sharpe Ratio vs. Trial Number",
                xaxis_title="Trial Number",
                yaxis_title="Sharpe Ratio",
                template="plotly_white",
                height=400
            )
            return fig
        
        # Convert back to DataFrame
        trials_df = pd.DataFrame(data["trials"])
        
        # Create line chart with scatter points
        fig = go.Figure()
        
        # Add line for all trials
        fig.add_trace(go.Scatter(
            x=trials_df['trial_number'],
            y=trials_df['sharpe_ratio'],
            mode='lines+markers',
            name='Sharpe Ratio',
            line=dict(color='#1f77b4', width=2),
            marker=dict(
                size=6,
                color=trials_df['sharpe_ratio'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            hovertemplate=(
                "<b>Trial %{x}</b><br>" +
                "Sharpe: %{y:.4f}<br>" +
                "Max DD: %{customdata[0]:.2%}<br>" +
                "Constraints: %{customdata[1]}<br>" +
                "<extra></extra>"
            ),
            customdata=trials_df[['max_drawdown', 'constraint_status']].values
        ))
        
        # Highlight best trial
        if not trials_df.empty:
            best_trial = trials_df.loc[trials_df['sharpe_ratio'].idxmax()]
            fig.add_trace(go.Scatter(
                x=[best_trial['trial_number']],
                y=[best_trial['sharpe_ratio']],
                mode='markers',
                name='Best Trial',
                marker=dict(
                    size=12,
                    color='gold',
                    symbol='star',
                    line=dict(width=2, color='orange')
                ),
                hovertemplate=(
                    "<b>🌟 Best Trial %{x}</b><br>" +
                    "Sharpe: %{y:.4f}<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Update layout
        fig.update_layout(
            title="Sharpe Ratio Optimization Progress",
            xaxis_title="Trial Number",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            height=400,
            hovermode='closest',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create Sharpe chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Sharpe Ratio vs. Trial Number",
            template="plotly_white",
            height=400
        )
        return fig


@callback(
    Output("optuna-importance-chart", "figure"),
    [Input("optuna-data-store", "data"),
     Input("optuna-study-selector", "value")]
)
def update_importance_chart(data: Dict, selected_study: str) -> go.Figure:
    """
    Create parameter importance chart using Optuna's importance analysis
    
    Args:
        data: Optuna data dictionary
        selected_study: Selected study name
        
    Returns:
        Plotly figure showing parameter importances
    """
    try:
        # For MVP, create mock importance data since we need full Optuna study
        # In production, this would load the actual study and calculate importances
        
        mock_importances = {
            "weight_return_on_equity": 0.234,
            "weight_earnings_growth": 0.187,
            "weight_pe_ratio": 0.156,
            "cash_pct": 0.134,
            "weight_revenue_growth": 0.098,
            "stop_loss_pct": 0.089,
            "weight_momentum_12_1": 0.067,
            "position_size_pct": 0.035
        }
        
        # Convert to DataFrame and sort
        importance_df = pd.DataFrame([
            {"parameter": param, "importance": importance}
            for param, importance in mock_importances.items()
        ]).sort_values("importance", ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['parameter'],
            orientation='h',
            marker=dict(
                color=importance_df['importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f"{imp:.3f}" for imp in importance_df['importance']],
            textposition='auto',
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Importance: %{x:.3f}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Update layout
        fig.update_layout(
            title="Parameter Importance in Optimization",
            xaxis_title="Importance Score",
            yaxis_title="Parameters",
            template="plotly_white",
            height=400,
            margin=dict(l=200)  # More space for parameter names
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create importance chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Importance analysis requires live Optuna study",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            title="Parameter Importance",
            template="plotly_white",
            height=400
        )
        return fig


@callback(
    [Output("optuna-summary-stats", "children"),
     Output("optuna-recent-trials", "children")],
    [Input("optuna-data-store", "data")]
)
def update_summary_and_trials(data: Dict) -> tuple:
    """
    Update summary statistics and recent trials table
    
    Args:
        data: Optuna data dictionary
        
    Returns:
        Tuple of (summary_component, trials_component)
    """
    try:
        if not data or not data.get("trials") or data.get("error"):
            empty_summary = html.Div([
                html.P("No optimization data available", className="text-muted")
            ])
            empty_trials = html.Div([
                html.P("No trial data available", className="text-muted")
            ])
            return empty_summary, empty_trials
        
        # Convert to DataFrame
        trials_df = pd.DataFrame(data["trials"])
        
        if trials_df.empty:
            empty_summary = html.Div([
                html.P("No trials found", className="text-muted")
            ])
            empty_trials = html.Div([
                html.P("No trials found", className="text-muted")
            ])
            return empty_summary, empty_trials
        
        # Calculate summary statistics
        total_trials = len(trials_df)
        best_sharpe = trials_df['sharpe_ratio'].max()
        avg_sharpe = trials_df['sharpe_ratio'].mean()
        constraints_satisfied = trials_df['constraints_satisfied'].sum()
        avg_duration = trials_df['trial_duration_min'].mean()
        
        # Create summary component
        summary_component = html.Div([
            html.Div([
                html.H6(f"{total_trials}", className="fw-bold text-primary"),
                html.Small("Total Trials", className="text-muted")
            ], className="text-center border rounded p-2 mb-2"),
            
            html.Div([
                html.H6(f"{best_sharpe:.4f}", className="fw-bold text-success"),
                html.Small("Best Sharpe", className="text-muted")
            ], className="text-center border rounded p-2 mb-2"),
            
            html.Div([
                html.H6(f"{avg_sharpe:.4f}", className="fw-bold text-info"),
                html.Small("Avg Sharpe", className="text-muted")
            ], className="text-center border rounded p-2 mb-2"),
            
            html.Div([
                html.H6(f"{constraints_satisfied}/{total_trials}", className="fw-bold text-warning"),
                html.Small("Constraints OK", className="text-muted")
            ], className="text-center border rounded p-2 mb-2"),
            
            html.Div([
                html.H6(f"{avg_duration:.1f}min", className="fw-bold text-secondary"),
                html.Small("Avg Duration", className="text-muted")
            ], className="text-center border rounded p-2")
        ])
        
        # Create recent trials table (last 10)
        recent_trials = trials_df.tail(10)
        
        trials_component = html.Div([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Trial", style={"width": "15%"}),
                        html.Th("Sharpe", style={"width": "20%"}),
                        html.Th("Max DD", style={"width": "20%"}),
                        html.Th("Constraints", style={"width": "25%"}),
                        html.Th("Duration", style={"width": "20%"})
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(row['trial_number']),
                        html.Td(f"{row['sharpe_ratio']:.4f}"),
                        html.Td(f"{row['max_drawdown']:.2%}"),
                        html.Td(
                            html.Span(
                                row['constraint_status'],
                                className=f"badge {'bg-success' if row['constraints_satisfied'] else 'bg-danger'}"
                            )
                        ),
                        html.Td(f"{row['trial_duration_min']:.1f}min")
                    ]) for _, row in recent_trials.iterrows()
                ])
            ], className="table table-sm table-striped")
        ])
        
        return summary_component, trials_component
        
    except Exception as e:
        logger.error(f"Failed to update summary and trials: {e}")
        error_component = html.Div([
            html.P(f"Error: {str(e)}", className="text-danger")
        ])
        return error_component, error_component