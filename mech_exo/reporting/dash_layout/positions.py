"""
Positions table tab layout and callbacks for Dash dashboard
Displays current portfolio positions with live refresh every 30 seconds
"""

import logging
from typing import Dict, Any

import pandas as pd
from dash import dcc, html, Input, Output, callback, dash_table

from ..query import get_positions_data

logger = logging.getLogger(__name__)


def create_positions_layout():
    """
    Create the positions table tab layout
    
    Returns:
        Dash layout components for positions table visualization
    """
    
    return html.Div([
        # Header row with title and summary stats
        html.Div([
            html.Div([
                html.H4([
                    html.I(className="fas fa-briefcase me-2"),
                    "Current Positions"
                ], className="mb-0")
            ], className="col"),
            html.Div([
                html.Div(id="positions-summary", className="text-end")
            ], className="col-auto")
        ], className="row mb-3"),
        
        # Portfolio summary cards
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-coins fa-2x text-primary mb-2"),
                        html.H5("Total Positions", className="mb-1"),
                        html.H4(id="total-positions-value", className="text-primary mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-2x text-success mb-2"),
                        html.H5("Market Value", className="mb-1"),
                        html.H4(id="market-value-total", className="text-success mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-percentage fa-2x text-info mb-2"),
                        html.H5("Unrealized P&L", className="mb-1"),
                        html.H4(id="unrealized-pnl-total", className="text-info mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3"),
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-clock fa-2x text-secondary mb-2"),
                        html.H5("Last Updated", className="mb-1"),
                        html.H6(id="last-updated-time", className="text-secondary mb-0")
                    ], className="text-center")
                ], className="metric-card")
            ], className="col-md-3")
        ], className="row mb-4"),
        
        # Main positions table
        html.Div([
            html.Div([
                html.H5([
                    html.I(className="fas fa-table me-2"),
                    "Position Details"
                ], className="mb-3"),
                html.Div(id="positions-table-container"),
                
                # Data loading indicator
                html.Div([
                    dcc.Loading(
                        id="positions-loading",
                        type="default",
                        children=html.Div(id="positions-loading-output")
                    )
                ], className="text-center mt-3")
                
            ], className="metric-card")
        ], className="row")
        
    ])


def register_positions_callbacks():
    """Register callbacks for positions table tab"""
    
    @callback(
        [Output('positions-table-container', 'children'),
         Output('total-positions-value', 'children'),
         Output('market-value-total', 'children'), 
         Output('unrealized-pnl-total', 'children'),
         Output('last-updated-time', 'children'),
         Output('positions-loading-output', 'children')],
        [Input('interval-positions', 'n_intervals')]  # Refresh every 30 seconds
    )
    def update_positions_table(n_intervals):
        """Update positions table and summary metrics"""
        try:
            # Get positions data
            positions_df = get_positions_data()
            
            if positions_df.empty:
                # Return empty state
                empty_table = create_empty_positions_table()
                return (empty_table, "0", "$0.00", "$0.00", "No Data", "")
            
            # Create the data table
            positions_table = create_positions_datatable(positions_df)
            
            # Calculate summary metrics
            total_positions = len(positions_df)
            market_value_total = positions_df['market_value'].sum()
            unrealized_pnl_total = positions_df['unrealized_pnl'].sum()
            
            # Format summary values
            market_value_str = f"${market_value_total:,.2f}"
            pnl_str = f"${unrealized_pnl_total:,.2f}"
            
            # Last updated timestamp
            from datetime import datetime
            last_updated = datetime.now().strftime("%H:%M:%S")
            
            return (positions_table, str(total_positions), market_value_str, 
                   pnl_str, last_updated, "")
            
        except Exception as e:
            logger.error(f"Failed to update positions table: {e}")
            error_table = create_error_positions_table(str(e))
            return (error_table, "Error", "Error", "Error", "Error", "")


def create_positions_datatable(positions_df: pd.DataFrame) -> dash_table.DataTable:
    """
    Create DataTable component for positions
    
    Args:
        positions_df: DataFrame with position data
        
    Returns:
        Dash DataTable component
    """
    
    # Prepare data for table display
    table_data = positions_df.copy()
    
    # Format columns for display
    table_data['avg_price'] = table_data['avg_price'].apply(lambda x: f"${x:.2f}")
    table_data['last_price'] = table_data['last_price'].apply(lambda x: f"${x:.2f}")
    table_data['market_value'] = table_data['market_value'].apply(lambda x: f"${x:,.2f}")
    table_data['unrealized_pnl'] = table_data['unrealized_pnl'].apply(lambda x: f"${x:,.2f}")
    table_data['pnl_pct'] = table_data['pnl_pct'].apply(lambda x: f"{x:.2f}%")
    
    # Define column configuration
    columns = [
        {"name": "Symbol", "id": "symbol", "type": "text"},
        {"name": "Quantity", "id": "quantity", "type": "numeric", "format": {"specifier": ",.0f"}},
        {"name": "Avg Price", "id": "avg_price", "type": "text"},
        {"name": "Last Price", "id": "last_price", "type": "text"},
        {"name": "Market Value", "id": "market_value", "type": "text"},
        {"name": "Unrealized P&L", "id": "unrealized_pnl", "type": "text"},
        {"name": "P&L %", "id": "pnl_pct", "type": "text"}
    ]
    
    return dash_table.DataTable(
        data=table_data.to_dict('records'),
        columns=columns,
        id='positions-table',
        style_table={
            'overflowX': 'auto',
            'minWidth': '100%'
        },
        style_header={
            'backgroundColor': '#007bff',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'border': '1px solid #ddd'
        },
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'border': '1px solid #ddd',
            'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
        },
        style_data_conditional=[
            # Highlight profitable positions in green
            {
                'if': {
                    'filter_query': '{unrealized_pnl} contains "$" && {unrealized_pnl} > "$0"',
                    'column_id': 'unrealized_pnl'
                },
                'backgroundColor': '#d4edda',
                'color': '#155724'
            },
            # Highlight losing positions in red
            {
                'if': {
                    'filter_query': '{unrealized_pnl} contains "-"',
                    'column_id': 'unrealized_pnl'
                },
                'backgroundColor': '#f8d7da',
                'color': '#721c24'
            },
            # Highlight P&L % column based on positive/negative
            {
                'if': {
                    'filter_query': '{pnl_pct} contains "-"',
                    'column_id': 'pnl_pct'
                },
                'backgroundColor': '#f8d7da',
                'color': '#721c24'
            },
            {
                'if': {
                    'filter_query': '{pnl_pct} > "0"',
                    'column_id': 'pnl_pct'
                },
                'backgroundColor': '#d4edda',
                'color': '#155724'
            }
        ],
        sort_action="native",
        sort_mode="single",
        sort_by=[{'column_id': 'market_value', 'direction': 'desc'}],  # Sort by market value desc by default
        page_action="native",
        page_current=0,
        page_size=20,
        tooltip_data=[
            {
                column: {'value': f"Last traded: {row.get('last_trade_date', 'N/A')}", 'type': 'markdown'}
                for column in row.keys()
            } for row in table_data.to_dict('records')
        ],
        tooltip_duration=None
    )


def create_empty_positions_table() -> html.Div:
    """Create empty state when no positions exist"""
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-inbox fa-3x text-muted mb-3"),
            html.H5("No Positions Found", className="text-muted"),
            html.P("No current positions to display. Start trading to see positions here.", 
                   className="text-muted")
        ], className="text-center py-5")
    ])


def create_error_positions_table(error_msg: str) -> html.Div:
    """Create error state when data loading fails"""
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-exclamation-triangle fa-3x text-danger mb-3"),
            html.H5("Error Loading Positions", className="text-danger"),
            html.P(f"Failed to load positions data: {error_msg}", 
                   className="text-muted small")
        ], className="text-center py-5")
    ])