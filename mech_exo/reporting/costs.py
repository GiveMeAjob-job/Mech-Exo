"""
Trading Cost & Slippage Analytics

Analyzes trading costs including commission, slippage, and spread costs.
Calculates per-trade and aggregate metrics for performance evaluation.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sqlite3

from ..datasource.storage import DataStorage
from ..execution.fill_store import FillStore

logger = logging.getLogger(__name__)


class TradingCostAnalyzer:
    """Analyzes trading costs and slippage from fill data"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize cost analyzer
        
        Args:
            db_path: Path to database file (optional)
        """
        self.storage = DataStorage(db_path)
        self.fill_store = FillStore(db_path)
    
    def analyze_costs(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Analyze trading costs for date range
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Dictionary with cost analysis results
        """
        logger.info(f"Analyzing trading costs from {start_date} to {end_date}")
        
        try:
            # Get fills data
            fills_df = self._get_fills_data(start_date, end_date)
            
            if fills_df.empty:
                return {
                    'period': {'start': start_date, 'end': end_date},
                    'summary': {'total_trades': 0, 'total_notional': 0.0},
                    'error': 'No fill data found for period'
                }
            
            # Calculate costs for each fill
            costs_df = self._calculate_fill_costs(fills_df)
            
            # Generate summary statistics
            summary = self._generate_cost_summary(costs_df, start_date, end_date)
            
            # Generate daily summaries
            daily_summary = self._generate_daily_summary(costs_df)
            
            # Generate symbol analysis
            symbol_analysis = self._generate_symbol_analysis(costs_df)
            
            # Generate time-of-day analysis
            tod_analysis = self._generate_time_of_day_analysis(costs_df)
            
            return {
                'period': {'start': start_date, 'end': end_date},
                'summary': summary,
                'daily_summary': daily_summary,
                'symbol_analysis': symbol_analysis,
                'time_of_day_analysis': tod_analysis,
                'detailed_costs': costs_df.to_dict('records') if len(costs_df) <= 1000 else None
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze costs: {e}")
            return {
                'period': {'start': start_date, 'end': end_date},
                'error': str(e)
            }
    
    def _get_fills_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Get fills data for analysis period"""
        try:
            # Query fills from database
            query = """
                SELECT 
                    fill_id, symbol, quantity, price, filled_at,
                    gross_value, total_fees, tag, strategy,
                    order_id
                FROM fills 
                WHERE DATE(filled_at) >= ? AND DATE(filled_at) <= ?
                ORDER BY filled_at
            """
            
            result = self.storage.conn.execute(
                query, [str(start_date), str(end_date)]
            ).fetchall()
            
            if not result:
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = [
                'fill_id', 'symbol', 'quantity', 'price', 'filled_at',
                'gross_value', 'total_fees', 'tag', 'strategy', 'order_id'
            ]
            
            df = pd.DataFrame(result, columns=columns)
            df['filled_at'] = pd.to_datetime(df['filled_at'])
            df['date'] = df['filled_at'].dt.date
            df['hour'] = df['filled_at'].dt.hour
            df['minute'] = df['filled_at'].dt.minute
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get fills data: {e}")
            return pd.DataFrame()
    
    def _calculate_fill_costs(self, fills_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate costs for each fill"""
        logger.info(f"Calculating costs for {len(fills_df)} fills")
        
        # Create copy for cost calculations
        costs_df = fills_df.copy()
        
        # Calculate basic metrics
        costs_df['abs_quantity'] = costs_df['quantity'].abs()
        costs_df['notional_value'] = costs_df['abs_quantity'] * costs_df['price']
        costs_df['commission'] = costs_df['total_fees']
        
        # Get market data for slippage calculation
        costs_df = self._calculate_slippage(costs_df)
        
        # Calculate spread costs
        costs_df = self._calculate_spread_costs(costs_df)
        
        # Calculate total costs
        costs_df['total_cost'] = (
            costs_df['commission'] + 
            costs_df.get('slippage_cost', 0) + 
            costs_df.get('spread_cost', 0)
        )
        
        # Calculate cost as percentage of notional
        costs_df['cost_bps'] = (costs_df['total_cost'] / costs_df['notional_value'] * 10000).fillna(0)
        costs_df['commission_bps'] = (costs_df['commission'] / costs_df['notional_value'] * 10000).fillna(0)
        costs_df['slippage_bps'] = (costs_df.get('slippage_cost', 0) / costs_df['notional_value'] * 10000).fillna(0)
        costs_df['spread_bps'] = (costs_df.get('spread_cost', 0) / costs_df['notional_value'] * 10000).fillna(0)
        
        return costs_df
    
    def _calculate_slippage(self, costs_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate slippage costs by comparing fill price to market mid"""
        logger.info("Calculating slippage costs")
        
        # For now, use a simplified slippage calculation
        # In production, this would fetch actual bid/ask data at fill time
        costs_df['market_mid'] = costs_df['price']  # Fallback to fill price
        costs_df['slippage_price'] = 0.0
        costs_df['slippage_cost'] = 0.0
        
        # Try to get market data from OHLC if available
        try:
            for symbol in costs_df['symbol'].unique():
                symbol_fills = costs_df[costs_df['symbol'] == symbol]
                
                for _, fill in symbol_fills.iterrows():
                    # Try to get market data around fill time
                    market_price = self._get_market_mid_price(
                        symbol, fill['filled_at']
                    )
                    
                    if market_price is not None:
                        costs_df.loc[costs_df['fill_id'] == fill['fill_id'], 'market_mid'] = market_price
                        
                        # Calculate slippage
                        slippage_price = abs(fill['price'] - market_price)
                        slippage_cost = slippage_price * fill['abs_quantity']
                        
                        costs_df.loc[costs_df['fill_id'] == fill['fill_id'], 'slippage_price'] = slippage_price
                        costs_df.loc[costs_df['fill_id'] == fill['fill_id'], 'slippage_cost'] = slippage_cost
                        
        except Exception as e:
            logger.warning(f"Failed to calculate precise slippage: {e}")
            # Use estimated slippage based on typical bid-ask spreads
            costs_df['slippage_cost'] = costs_df['notional_value'] * 0.0002  # 2 bps estimate
        
        return costs_df
    
    def _get_market_mid_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get market mid price at timestamp (simplified)"""
        try:
            # Try to get OHLC data around the timestamp
            query = """
                SELECT close_price, high_price, low_price
                FROM ohlc_data 
                WHERE symbol = ? AND date = DATE(?)
                ORDER BY date DESC
                LIMIT 1
            """
            
            result = self.storage.conn.execute(
                query, [symbol, timestamp.strftime('%Y-%m-%d')]
            ).fetchone()
            
            if result:
                close_price, high_price, low_price = result
                # Use OHLC mid as approximation
                mid_price = (high_price + low_price) / 2
                return mid_price
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def _calculate_spread_costs(self, costs_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate spread costs (simplified)"""
        # For now, use estimated spread costs based on symbol type
        costs_df['spread_cost'] = 0.0
        
        # ETF spreads (typically tighter)
        etf_symbols = ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO']
        etf_mask = costs_df['symbol'].isin(etf_symbols)
        costs_df.loc[etf_mask, 'spread_cost'] = costs_df.loc[etf_mask, 'notional_value'] * 0.0001  # 1 bps
        
        # Individual stocks (wider spreads)
        stock_mask = ~etf_mask
        costs_df.loc[stock_mask, 'spread_cost'] = costs_df.loc[stock_mask, 'notional_value'] * 0.0003  # 3 bps
        
        return costs_df
    
    def _generate_cost_summary(self, costs_df: pd.DataFrame, start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate overall cost summary"""
        if costs_df.empty:
            return {
                'total_trades': 0,
                'total_notional': 0.0,
                'total_costs': 0.0,
                'avg_cost_bps': 0.0
            }
        
        return {
            'total_trades': len(costs_df),
            'total_notional': costs_df['notional_value'].sum(),
            'total_costs': costs_df['total_cost'].sum(),
            'total_commission': costs_df['commission'].sum(),
            'total_slippage': costs_df['slippage_cost'].sum(),
            'total_spread': costs_df['spread_cost'].sum(),
            'avg_cost_bps': costs_df['cost_bps'].mean(),
            'avg_commission_bps': costs_df['commission_bps'].mean(),
            'avg_slippage_bps': costs_df['slippage_bps'].mean(),
            'avg_spread_bps': costs_df['spread_bps'].mean(),
            'median_cost_bps': costs_df['cost_bps'].median(),
            'cost_bps_95th': costs_df['cost_bps'].quantile(0.95),
            'avg_trade_size': costs_df['notional_value'].mean(),
            'largest_trade': costs_df['notional_value'].max(),
            'period_days': (end_date - start_date).days + 1,
            'avg_trades_per_day': len(costs_df) / max(1, (end_date - start_date).days + 1)
        }
    
    def _generate_daily_summary(self, costs_df: pd.DataFrame) -> pd.DataFrame:
        """Generate daily cost summaries"""
        if costs_df.empty:
            return pd.DataFrame()
        
        daily = costs_df.groupby('date').agg({
            'fill_id': 'count',
            'notional_value': ['sum', 'mean'],
            'total_cost': 'sum',
            'commission': 'sum',
            'slippage_cost': 'sum',
            'spread_cost': 'sum',
            'cost_bps': ['mean', 'median'],
            'commission_bps': 'mean',
            'slippage_bps': 'mean',
            'spread_bps': 'mean'
        }).round(4)
        
        # Flatten column names
        daily.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in daily.columns]
        daily = daily.rename(columns={
            'fill_id_count': 'trades',
            'notional_value_sum': 'total_notional',
            'notional_value_mean': 'avg_trade_size',
            'cost_bps_mean': 'avg_cost_bps',
            'cost_bps_median': 'median_cost_bps'
        })
        
        return daily.reset_index()
    
    def _generate_symbol_analysis(self, costs_df: pd.DataFrame) -> pd.DataFrame:
        """Generate per-symbol cost analysis"""
        if costs_df.empty:
            return pd.DataFrame()
        
        symbol_analysis = costs_df.groupby('symbol').agg({
            'fill_id': 'count',
            'notional_value': ['sum', 'mean'],
            'total_cost': 'sum',
            'cost_bps': ['mean', 'median', 'std'],
            'commission_bps': 'mean',
            'slippage_bps': 'mean',
            'spread_bps': 'mean'
        }).round(4)
        
        # Flatten column names
        symbol_analysis.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in symbol_analysis.columns]
        symbol_analysis = symbol_analysis.rename(columns={
            'fill_id_count': 'trades',
            'notional_value_sum': 'total_notional',
            'notional_value_mean': 'avg_trade_size',
            'cost_bps_mean': 'avg_cost_bps',
            'cost_bps_median': 'median_cost_bps',
            'cost_bps_std': 'cost_bps_std'
        })
        
        return symbol_analysis.reset_index().sort_values('total_notional', ascending=False)
    
    def _generate_time_of_day_analysis(self, costs_df: pd.DataFrame) -> pd.DataFrame:
        """Generate time-of-day cost analysis"""
        if costs_df.empty:
            return pd.DataFrame()
        
        tod_analysis = costs_df.groupby('hour').agg({
            'fill_id': 'count',
            'notional_value': 'sum',
            'cost_bps': ['mean', 'median'],
            'slippage_bps': 'mean',
            'spread_bps': 'mean'
        }).round(4)
        
        # Flatten column names
        tod_analysis.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in tod_analysis.columns]
        tod_analysis = tod_analysis.rename(columns={
            'fill_id_count': 'trades',
            'notional_value_sum': 'total_notional',
            'cost_bps_mean': 'avg_cost_bps',
            'cost_bps_median': 'median_cost_bps'
        })
        
        return tod_analysis.reset_index()
    
    def create_trade_costs_table(self, start_date: date, end_date: date) -> bool:
        """Create/update trade_costs summary table"""
        logger.info(f"Creating trade_costs table for {start_date} to {end_date}")
        
        try:
            # Analyze costs
            cost_analysis = self.analyze_costs(start_date, end_date)
            
            if 'error' in cost_analysis:
                logger.error(f"Cost analysis failed: {cost_analysis['error']}")
                return False
            
            # Create table if not exists
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS trade_costs (
                    date DATE PRIMARY KEY,
                    trades INTEGER,
                    total_notional DOUBLE,
                    total_costs DOUBLE,
                    total_commission DOUBLE,
                    total_slippage DOUBLE,
                    total_spread DOUBLE,
                    avg_cost_bps DOUBLE,
                    avg_commission_bps DOUBLE,
                    avg_slippage_bps DOUBLE,
                    avg_spread_bps DOUBLE,
                    median_cost_bps DOUBLE,
                    avg_trade_size DOUBLE,
                    largest_trade DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            
            self.storage.conn.execute(create_table_sql)
            
            # Insert daily summaries
            daily_summary = cost_analysis.get('daily_summary')
            if daily_summary is not None and not daily_summary.empty:
                for _, row in daily_summary.iterrows():
                    upsert_sql = """
                        INSERT OR REPLACE INTO trade_costs (
                            date, trades, total_notional, total_costs,
                            total_commission, total_slippage, total_spread,
                            avg_cost_bps, avg_commission_bps, avg_slippage_bps, avg_spread_bps,
                            median_cost_bps, avg_trade_size, largest_trade
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    # Get largest trade for this date (need to recalculate)
                    largest_trade = self._get_largest_trade_for_date(row['date'])
                    
                    self.storage.conn.execute(upsert_sql, [
                        str(row['date']),
                        row['trades'],
                        row['total_notional'],
                        row['total_cost'],
                        row['commission'],
                        row['slippage_cost'],
                        row['spread_cost'],
                        row['avg_cost_bps'],
                        row['commission_bps'],
                        row['slippage_bps'],
                        row['spread_bps'],
                        row['median_cost_bps'],
                        row['avg_trade_size'],
                        largest_trade
                    ])
                
                self.storage.conn.commit()
                logger.info(f"Updated trade_costs table with {len(daily_summary)} daily records")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to create trade_costs table: {e}")
            return False
    
    def _get_largest_trade_for_date(self, trade_date: date) -> float:
        """Get largest trade notional for a specific date"""
        try:
            query = """
                SELECT MAX(ABS(quantity) * price) as largest_trade
                FROM fills 
                WHERE DATE(filled_at) = ?
            """
            
            result = self.storage.conn.execute(query, [str(trade_date)]).fetchone()
            return result[0] if result and result[0] else 0.0
            
        except Exception as e:
            logger.error(f"Failed to get largest trade for {trade_date}: {e}")
            return 0.0
    
    def export_html_report(self, cost_analysis: Dict[str, Any], output_path: str) -> bool:
        """Export cost analysis as HTML report"""
        try:
            html_content = self._generate_html_report(cost_analysis)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML cost report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export HTML report: {e}")
            return False
    
    def _generate_html_report(self, cost_analysis: Dict[str, Any]) -> str:
        """Generate HTML cost report"""
        period = cost_analysis['period']
        summary = cost_analysis.get('summary', {})
        daily_summary = cost_analysis.get('daily_summary')
        symbol_analysis = cost_analysis.get('symbol_analysis')
        tod_analysis = cost_analysis.get('time_of_day_analysis')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Cost Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .summary-box {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 1.2em; font-weight: bold; color: #007bff; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .warning {{ color: #ff6b35; }}
                .good {{ color: #28a745; }}
            </style>
        </head>
        <body>
            <h1>Trading Cost Analysis Report</h1>
            <p><strong>Period:</strong> {period['start']} to {period['end']}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary-box">
                <h2>Summary</h2>
                <div class="metric">
                    <div>Total Trades</div>
                    <div class="metric-value">{summary.get('total_trades', 0):,}</div>
                </div>
                <div class="metric">
                    <div>Total Notional</div>
                    <div class="metric-value">${summary.get('total_notional', 0):,.0f}</div>
                </div>
                <div class="metric">
                    <div>Total Costs</div>
                    <div class="metric-value">${summary.get('total_costs', 0):,.2f}</div>
                </div>
                <div class="metric">
                    <div>Average Cost</div>
                    <div class="metric-value {'good' if summary.get('avg_cost_bps', 0) < 10 else 'warning'}">{summary.get('avg_cost_bps', 0):.1f} bps</div>
                </div>
                <div class="metric">
                    <div>Commission</div>
                    <div class="metric-value">{summary.get('avg_commission_bps', 0):.1f} bps</div>
                </div>
                <div class="metric">
                    <div>Slippage</div>
                    <div class="metric-value">{summary.get('avg_slippage_bps', 0):.1f} bps</div>
                </div>
                <div class="metric">
                    <div>Spread Cost</div>
                    <div class="metric-value">{summary.get('avg_spread_bps', 0):.1f} bps</div>
                </div>
            </div>
        """
        
        # Add daily summary table
        if daily_summary is not None and not daily_summary.empty:
            html += "<h2>Daily Summary</h2><table><tr>"
            for col in daily_summary.columns:
                html += f"<th>{col.replace('_', ' ').title()}</th>"
            html += "</tr>"
            
            for _, row in daily_summary.iterrows():
                html += "<tr>"
                for col in daily_summary.columns:
                    value = row[col]
                    if isinstance(value, float):
                        if 'bps' in col:
                            html += f"<td>{value:.1f}</td>"
                        elif 'notional' in col or 'cost' in col:
                            html += f"<td>${value:,.0f}</td>"
                        else:
                            html += f"<td>{value:.2f}</td>"
                    else:
                        html += f"<td>{value}</td>"
                html += "</tr>"
            html += "</table>"
        
        # Add symbol analysis table
        if symbol_analysis is not None and not symbol_analysis.empty:
            html += "<h2>Symbol Analysis</h2><table><tr>"
            for col in symbol_analysis.columns:
                html += f"<th>{col.replace('_', ' ').title()}</th>"
            html += "</tr>"
            
            for _, row in symbol_analysis.iterrows():
                html += "<tr>"
                for col in symbol_analysis.columns:
                    value = row[col]
                    if isinstance(value, float):
                        if 'bps' in col:
                            html += f"<td>{value:.1f}</td>"
                        elif 'notional' in col or 'cost' in col:
                            html += f"<td>${value:,.0f}</td>"
                        else:
                            html += f"<td>{value:.2f}</td>"
                    else:
                        html += f"<td>{value}</td>"
                html += "</tr>"
            html += "</table>"
        
        html += """
            <hr>
            <p><small>Generated by Mech-Exo Trading Cost Analyzer</small></p>
        </body>
        </html>
        """
        
        return html
    
    def analyze_commission_accuracy(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Analyze commission estimation accuracy vs actual broker commissions
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Dictionary with commission accuracy analysis
        """
        logger.info(f"Analyzing commission accuracy from {start_date} to {end_date}")
        
        try:
            # Get fills with commission source information
            query = """
            SELECT 
                fill_id,
                symbol,
                quantity,
                fill_price,
                original_commission_usd as estimated_commission,
                commission_usd as actual_commission,
                commission_source,
                fill_time,
                last_reconciled_at,
                (quantity * fill_price) as notional_value
            FROM fills 
            WHERE DATE(fill_time) BETWEEN ? AND ?
            AND commission_source = 'broker'
            AND original_commission_usd IS NOT NULL
            ORDER BY fill_time
            """
            
            result = self.storage.conn.execute(query, [
                start_date.isoformat(), 
                end_date.isoformat()
            ]).fetchall()
            
            if not result:
                logger.warning("No reconciled fills found for commission accuracy analysis")
                return self._empty_commission_analysis()
            
            # Convert to DataFrame
            columns = ['fill_id', 'symbol', 'quantity', 'fill_price', 'estimated_commission',
                      'actual_commission', 'commission_source', 'fill_time', 'last_reconciled_at', 'notional_value']
            df = pd.DataFrame(result, columns=columns)
            
            # Calculate differences and error metrics
            df['commission_diff'] = df['actual_commission'] - df['estimated_commission']
            df['commission_error_pct'] = (df['commission_diff'] / df['estimated_commission'] * 100).fillna(0)
            df['commission_error_bps'] = df['commission_diff'] / df['notional_value'] * 10000
            df['abs_commission_diff'] = df['commission_diff'].abs()
            df['abs_error_pct'] = df['commission_error_pct'].abs()
            
            # Convert timestamps
            df['fill_time'] = pd.to_datetime(df['fill_time'])
            df['last_reconciled_at'] = pd.to_datetime(df['last_reconciled_at'])
            
            # Generate analysis
            analysis = {
                'summary': self._generate_commission_summary(df),
                'daily_errors': self._generate_daily_commission_errors(df),
                'symbol_errors': self._generate_symbol_commission_errors(df),
                'error_distribution': self._generate_commission_error_distribution(df),
                'raw_data': df
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze commission accuracy: {e}")
            return self._empty_commission_analysis()
    
    def _empty_commission_analysis(self) -> Dict[str, Any]:
        """Return empty commission analysis structure"""
        return {
            'summary': {
                'total_reconciled_trades': 0,
                'total_estimated_commission': 0.0,
                'total_actual_commission': 0.0,
                'total_commission_diff': 0.0,
                'avg_error_pct': 0.0,
                'avg_error_bps': 0.0,
                'median_error_pct': 0.0,
                'rmse_pct': 0.0
            },
            'daily_errors': pd.DataFrame(),
            'symbol_errors': pd.DataFrame(),
            'error_distribution': pd.DataFrame(),
            'raw_data': pd.DataFrame()
        }
    
    def _generate_commission_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate commission accuracy summary statistics"""
        if df.empty:
            return self._empty_commission_analysis()['summary']
        
        return {
            'total_reconciled_trades': len(df),
            'total_estimated_commission': df['estimated_commission'].sum(),
            'total_actual_commission': df['actual_commission'].sum(), 
            'total_commission_diff': df['commission_diff'].sum(),
            'avg_error_pct': df['commission_error_pct'].mean(),
            'avg_error_bps': df['commission_error_bps'].mean(),
            'median_error_pct': df['commission_error_pct'].median(),
            'rmse_pct': np.sqrt((df['commission_error_pct'] ** 2).mean()),
            'mae_pct': df['abs_error_pct'].mean(),
            'max_error_pct': df['abs_error_pct'].max(),
            'trades_within_1pct': (df['abs_error_pct'] <= 1.0).sum(),
            'trades_within_5pct': (df['abs_error_pct'] <= 5.0).sum(),
            'trades_within_10pct': (df['abs_error_pct'] <= 10.0).sum(),
            'estimation_accuracy_1pct': (df['abs_error_pct'] <= 1.0).mean() * 100,
            'estimation_accuracy_5pct': (df['abs_error_pct'] <= 5.0).mean() * 100,
            'estimation_accuracy_10pct': (df['abs_error_pct'] <= 10.0).mean() * 100
        }
    
    def _generate_symbol_commission_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate per-symbol commission error analysis"""
        if df.empty:
            return pd.DataFrame()
        
        symbol_errors = df.groupby('symbol').agg({
            'fill_id': 'count',
            'commission_diff': ['sum', 'mean'],
            'commission_error_pct': ['mean', 'std'],
            'abs_error_pct': ['mean', 'max'],
            'commission_error_bps': 'mean',
            'notional_value': 'sum'
        }).round(4)
        
        # Flatten column names  
        symbol_errors.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in symbol_errors.columns]
        symbol_errors = symbol_errors.rename(columns={
            'fill_id_count': 'trades',
            'commission_diff_sum': 'total_diff',
            'commission_diff_mean': 'avg_diff',
            'commission_error_pct_mean': 'avg_error_pct',
            'commission_error_pct_std': 'error_pct_std',
            'abs_error_pct_mean': 'mae_pct',
            'abs_error_pct_max': 'max_error_pct',
            'commission_error_bps_mean': 'avg_error_bps',
            'notional_value_sum': 'total_notional'
        })
        
        return symbol_errors.reset_index().sort_values('total_notional', ascending=False)
    
    def _generate_daily_commission_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate daily commission error analysis"""
        if df.empty:
            return pd.DataFrame()
        
        df['date'] = df['fill_time'].dt.date
        daily_errors = df.groupby('date').agg({
            'fill_id': 'count',
            'commission_diff': ['sum', 'mean'],
            'commission_error_pct': ['mean', 'std'],
            'abs_error_pct': ['mean', 'max'],
            'commission_error_bps': 'mean',
            'notional_value': 'sum'
        }).round(4)
        
        # Flatten column names
        daily_errors.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in daily_errors.columns]
        daily_errors = daily_errors.rename(columns={
            'fill_id_count': 'trades',
            'commission_diff_sum': 'total_diff',
            'commission_diff_mean': 'avg_diff',
            'commission_error_pct_mean': 'avg_error_pct',
            'commission_error_pct_std': 'error_pct_std',
            'abs_error_pct_mean': 'mae_pct',
            'abs_error_pct_max': 'max_error_pct',
            'commission_error_bps_mean': 'avg_error_bps',
            'notional_value_sum': 'total_notional'
        })
        
        return daily_errors.reset_index()
    
    def _generate_commission_error_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate commission error distribution analysis"""
        if df.empty:
            return pd.DataFrame()
        
        # Create error buckets
        error_buckets = [
            ('0-1%', (df['abs_error_pct'] <= 1.0).sum()),
            ('1-5%', ((df['abs_error_pct'] > 1.0) & (df['abs_error_pct'] <= 5.0)).sum()),
            ('5-10%', ((df['abs_error_pct'] > 5.0) & (df['abs_error_pct'] <= 10.0)).sum()),
            ('10-25%', ((df['abs_error_pct'] > 10.0) & (df['abs_error_pct'] <= 25.0)).sum()),
            ('>25%', (df['abs_error_pct'] > 25.0).sum())
        ]
        
        distribution = pd.DataFrame(error_buckets, columns=['error_range', 'trade_count'])
        distribution['percentage'] = (distribution['trade_count'] / len(df) * 100).round(2)
        
        return distribution
    
    def generate_cost_html_report_with_accuracy(self, start_date: date, end_date: date) -> str:
        """
        Generate HTML cost analysis report with commission accuracy section
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            HTML report string with commission accuracy analysis
        """
        logger.info(f"Generating enhanced cost HTML report from {start_date} to {end_date}")
        
        try:
            # Get main cost analysis
            cost_analysis = self.analyze_costs(start_date, end_date)
            
            # Get commission accuracy 
            commission_accuracy = self.analyze_commission_accuracy(start_date, end_date)
            
            # Get existing HTML
            base_html = self._generate_html_report(cost_analysis)
            
            # Add commission accuracy section
            if commission_accuracy['summary']['total_reconciled_trades'] > 0:
                accuracy_html = self._generate_commission_accuracy_html(commission_accuracy)
                # Insert before the closing body tag
                base_html = base_html.replace('</body>', accuracy_html + '</body>')
            
            return base_html
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced cost HTML report: {e}")
            return f"<html><body><h1>Error</h1><p>Failed to generate report: {e}</p></body></html>"
    
    def _generate_commission_accuracy_html(self, commission_accuracy: Dict[str, Any]) -> str:
        """Generate HTML section for commission accuracy"""
        acc_summary = commission_accuracy['summary']
        
        html = f"""
        <div style="margin-top: 30px; border-top: 2px solid #ccc; padding-top: 20px;">
            <h2>Commission Estimation Accuracy</h2>
            <div class="summary-box">
                <div class="metric">
                    <div>Reconciled Trades</div>
                    <div class="metric-value">{acc_summary['total_reconciled_trades']:,}</div>
                </div>
                <div class="metric">
                    <div>Estimated Total</div>
                    <div class="metric-value">${acc_summary['total_estimated_commission']:.2f}</div>
                </div>
                <div class="metric">
                    <div>Actual Total</div>
                    <div class="metric-value">${acc_summary['total_actual_commission']:.2f}</div>
                </div>
                <div class="metric">
                    <div>Total Difference</div>
                    <div class="metric-value {'warning' if abs(acc_summary['total_commission_diff']) > 10 else 'good'}">${acc_summary['total_commission_diff']:.2f}</div>
                </div>
                <div class="metric">
                    <div>Avg Error</div>
                    <div class="metric-value {'warning' if abs(acc_summary['avg_error_pct']) > 5 else 'good'}">{acc_summary['avg_error_pct']:.2f}%</div>
                </div>
                <div class="metric">
                    <div>Accuracy (±5%)</div>
                    <div class="metric-value {'good' if acc_summary['estimation_accuracy_5pct'] > 90 else 'warning'}">{acc_summary['estimation_accuracy_5pct']:.1f}%</div>
                </div>
            </div>
        """
        
        # Add symbol error table if available
        if not commission_accuracy['symbol_errors'].empty:
            html += """
            <h3>Commission Error by Symbol (Top 10)</h3>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Trades</th>
                    <th>Total Diff ($)</th>
                    <th>Avg Error (%)</th>
                    <th>MAE (%)</th>
                    <th>Max Error (%)</th>
                </tr>
            """
            
            for _, row in commission_accuracy['symbol_errors'].head(10).iterrows():
                error_class = 'warning' if abs(row['avg_error_pct']) > 5 else 'good'
                html += f"""
                <tr>
                    <td>{row['symbol']}</td>
                    <td>{row['trades']}</td>
                    <td class="{error_class}">${row['total_diff']:.2f}</td>
                    <td class="{error_class}">{row['avg_error_pct']:.2f}%</td>
                    <td>{row['mae_pct']:.2f}%</td>
                    <td>{row['max_error_pct']:.2f}%</td>
                </tr>
                """
            
            html += "</table>"
        
        html += "</div>"
        return html
    
    def close(self):
        """Close database connections"""
        if self.storage:
            self.storage.close()
        if self.fill_store:
            self.fill_store.close()


def analyze_trading_costs(start_date: str, end_date: str, export_html: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze trading costs
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        export_html: Optional HTML export path
        
    Returns:
        Cost analysis results
    """
    from datetime import datetime
    
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    analyzer = TradingCostAnalyzer()
    
    try:
        # Perform analysis
        results = analyzer.analyze_costs(start, end)
        
        # Update trade_costs table
        analyzer.create_trade_costs_table(start, end)
        
        # Export HTML if requested
        if export_html and 'error' not in results:
            analyzer.export_html_report(results, export_html)
        
        return results
        
    finally:
        analyzer.close()


if __name__ == "__main__":
    # Example usage
    results = analyze_trading_costs('2025-01-01', '2025-01-11', 'trade_costs.html')
    print(f"Analysis complete. Summary: {results.get('summary', {})}")