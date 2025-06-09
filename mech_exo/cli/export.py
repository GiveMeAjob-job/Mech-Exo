"""
Data export CLI for Mech-Exo trading system

Provides commands to export trading data to CSV/Parquet formats
for external analysis, auditing, and regulatory compliance.
"""

import logging
import os
import gzip
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import click
import pandas as pd

from ..datasource.storage import DataStorage
from ..execution.fill_store import FillStore

logger = logging.getLogger(__name__)


class DataExporter:
    """
    Data exporter for trading system data
    
    Supports exporting fills, positions, backtest metrics, and drift metrics
    to CSV or Parquet format with optional compression.
    """
    
    SUPPORTED_TABLES = {
        'fills': 'fills',
        'positions': 'positions',  # Will be calculated from fills
        'backtest_metrics': 'backtest_metrics',
        'drift_metrics': 'drift_metrics'
    }
    
    SUPPORTED_FORMATS = ['csv', 'parquet']
    
    def __init__(self):
        """Initialize data exporter"""
        self.storage = DataStorage()
        self.fill_store = FillStore()
        
        # Create exports directory
        self.exports_dir = Path("exports")
        self.exports_dir.mkdir(exist_ok=True)
        
        logger.info(f"DataExporter initialized, exports dir: {self.exports_dir}")
    
    def parse_date_range(self, range_str: str) -> tuple[date, date]:
        """
        Parse date range string into start and end dates
        
        Args:
            range_str: Date range in format "YYYY-MM-DD:YYYY-MM-DD" or keywords
            
        Returns:
            Tuple of (start_date, end_date)
        """
        
        # Handle keyword shortcuts
        today = date.today()
        
        if range_str.lower() == "last7d":
            return today - timedelta(days=7), today
        elif range_str.lower() == "last30d":
            return today - timedelta(days=30), today
        elif range_str.lower() == "last90d":
            return today - timedelta(days=90), today
        elif range_str.lower() == "last365d":
            return today - timedelta(days=365), today
        elif range_str.lower() == "ytd":
            return date(today.year, 1, 1), today
        
        # Parse explicit date range
        if ":" not in range_str:
            raise ValueError("Date range must be in format 'YYYY-MM-DD:YYYY-MM-DD' or use keywords like 'last7d', 'last30d'")
        
        start_str, end_str = range_str.split(":", 1)
        
        try:
            start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d").date()
            end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD format: {e}")
        
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
        
        return start_date, end_date
    
    def get_table_data(self, table: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get data from specified table for date range
        
        Args:
            table: Table name to export
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with table data
        """
        
        if table == 'fills':
            return self._get_fills_data(start_date, end_date)
        elif table == 'positions':
            return self._get_positions_data(start_date, end_date)
        elif table == 'backtest_metrics':
            return self._get_backtest_data(start_date, end_date)
        elif table == 'drift_metrics':
            return self._get_drift_data(start_date, end_date)
        else:
            raise ValueError(f"Unsupported table: {table}. Supported tables: {list(self.SUPPORTED_TABLES.keys())}")
    
    def _get_fills_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Get fills data for date range"""
        try:
            df = self.fill_store.get_fills_df(start_date, end_date)
            
            if not df.empty:
                # Convert timestamps to strings for better CSV compatibility
                if 'filled_at' in df.columns:
                    df['filled_at'] = df['filled_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Add calculated fields
                df['gross_value'] = abs(df['quantity']) * df['price']
                df['net_value'] = df['gross_value'] - df['commission'].fillna(0) - df['fees'].fillna(0)
                
                logger.info(f"Retrieved {len(df)} fills from {start_date} to {end_date}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get fills data: {e}")
            return pd.DataFrame()
    
    def _get_positions_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Get positions data calculated from fills"""
        try:
            # Get fills for the period
            fills_df = self.fill_store.get_fills_df(start_date, end_date)
            
            if fills_df.empty:
                return pd.DataFrame()
            
            # Calculate positions by symbol
            positions = []
            for symbol in fills_df['symbol'].unique():
                symbol_fills = fills_df[fills_df['symbol'] == symbol]
                
                total_qty = symbol_fills['quantity'].sum()
                
                if total_qty != 0:  # Only include non-zero positions
                    # Calculate weighted average price for buys
                    buys = symbol_fills[symbol_fills['quantity'] > 0]
                    if not buys.empty:
                        total_cost = (buys['quantity'] * buys['price']).sum()
                        total_shares = buys['quantity'].sum()
                        avg_price = total_cost / total_shares if total_shares > 0 else 0
                    else:
                        avg_price = 0
                    
                    # Calculate total fees
                    total_fees = symbol_fills['commission'].fillna(0).sum() + symbol_fills['fees'].fillna(0).sum()
                    
                    # Last trade date
                    last_trade = symbol_fills['filled_at'].max()
                    
                    positions.append({
                        'symbol': symbol,
                        'quantity': total_qty,
                        'avg_price': round(avg_price, 4),
                        'market_value': round(total_qty * avg_price, 2),
                        'total_fees': round(total_fees, 2),
                        'last_trade_date': last_trade.strftime('%Y-%m-%d') if pd.notna(last_trade) else None,
                        'trades_count': len(symbol_fills)
                    })
            
            df = pd.DataFrame(positions)
            logger.info(f"Calculated {len(df)} positions from fills data")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get positions data: {e}")
            return pd.DataFrame()
    
    def _get_backtest_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Get backtest metrics data"""
        try:
            query = """
            SELECT *
            FROM backtest_metrics
            WHERE backtest_date >= ? AND backtest_date <= ?
            ORDER BY backtest_date DESC
            """
            
            df = pd.read_sql_query(query, self.storage.conn, params=[start_date, end_date])
            
            if not df.empty:
                # Convert timestamps to strings
                for col in ['backtest_date', 'period_start', 'period_end', 'created_at']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                logger.info(f"Retrieved {len(df)} backtest records from {start_date} to {end_date}")
            
            return df
            
        except Exception as e:
            if "no such table" in str(e).lower() or "does not exist" in str(e).lower():
                logger.warning("Backtest metrics table does not exist")
                return pd.DataFrame()
            else:
                logger.error(f"Failed to get backtest data: {e}")
                return pd.DataFrame()
    
    def _get_drift_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Get drift metrics data"""
        try:
            query = """
            SELECT *
            FROM drift_metrics
            WHERE drift_date >= ? AND drift_date <= ?
            ORDER BY drift_date DESC
            """
            
            df = pd.read_sql_query(query, self.storage.conn, params=[start_date, end_date])
            
            if not df.empty:
                # Convert timestamps to strings
                for col in ['drift_date', 'calculated_at', 'created_at']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                logger.info(f"Retrieved {len(df)} drift records from {start_date} to {end_date}")
            
            return df
            
        except Exception as e:
            if "no such table" in str(e).lower() or "does not exist" in str(e).lower():
                logger.warning("Drift metrics table does not exist")
                return pd.DataFrame()
            else:
                logger.error(f"Failed to get drift data: {e}")
                return pd.DataFrame()
    
    def export_data(self, table: str, start_date: date, end_date: date, 
                   format: str = "parquet", gzip_compress: bool = False) -> Dict[str, Any]:
        """
        Export data to file
        
        Args:
            table: Table name to export
            start_date: Start date for data
            end_date: End date for data
            format: Output format ('csv' or 'parquet')
            gzip_compress: Whether to gzip compress the output
            
        Returns:
            Dictionary with export results
        """
        
        # Validate inputs
        if table not in self.SUPPORTED_TABLES:
            raise ValueError(f"Unsupported table: {table}")
        
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        # Get data
        df = self.get_table_data(table, start_date, end_date)
        
        if df.empty:
            logger.warning(f"No data found for {table} in range {start_date} to {end_date}")
            return {
                'success': False,
                'message': f"No data found for {table} in date range",
                'rows': 0,
                'file_path': None,
                'file_size': 0
            }
        
        # Generate filename
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        base_filename = f"{table}_{start_str}_{end_str}.{format}"
        
        if gzip_compress:
            filename = f"{base_filename}.gz"
        else:
            filename = base_filename
        
        file_path = self.exports_dir / filename
        
        # Export data
        try:
            if format == "csv":
                if gzip_compress:
                    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                        df.to_csv(f, index=False, float_format="%.5g")
                else:
                    df.to_csv(file_path, index=False, float_format="%.5g")
            
            elif format == "parquet":
                compression = "gzip" if gzip_compress else None
                df.to_parquet(file_path, engine="pyarrow", compression=compression, index=False)
            
            file_size = file_path.stat().st_size
            
            logger.info(f"Exported {len(df)} rows to {file_path} ({file_size:,} bytes)")
            
            return {
                'success': True,
                'message': f"Successfully exported {len(df)} rows",
                'rows': len(df),
                'file_path': str(file_path),
                'file_size': file_size,
                'columns': list(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return {
                'success': False,
                'message': f"Export failed: {e}",
                'rows': len(df),
                'file_path': None,
                'file_size': 0
            }
    
    def close(self):
        """Clean up resources"""
        if self.storage:
            self.storage.close()
        if self.fill_store:
            self.fill_store.close()


@click.command()
@click.option('--table', required=True, 
              type=click.Choice(['fills', 'positions', 'backtest_metrics', 'drift_metrics']),
              help='Table to export')
@click.option('--range', 'date_range', required=True,
              help='Date range: YYYY-MM-DD:YYYY-MM-DD or keywords (last7d, last30d, etc.)')
@click.option('--fmt', 'format', default='parquet',
              type=click.Choice(['csv', 'parquet']),
              help='Output format (default: parquet)')
@click.option('--gzip', is_flag=True, default=False,
              help='Compress output with gzip')
@click.option('--output-dir', default='exports',
              help='Output directory (default: exports)')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Verbose output')
def export_command(table: str, date_range: str, format: str, gzip: bool, 
                  output_dir: str, verbose: bool):
    """
    Export trading data to CSV or Parquet format
    
    Examples:
        exo export --table fills --range last7d --fmt csv
        exo export --table positions --range 2024-01-01:2024-06-01 --fmt parquet --gzip
        exo export --table drift_metrics --range last30d --fmt csv
    """
    
    # Set up logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        exporter = DataExporter()
        
        # Override exports directory if specified
        if output_dir != 'exports':
            exporter.exports_dir = Path(output_dir)
            exporter.exports_dir.mkdir(exist_ok=True)
        
        # Parse date range
        start_date, end_date = exporter.parse_date_range(date_range)
        
        click.echo(f"🔍 Exporting {table} data from {start_date} to {end_date}")
        click.echo(f"📁 Output directory: {exporter.exports_dir}")
        click.echo(f"📄 Format: {format}" + (" (gzipped)" if gzip else ""))
        
        # Export data
        result = exporter.export_data(table, start_date, end_date, format, gzip)
        
        if result['success']:
            click.echo(f"✅ Export completed successfully!")
            click.echo(f"   • Rows exported: {result['rows']:,}")
            click.echo(f"   • File path: {result['file_path']}")
            click.echo(f"   • File size: {result['file_size']:,} bytes")
            if 'columns' in result:
                click.echo(f"   • Columns: {len(result['columns'])}")
                if verbose:
                    click.echo(f"   • Column names: {', '.join(result['columns'])}")
        else:
            click.echo(f"❌ Export failed: {result['message']}")
            if result['rows'] > 0:
                click.echo(f"   • Found {result['rows']} rows but export failed")
            exit(1)
        
    except Exception as e:
        click.echo(f"❌ Export command failed: {e}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())
        exit(1)
    
    finally:
        try:
            exporter.close()
        except:
            pass


if __name__ == "__main__":
    export_command()