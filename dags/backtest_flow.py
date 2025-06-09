"""
Prefect flow for nightly backtesting automation
Runs backtest on recent data and stores metrics in DuckDB
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

import pandas as pd

try:
    from prefect import flow, task
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    # Create dummy decorators for when Prefect is not available
    def flow(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func
    
    def task(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

try:
    # Try relative imports first (when run as part of package)
    from ..datasource.storage import DataStorage
    from ..backtest.core import Backtester, create_simple_signals
    from ..utils.alerts import AlertManager
except ImportError:
    # Fall back to absolute imports (when run standalone)
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from mech_exo.datasource.storage import DataStorage
    from mech_exo.backtest.core import Backtester, create_simple_signals
    from mech_exo.utils.alerts import AlertManager

logger = logging.getLogger(__name__)


@task(name="generate_recent_signals", retries=2)
def generate_recent_signals(lookback: str = "730D", symbols: list = None) -> pd.DataFrame:
    """
    Generate trading signals for recent period
    
    Args:
        lookback: Period to look back (e.g., "730D", "24M")
        symbols: List of symbols to generate signals for
        
    Returns:
        DataFrame with boolean trading signals
    """
    try:
        if symbols is None:
            symbols = ["SPY", "QQQ", "IWM", "EFA", "EEM"]  # Default diversified portfolio
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Parse lookback period
        if lookback.endswith('D'):
            days = int(lookback[:-1])
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        elif lookback.endswith('M'):
            months = int(lookback[:-1])
            start_date = (datetime.now() - timedelta(days=months * 30.44)).strftime('%Y-%m-%d')
        else:
            # Default to 2 years
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        logger.info(f"Generating signals for {symbols} from {start_date} to {end_date}")
        
        # For now, use simple buy-and-hold signals with monthly rebalancing
        # In production, this would load actual strategy signals
        signals = create_simple_signals(symbols, start_date, end_date, frequency='monthly')
        
        logger.info(f"Generated {len(signals)} signal rows for {len(signals.columns)} symbols")
        return signals
        
    except Exception as e:
        logger.error(f"Failed to generate signals: {e}")
        raise


@task(name="run_recent_backtest", retries=1)
def run_recent_backtest(signals: pd.DataFrame, lookback: str = "730D") -> Dict:
    """
    Run backtest on recent signals and return metrics
    
    Args:
        signals: Trading signals DataFrame
        lookback: Lookback period for date calculation
        
    Returns:
        Dictionary with backtest metrics
    """
    try:
        if signals.empty:
            raise ValueError("No signals provided for backtesting")
        
        # Calculate date range from signals
        start_date = signals.index.min().strftime('%Y-%m-%d')
        end_date = signals.index.max().strftime('%Y-%m-%d')
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Initialize backtester
        backtester = Backtester(
            start=start_date,
            end=end_date,
            cash=100000,  # $100k standard backtest
            config_path="config/backtest.yml"
        )
        
        # Run backtest
        results = backtester.run(signals)
        
        # Extract key metrics for storage
        metrics = {
            'backtest_date': datetime.now().isoformat(),
            'period_start': start_date,
            'period_end': end_date,
            'lookback_period': lookback,
            'initial_cash': 100000,
            
            # Performance metrics
            'total_return_net': float(results.metrics.get('total_return_net', 0)),
            'cagr_net': float(results.metrics.get('cagr_net', 0)),
            'sharpe_net': float(results.metrics.get('sharpe_net', 0)),
            'volatility': float(results.metrics.get('volatility', 0)),
            'max_drawdown': float(results.metrics.get('max_drawdown', 0)),
            'sortino': float(results.metrics.get('sortino', 0)),
            'calmar_ratio': float(results.metrics.get('calmar_ratio', 0)),
            
            # Trading metrics
            'total_trades': int(results.metrics.get('total_trades', 0)),
            'win_rate': float(results.metrics.get('win_rate', 0)),
            'profit_factor': float(results.metrics.get('profit_factor', 0)),
            'avg_trade_duration': float(results.metrics.get('avg_trade_duration', 0)),
            
            # Cost metrics
            'total_fees': float(results.metrics.get('total_fees', 0)),
            'cost_drag_annual': float(results.metrics.get('cost_drag_annual', 0)),
            
            # Symbols traded
            'symbols_traded': list(signals.columns),
            'num_symbols': len(signals.columns)
        }
        
        logger.info(f"Backtest completed: CAGR={metrics['cagr_net']:.2%}, "
                   f"Sharpe={metrics['sharpe_net']:.2f}, Max DD={metrics['max_drawdown']:.2%}")
        
        return {
            'metrics': metrics,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


@task(name="store_backtest_metrics", retries=3)
def store_backtest_metrics(backtest_data: Dict) -> bool:
    """
    Store backtest metrics in DuckDB
    
    Args:
        backtest_data: Dictionary containing metrics and results
        
    Returns:
        True if successful
    """
    try:
        metrics = backtest_data['metrics']
        
        # Connect to DuckDB
        storage = DataStorage()
        
        # Create table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS backtest_metrics (
            backtest_date TIMESTAMP,
            period_start DATE,
            period_end DATE,
            lookback_period VARCHAR,
            initial_cash DECIMAL,
            total_return_net DECIMAL,
            cagr_net DECIMAL,
            sharpe_net DECIMAL,
            volatility DECIMAL,
            max_drawdown DECIMAL,
            sortino DECIMAL,
            calmar_ratio DECIMAL,
            total_trades INTEGER,
            win_rate DECIMAL,
            profit_factor DECIMAL,
            avg_trade_duration DECIMAL,
            total_fees DECIMAL,
            cost_drag_annual DECIMAL,
            symbols_traded VARCHAR,
            num_symbols INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        storage.conn.execute(create_table_sql)
        
        # Insert metrics
        insert_sql = """
        INSERT INTO backtest_metrics (
            backtest_date, period_start, period_end, lookback_period, initial_cash,
            total_return_net, cagr_net, sharpe_net, volatility, max_drawdown,
            sortino, calmar_ratio, total_trades, win_rate, profit_factor,
            avg_trade_duration, total_fees, cost_drag_annual, symbols_traded, num_symbols
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Convert symbols list to JSON string for storage
        symbols_json = json.dumps(metrics['symbols_traded'])
        
        storage.conn.execute(insert_sql, [
            metrics['backtest_date'],
            metrics['period_start'],
            metrics['period_end'],
            metrics['lookback_period'],
            metrics['initial_cash'],
            metrics['total_return_net'],
            metrics['cagr_net'],
            metrics['sharpe_net'],
            metrics['volatility'],
            metrics['max_drawdown'],
            metrics['sortino'],
            metrics['calmar_ratio'],
            metrics['total_trades'],
            metrics['win_rate'],
            metrics['profit_factor'],
            metrics['avg_trade_duration'],
            metrics['total_fees'],
            metrics['cost_drag_annual'],
            symbols_json,
            metrics['num_symbols']
        ])
        
        storage.close()
        
        logger.info(f"Stored backtest metrics for {metrics['period_start']} to {metrics['period_end']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store backtest metrics: {e}")
        raise


@task(name="generate_tearsheet_artifact")
def generate_tearsheet_artifact(backtest_data: Dict) -> str:
    """
    Generate HTML tearsheet and store as Prefect artifact
    
    Args:
        backtest_data: Dictionary containing metrics and results
        
    Returns:
        Path to generated HTML file
    """
    try:
        results = backtest_data['results']
        metrics = backtest_data['metrics']
        
        # Generate tearsheet
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = f"nightly_tearsheet_{timestamp}.html"
        
        tearsheet_path = results.export_html(
            html_path, 
            strategy_name=f"Nightly Backtest ({metrics['period_start']} to {metrics['period_end']})"
        )
        
        # Read HTML content for Prefect artifact
        with open(tearsheet_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Create Prefect artifact if available
        if PREFECT_AVAILABLE:
            try:
                from prefect.artifacts import create_markdown_artifact
                
                # Create artifact with link to tearsheet
                artifact_content = f"""
# Nightly Backtest Tearsheet

**Period**: {metrics['period_start']} to {metrics['period_end']}  
**Generated**: {metrics['backtest_date']}

## Key Metrics
- **CAGR**: {metrics['cagr_net']:.2%}
- **Sharpe Ratio**: {metrics['sharpe_net']:.2f}
- **Max Drawdown**: {metrics['max_drawdown']:.2%}
- **Total Trades**: {metrics['total_trades']}
- **Win Rate**: {metrics['win_rate']:.1%}

**Tearsheet File**: `{tearsheet_path}`

[View Full Tearsheet]({tearsheet_path})
                """
                
                create_markdown_artifact(
                    key="nightly_tearsheet",
                    markdown=artifact_content,
                    description=f"Nightly backtest tearsheet for {metrics['period_start']} to {metrics['period_end']}"
                )
                
                logger.info(f"Created Prefect artifact for tearsheet")
                
            except Exception as e:
                logger.warning(f"Failed to create Prefect artifact: {e}")
        
        logger.info(f"Generated tearsheet: {tearsheet_path}")
        return tearsheet_path
        
    except Exception as e:
        logger.error(f"Failed to generate tearsheet: {e}")
        raise


@task(name="check_backtest_alerts")
def check_backtest_alerts(backtest_data: Dict) -> bool:
    """
    Check backtest metrics against alert thresholds
    
    Args:
        backtest_data: Dictionary containing metrics and results
        
    Returns:
        True if alerts were sent
    """
    try:
        import os
        
        metrics = backtest_data['metrics']
        
        # Get alert thresholds from environment or use defaults
        min_sharpe = float(os.getenv('ALERT_SHARPE_MIN', '0.5'))
        max_dd_pct = float(os.getenv('ALERT_MAX_DD_PCT', '20.0'))  # Percentage
        
        alerts_sent = False
        
        # Check Sharpe ratio threshold
        if metrics['sharpe_net'] < min_sharpe:
            try:
                alert_manager = AlertManager()
                message = (
                    f"🚨 **Backtest Alert: Low Sharpe Ratio**\n\n"
                    f"**Period**: {metrics['period_start']} to {metrics['period_end']}\n"
                    f"**Sharpe Ratio**: {metrics['sharpe_net']:.2f} (threshold: {min_sharpe})\n"
                    f"**CAGR**: {metrics['cagr_net']:.2%}\n"
                    f"**Max Drawdown**: {metrics['max_drawdown']:.2%}\n"
                    f"**Total Trades**: {metrics['total_trades']}\n\n"
                    f"Strategy performance may need review."
                )
                
                alert_manager.send_alert(
                    message=message,
                    severity="warning",
                    tags=["backtest", "performance", "sharpe"]
                )
                
                alerts_sent = True
                logger.warning(f"Sent Sharpe ratio alert: {metrics['sharpe_net']:.2f} < {min_sharpe}")
                
            except Exception as e:
                logger.error(f"Failed to send Sharpe alert: {e}")
        
        # Check max drawdown threshold
        max_dd_decimal = abs(metrics['max_drawdown'])  # Convert to positive decimal
        if max_dd_decimal * 100 > max_dd_pct:
            try:
                alert_manager = AlertManager()
                message = (
                    f"🚨 **Backtest Alert: High Drawdown**\n\n"
                    f"**Period**: {metrics['period_start']} to {metrics['period_end']}\n"
                    f"**Max Drawdown**: {max_dd_decimal:.2%} (threshold: {max_dd_pct}%)\n"
                    f"**CAGR**: {metrics['cagr_net']:.2%}\n"
                    f"**Sharpe Ratio**: {metrics['sharpe_net']:.2f}\n"
                    f"**Total Trades**: {metrics['total_trades']}\n\n"
                    f"Risk management review recommended."
                )
                
                alert_manager.send_alert(
                    message=message,
                    severity="warning", 
                    tags=["backtest", "risk", "drawdown"]
                )
                
                alerts_sent = True
                logger.warning(f"Sent drawdown alert: {max_dd_decimal:.2%} > {max_dd_pct}%")
                
            except Exception as e:
                logger.error(f"Failed to send drawdown alert: {e}")
        
        if not alerts_sent:
            logger.info(f"Backtest metrics within normal ranges: "
                       f"Sharpe={metrics['sharpe_net']:.2f}, DD={abs(metrics['max_drawdown']):.2%}")
        
        return alerts_sent
        
    except Exception as e:
        logger.error(f"Failed to check backtest alerts: {e}")
        return False


@flow(name="nightly-backtest", description="Automated nightly backtesting with metrics storage")
def nightly_backtest_flow(lookback: str = "730D", symbols: list = None):
    """
    Main Prefect flow for nightly backtesting
    
    Args:
        lookback: Period to backtest (e.g., "730D", "24M")
        symbols: List of symbols to test (default: diversified ETF portfolio)
    """
    try:
        logger.info(f"Starting nightly backtest flow with lookback={lookback}")
        
        # Step 1: Generate signals
        signals = generate_recent_signals(lookback=lookback, symbols=symbols)
        
        # Step 2: Run backtest
        backtest_data = run_recent_backtest(signals, lookback=lookback)
        
        # Step 3: Store metrics in DuckDB
        storage_success = store_backtest_metrics(backtest_data)
        
        # Step 4: Generate tearsheet artifact
        tearsheet_path = generate_tearsheet_artifact(backtest_data)
        
        # Step 5: Check alert thresholds
        alerts_sent = check_backtest_alerts(backtest_data)
        
        # Log completion
        metrics = backtest_data['metrics']
        logger.info(f"Nightly backtest completed successfully:")
        logger.info(f"  CAGR: {metrics['cagr_net']:.2%}")
        logger.info(f"  Sharpe: {metrics['sharpe_net']:.2f}")
        logger.info(f"  Max DD: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Trades: {metrics['total_trades']}")
        logger.info(f"  Tearsheet: {tearsheet_path}")
        logger.info(f"  Alerts sent: {alerts_sent}")
        
        return {
            'success': True,
            'metrics': metrics,
            'tearsheet_path': tearsheet_path,
            'alerts_sent': alerts_sent,
            'storage_success': storage_success
        }
        
    except Exception as e:
        logger.error(f"Nightly backtest flow failed: {e}")
        
        # Send failure alert
        try:
            alert_manager = AlertManager()
            alert_manager.send_alert(
                message=f"🚨 **Nightly Backtest Failed**\n\nError: {str(e)}",
                severity="error",
                tags=["backtest", "failure", "automation"]
            )
        except Exception as alert_error:
            logger.error(f"Failed to send failure alert: {alert_error}")
        
        raise


def create_nightly_backtest_deployment():
    """
    Create Prefect deployment for nightly backtest scheduling
    """
    if not PREFECT_AVAILABLE:
        logger.warning("Prefect not available - cannot create deployment")
        return None
    
    try:
        # Create deployment with cron schedule
        # 03:30 EST = 08:30 UTC (during standard time)
        deployment = Deployment.build_from_flow(
            flow=nightly_backtest_flow,
            name="nightly-backtest-deployment",
            description="Automated nightly backtesting at 03:30 EST",
            schedule=CronSchedule(cron="30 8 * * *"),  # 08:30 UTC = 03:30 EST
            parameters={
                "lookback": "730D",
                "symbols": ["SPY", "QQQ", "IWM", "EFA", "EEM"]
            },
            tags=["backtest", "nightly", "automation"]
        )
        
        # Apply deployment
        deployment_id = deployment.apply()
        
        logger.info(f"Created nightly backtest deployment: {deployment_id}")
        return deployment_id
        
    except Exception as e:
        logger.error(f"Failed to create deployment: {e}")
        raise


# Convenience function for manual execution
def run_manual_backtest(lookback: str = "730D", symbols: list = None):
    """
    Run backtest manually (outside of Prefect scheduling)
    
    Args:
        lookback: Period to backtest
        symbols: List of symbols to test
    """
    try:
        logger.info(f"Running manual backtest with lookback={lookback}")
        
        # Run the flow tasks manually (call functions directly)
        # Check if functions have .fn attribute (Prefect decorated) or call directly
        signals_func = generate_recent_signals.fn if hasattr(generate_recent_signals, 'fn') else generate_recent_signals
        backtest_func = run_recent_backtest.fn if hasattr(run_recent_backtest, 'fn') else run_recent_backtest
        storage_func = store_backtest_metrics.fn if hasattr(store_backtest_metrics, 'fn') else store_backtest_metrics
        tearsheet_func = generate_tearsheet_artifact.fn if hasattr(generate_tearsheet_artifact, 'fn') else generate_tearsheet_artifact
        alerts_func = check_backtest_alerts.fn if hasattr(check_backtest_alerts, 'fn') else check_backtest_alerts
        
        signals = signals_func(lookback=lookback, symbols=symbols)
        backtest_data = backtest_func(signals, lookback=lookback)
        storage_success = storage_func(backtest_data)
        tearsheet_path = tearsheet_func(backtest_data)
        alerts_sent = alerts_func(backtest_data)
        
        logger.info(f"Manual backtest completed successfully")
        return {
            'success': True,
            'metrics': backtest_data['metrics'],
            'tearsheet_path': tearsheet_path,
            'alerts_sent': alerts_sent,
            'storage_success': storage_success
        }
        
    except Exception as e:
        logger.error(f"Manual backtest failed: {e}")
        raise


if __name__ == "__main__":
    # Run manual backtest for testing
    result = run_manual_backtest(lookback="365D", symbols=["SPY", "QQQ"])
    print(f"Manual backtest result: {result}")