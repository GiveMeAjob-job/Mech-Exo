"""
Alpha-Decay Monitoring Flow

Daily Prefect flow to calculate factor alpha decay metrics and store them
in the database for dashboard display and alerting.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

logger = logging.getLogger(__name__)


@task(name="load_factor_returns", retries=2, retry_delay_seconds=30)
def load_factor_returns(lookback_days: int = 730) -> Dict[str, Any]:
    """
    Load factor scores and forward returns for decay analysis
    
    Args:
        lookback_days: Number of days of data to load for analysis
        
    Returns:
        Dictionary with factor_data, returns_data, and metadata
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.datasource.storage import DataStorage
        from mech_exo.scoring.scorer import IdeaScorer
        
        logger.info(f"Loading factor returns data for {lookback_days} days")
        
        # Initialize data storage
        storage = DataStorage()
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Load OHLC data for returns calculation
        ohlc_df = storage.get_ohlc_data(start_date=start_date, end_date=end_date)
        
        if ohlc_df.empty:
            logger.warning("No OHLC data available for factor returns calculation")
            return {
                'success': False,
                'error': 'No OHLC data available',
                'message': 'Insufficient data for decay analysis'
            }
        
        # Calculate forward returns (next day return)
        if 'close' in ohlc_df.columns:
            ohlc_df['forward_return'] = ohlc_df.groupby('symbol')['close'].pct_change().shift(-1)
            ohlc_df = ohlc_df.dropna()
        else:
            logger.warning("No close price data available for forward returns calculation")
            ohlc_df = pd.DataFrame()  # Empty DataFrame
        
        logger.info(f"Loaded {len(ohlc_df)} OHLC records")
        
        # Load historical factor scores using IdeaScorer
        scorer = IdeaScorer()
        
        # Get factor configuration
        factor_config_path = Path("config/factors.yml")
        if not factor_config_path.exists():
            logger.warning("Factor config not found, using default factors")
            factor_names = ['pe_ratio', 'return_on_equity', 'momentum_12_1', 'rsi_14']
        else:
            try:
                import yaml
                with open(factor_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                factor_names = []
                for category in config.get('factors', {}).values():
                    if isinstance(category, dict):
                        factor_names.extend(category.keys())
                        
            except Exception as e:
                logger.warning(f"Failed to load factor config: {e}")
                factor_names = ['pe_ratio', 'return_on_equity', 'momentum_12_1', 'rsi_14']
        
        # Generate factor scores for the period
        # Note: In a real implementation, this would load pre-calculated scores
        # For now, we'll create synthetic factor data based on fundamentals
        factor_data = {}
        
        # Load fundamental data for factor calculation
        fund_df = storage.get_fundamental_data(start_date=start_date, end_date=end_date)
        
        if not fund_df.empty:
            # Calculate factors from available data
            for symbol_group in fund_df.groupby('symbol'):
                symbol = symbol_group[0]
                symbol_data = symbol_group[1].set_index('date')
                
                # Simple factor calculations
                if 'pe_ratio' in symbol_data.columns:
                    factor_data.setdefault('pe_ratio', []).extend(
                        zip(symbol_data.index, [symbol] * len(symbol_data), 
                            1 / symbol_data['pe_ratio'].fillna(20))  # Inverted P/E
                    )
                
                if 'return_on_equity' in symbol_data.columns:
                    factor_data.setdefault('return_on_equity', []).extend(
                        zip(symbol_data.index, [symbol] * len(symbol_data), 
                            symbol_data['return_on_equity'].fillna(0.1))
                    )
        
        # Add technical factors using OHLC data
        for symbol_group in ohlc_df.groupby('symbol'):
            symbol = symbol_group[0]
            symbol_data = symbol_group[1].set_index('date').sort_index()
            
            if len(symbol_data) >= 20:  # Need minimum data for technical factors
                # 12-1 momentum
                momentum = symbol_data['close'].pct_change(12) - symbol_data['close'].pct_change(1)
                factor_data.setdefault('momentum_12_1', []).extend(
                    zip(momentum.index, [symbol] * len(momentum), momentum.fillna(0))
                )
                
                # RSI-like mean reversion factor
                returns = symbol_data['close'].pct_change()
                rsi_factor = -returns.rolling(14).std()  # Negative volatility
                factor_data.setdefault('rsi_14', []).extend(
                    zip(rsi_factor.index, [symbol] * len(rsi_factor), rsi_factor.fillna(0))
                )
        
        # Convert factor data to DataFrames
        factor_dfs = {}
        returns_data = {}
        
        for factor_name, factor_list in factor_data.items():
            if factor_list:
                factor_df = pd.DataFrame(factor_list, columns=['date', 'symbol', 'factor_value'])
                factor_df['date'] = pd.to_datetime(factor_df['date'])
                
                # Pivot to get time series
                factor_pivot = factor_df.pivot(index='date', columns='symbol', values='factor_value')
                factor_ts = factor_pivot.mean(axis=1)  # Average across symbols
                factor_dfs[factor_name] = factor_ts
                
                # Get corresponding returns
                returns_df = ohlc_df.groupby('date')['forward_return'].mean()
                returns_data[factor_name] = returns_df
        
        if not factor_dfs:
            logger.warning("No factor data generated")
            return {
                'success': False,
                'error': 'No factor data available',
                'message': 'Unable to generate factor scores'
            }
        
        # Close storage connection
        storage.close()
        
        logger.info(f"Successfully loaded {len(factor_dfs)} factors for decay analysis")
        
        return {
            'success': True,
            'factor_data': factor_dfs,
            'returns_data': returns_data,
            'date_range': f"{start_date} to {end_date}",
            'factors_count': len(factor_dfs),
            'message': f"Loaded {len(factor_dfs)} factors with {lookback_days} days of data"
        }
        
    except Exception as e:
        logger.error(f"Failed to load factor returns: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to load factor returns: {e}"
        }


@task(name="calc_decay", retries=2, retry_delay_seconds=30)
def calc_decay(factor_data_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate alpha decay metrics for all factors
    
    Args:
        factor_data_result: Result from load_factor_returns task
        
    Returns:
        Dictionary with decay metrics for all factors
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.research.alpha_decay import AlphaDecayEngine
        
        if not factor_data_result.get('success', False):
            logger.error(f"Cannot calculate decay - factor data load failed: {factor_data_result.get('error')}")
            return {
                'success': False,
                'error': 'Factor data unavailable',
                'message': 'Cannot calculate decay without factor data'
            }
        
        factor_data = factor_data_result['factor_data']
        returns_data = factor_data_result['returns_data']
        
        logger.info(f"Calculating alpha decay for {len(factor_data)} factors")
        
        # Initialize decay engine
        decay_engine = AlphaDecayEngine(window=252, min_periods=60)
        
        # Calculate decay for each factor
        decay_results = []
        
        for factor_name in factor_data.keys():
            if factor_name in returns_data:
                logger.info(f"Calculating decay for factor: {factor_name}")
                
                factor_series = factor_data[factor_name]
                returns_series = returns_data[factor_name]
                
                # Align the series by date
                aligned_df = pd.DataFrame({
                    'factor': factor_series,
                    'returns': returns_series
                }).dropna()
                
                if len(aligned_df) >= 60:  # Minimum data requirement
                    decay_metrics = decay_engine.calc_half_life(
                        aligned_df['factor'], 
                        aligned_df['returns']
                    )
                    
                    # Add metadata
                    decay_metrics.update({
                        'factor_name': factor_name,
                        'calculation_date': datetime.now(),
                        'data_points': len(aligned_df)
                    })
                    
                    decay_results.append(decay_metrics)
                    
                    logger.info(f"Factor {factor_name}: half-life={decay_metrics['half_life']:.1f}d, IC={decay_metrics['latest_ic']:.3f}")
                else:
                    logger.warning(f"Insufficient data for factor {factor_name}: {len(aligned_df)} points")
        
        if not decay_results:
            logger.warning("No decay metrics calculated")
            return {
                'success': False,
                'error': 'No decay metrics calculated',
                'message': 'Insufficient data for any factors'
            }
        
        # Convert to DataFrame for easier handling
        decay_df = pd.DataFrame(decay_results)
        
        logger.info(f"Successfully calculated decay for {len(decay_results)} factors")
        
        return {
            'success': True,
            'decay_metrics': decay_results,
            'decay_df': decay_df,
            'factors_processed': len(decay_results),
            'message': f"Calculated decay metrics for {len(decay_results)} factors"
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate decay: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to calculate decay: {e}"
        }


@task(name="store_decay_metrics", retries=2, retry_delay_seconds=30)
def store_decay_metrics(decay_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store alpha decay metrics in DuckDB factor_decay table
    
    Args:
        decay_result: Result from calc_decay task
        
    Returns:
        Dictionary with storage results
    """
    logger = get_run_logger()
    
    try:
        from mech_exo.datasource.storage import DataStorage
        
        if not decay_result.get('success', False):
            logger.error(f"Cannot store decay metrics - calculation failed: {decay_result.get('error')}")
            return {
                'success': False,
                'error': 'Decay calculation failed',
                'message': 'Cannot store metrics without valid decay data'
            }
        
        decay_metrics = decay_result['decay_metrics']
        
        logger.info(f"Storing {len(decay_metrics)} decay metrics to database")
        
        # Initialize storage
        storage = DataStorage()
        
        # Create factor_decay table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS factor_decay (
            date DATE NOT NULL,
            factor VARCHAR NOT NULL,
            half_life DECIMAL(8,2),
            latest_ic DECIMAL(8,4),
            ic_observations INTEGER,
            ic_mean DECIMAL(8,4),
            ic_std DECIMAL(8,4),
            ic_trend DECIMAL(10,6),
            data_points INTEGER,
            calculation_timestamp TIMESTAMP,
            status VARCHAR,
            PRIMARY KEY (date, factor)
        )
        """
        
        storage.conn.execute(create_table_sql)
        
        # Prepare data for insertion
        insert_data = []
        calculation_date = date.today()
        
        for metrics in decay_metrics:
            row = (
                calculation_date,
                metrics['factor_name'],
                metrics.get('half_life'),
                metrics.get('latest_ic'),
                metrics.get('ic_observations', 0),
                metrics.get('ic_mean'),
                metrics.get('ic_std'),
                metrics.get('ic_trend'),
                metrics.get('data_points', 0),
                metrics['calculation_date'],
                metrics.get('status', 'unknown')
            )
            insert_data.append(row)
        
        # Insert decay metrics
        insert_sql = """
        INSERT OR REPLACE INTO factor_decay 
        (date, factor, half_life, latest_ic, ic_observations, ic_mean, ic_std, 
         ic_trend, data_points, calculation_timestamp, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Insert data using executemany
        storage.conn.executemany(insert_sql, insert_data)
        
        # Verify insertion
        count_sql = "SELECT COUNT(*) as count FROM factor_decay WHERE date = ?"
        result = storage.conn.execute(count_sql, [calculation_date]).fetchdf()
        stored_count = result.iloc[0]['count'] if not result.empty else 0
        
        # Close storage
        storage.close()
        
        logger.info(f"Successfully stored {stored_count} decay metrics for {calculation_date}")
        
        return {
            'success': True,
            'stored_metrics': stored_count,
            'calculation_date': calculation_date,
            'factors_stored': [m['factor_name'] for m in decay_metrics],
            'message': f"Stored {stored_count} decay metrics successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to store decay metrics: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to store decay metrics: {e}"
        }


@task(name="alert_decay", retries=2, retry_delay_seconds=30)
def alert_decay(decay_result: Dict[str, Any], 
                alert_threshold: float = 7.0) -> Dict[str, Any]:
    """
    Send Telegram alerts for factors with rapid alpha decay
    
    Args:
        decay_result: Result from calc_decay task
        alert_threshold: Half-life threshold for alerts (days)
        
    Returns:
        Dictionary with alert results
    """
    logger = get_run_logger()
    
    try:
        import os
        from mech_exo.utils.alerts import TelegramAlerter
        
        if not decay_result.get('success', False):
            logger.info("Skipping decay alerts - no valid decay data")
            return {
                'success': True,
                'alerts_sent': 0,
                'message': 'No alerts needed - no valid decay data'
            }
        
        decay_metrics = decay_result['decay_metrics']
        
        # Find factors with rapid decay
        rapid_decay_factors = []
        for metrics in decay_metrics:
            half_life = metrics.get('half_life')
            if half_life is not None and not np.isnan(half_life) and half_life < alert_threshold:
                rapid_decay_factors.append({
                    'factor': metrics['factor_name'],
                    'half_life': half_life,
                    'latest_ic': metrics.get('latest_ic', 0)
                })
        
        if not rapid_decay_factors:
            logger.info(f"No factors with half-life < {alert_threshold} days")
            return {
                'success': True,
                'alerts_sent': 0,
                'message': f'No rapid decay detected (threshold: {alert_threshold}d)'
            }
        
        logger.info(f"Found {len(rapid_decay_factors)} factors with rapid decay")
        
        # Check Telegram configuration
        telegram_config = {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
        
        if not telegram_config['bot_token'] or not telegram_config['chat_id']:
            logger.warning("Telegram credentials not configured, skipping alerts")
            return {
                'success': False,
                'error': 'Missing Telegram credentials',
                'alerts_sent': 0,
                'rapid_decay_factors': len(rapid_decay_factors)
            }
        
        # Initialize Telegram alerter
        alerter = TelegramAlerter(telegram_config)
        
        # Create batch alert message
        alert_message = "⚠️ *Alpha\\-decay Alert*\n\n"
        
        for factor_info in rapid_decay_factors:
            factor_name = alerter.escape_markdown(factor_info['factor'])
            half_life = factor_info['half_life']
            ic = factor_info['latest_ic']
            
            alert_message += f"📉 *{factor_name}*: half\\-life {half_life:.1f}d \\(<{alert_threshold}\\)\n"
            alert_message += f"    Latest IC: {ic:.3f}\n\n"
        
        alert_message += f"🔍 *Threshold*: {alert_threshold} days\n"
        today_str = date.today().strftime('%Y-%m-%d').replace('-', '\\-')
        alert_message += f"📅 *Date*: {today_str}"
        
        # Send alert
        success = alerter.send_message(alert_message)
        
        if success:
            logger.info(f"Successfully sent decay alert for {len(rapid_decay_factors)} factors")
        else:
            logger.error("Failed to send decay alert")
        
        return {
            'success': success,
            'alerts_sent': 1 if success else 0,
            'rapid_decay_factors': len(rapid_decay_factors),
            'factors_alerted': [f['factor'] for f in rapid_decay_factors],
            'message': f"Alert sent for {len(rapid_decay_factors)} factors with rapid decay"
        }
        
    except Exception as e:
        logger.error(f"Failed to send decay alerts: {e}")
        return {
            'success': False,
            'error': str(e),
            'alerts_sent': 0,
            'message': f"Failed to send decay alerts: {e}"
        }


@flow(name="alpha-decay-flow", 
      task_runner=SequentialTaskRunner(),
      description="Daily alpha decay monitoring and alerting")
def alpha_decay_flow(lookback_days: int = 730, 
                    alert_threshold: float = 7.0) -> Dict[str, Any]:
    """
    Main flow for alpha decay monitoring
    
    Args:
        lookback_days: Days of historical data to analyze
        alert_threshold: Half-life threshold for alerts (days)
        
    Returns:
        Dictionary with flow execution results
    """
    logger = get_run_logger()
    
    logger.info("🔄 Starting alpha decay monitoring flow")
    logger.info(f"Parameters: lookback_days={lookback_days}, alert_threshold={alert_threshold}")
    
    # Step 1: Load factor returns data
    factor_data_result = load_factor_returns(lookback_days)
    
    # Step 2: Calculate decay metrics
    decay_result = calc_decay(factor_data_result)
    
    # Step 3: Store metrics in database
    storage_result = store_decay_metrics(decay_result)
    
    # Step 4: Send alerts for rapid decay
    alert_result = alert_decay(decay_result, alert_threshold)
    
    # Compile flow results
    flow_results = {
        'status': 'completed' if storage_result.get('success') else 'completed_with_issues',
        'factor_loading': factor_data_result,
        'decay_calculation': decay_result,
        'storage': storage_result,
        'alerts': alert_result,
        'execution_time': datetime.now().isoformat()
    }
    
    # Log summary
    if decay_result.get('success'):
        factors_processed = decay_result.get('factors_processed', 0)
        alerts_sent = alert_result.get('alerts_sent', 0)
        logger.info(f"🔄 Alpha decay flow completed: {factors_processed} factors processed, {alerts_sent} alerts sent")
    else:
        logger.warning(f"🔄 Alpha decay flow completed with issues: {decay_result.get('error', 'Unknown error')}")
    
    # Generate heat-map artifact (for Weekend enhancement)
    try:
        from pathlib import Path
        import plotly.graph_objects as go
        
        if decay_result.get('success') and decay_result.get('decay_metrics'):
            logger.info("🎨 Generating decay heat-map artifact...")
            
            # Create heat-map figure
            decay_metrics = decay_result['decay_metrics']
            factors = [m['factor_name'] for m in decay_metrics]
            half_lives = [m.get('half_life', 90) for m in decay_metrics]
            
            # Fill NaN values with 90 for visualization
            import numpy as np
            half_lives = [90 if not hl or np.isnan(hl) else hl for hl in half_lives]
            
            fig = go.Figure(data=go.Heatmap(
                x=factors,
                y=['Half-life (days)'],
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
                    titleside="right"
                )
            ))
            
            fig.update_layout(
                title="Factor Alpha Decay Heat-map",
                xaxis_title="Factors",
                yaxis_title="",
                height=300,
                margin=dict(l=50, r=50, t=50, b=50),
                font=dict(size=12)
            )
            
            # Save as PNG
            artifact_path = Path("decay_heatmap.png")
            try:
                fig.write_image(str(artifact_path))
                logger.info(f"📊 Heat-map artifact saved: {artifact_path}")
                
                # Create Prefect artifact (would be done in actual Prefect flow)
                logger.info("🏗️ Creating Prefect artifact (simulation)")
                # flow.create_artifact("decay_heatmap.png")  # Actual Prefect command
                
            except Exception as img_error:
                logger.warning(f"Failed to save heat-map image: {img_error}")
                logger.info("💡 Install kaleido for PNG export: pip install kaleido")
                
    except Exception as e:
        logger.warning(f"Heat-map artifact generation failed: {e}")
    
    return flow_results


# Manual execution function for testing
def run_manual_decay_analysis(lookback_days: int = 365, alert_threshold: float = 7.0) -> Dict[str, Any]:
    """
    Run alpha decay analysis manually for testing
    
    Args:
        lookback_days: Days of historical data to analyze
        alert_threshold: Half-life threshold for alerts
        
    Returns:
        Flow execution results
    """
    return alpha_decay_flow(lookback_days=lookback_days, alert_threshold=alert_threshold)


if __name__ == "__main__":
    # Test the alpha decay flow
    print("🔄 Testing Alpha Decay Flow...")
    
    # Run with shorter lookback for testing
    result = run_manual_decay_analysis(lookback_days=180, alert_threshold=10.0)
    
    print(f"✅ Flow completed with status: {result.get('status')}")
    
    if result.get('storage', {}).get('success'):
        stored_count = result['storage'].get('stored_metrics', 0)
        print(f"📊 Stored {stored_count} decay metrics")
    
    if result.get('alerts', {}).get('success'):
        alerts_sent = result['alerts'].get('alerts_sent', 0)
        factors_alerted = result['alerts'].get('rapid_decay_factors', 0)
        print(f"🚨 Sent {alerts_sent} alerts for {factors_alerted} factors with rapid decay")