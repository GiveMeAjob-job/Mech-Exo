"""
Walk-forward analysis for backtesting robustness validation
Implements rolling window backtests with out-of-sample testing
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np

from .core import Backtester, BacktestResults
from .signal_builder import idea_rank_to_signals

logger = logging.getLogger(__name__)


def make_walk_windows(start: str, end: str, train: str = "36M", test: str = "12M") -> List[Tuple[str, str, str, str]]:
    """
    Generate walk-forward analysis windows
    
    Args:
        start: Overall backtest start date (YYYY-MM-DD)
        end: Overall backtest end date (YYYY-MM-DD)
        train: Training period length (e.g., "36M", "252D", "2Y")
        test: Test period length (e.g., "12M", "63D", "6M")
        
    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    try:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        # Parse period strings to pandas offsets
        train_offset = _parse_period_to_offset(train)
        test_offset = _parse_period_to_offset(test)
        
        windows = []
        current_start = start_date
        
        while current_start + train_offset + test_offset <= end_date:
            # Training window
            train_start = current_start
            train_end = current_start + train_offset
            
            # Test window (immediately following training)
            test_start = train_end
            test_end = test_start + test_offset
            
            windows.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))
            
            # Move forward by test period for next window
            current_start = test_start
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
        
    except Exception as e:
        logger.error(f"Failed to generate walk-forward windows: {e}")
        raise


def _parse_period_to_offset(period: str) -> pd.Timedelta:
    """
    Parse period string to pandas Timedelta, handling months and years
    
    Args:
        period: Period string like "36M", "252D", "2Y"
        
    Returns:
        pandas Timedelta object
    """
    import re
    
    # Extract number and unit
    match = re.match(r'(\d+)([MDYmdy])', period.upper())
    if not match:
        # Fallback to direct parsing for simple cases
        return pd.Timedelta(period)
    
    number, unit = match.groups()
    number = int(number)
    
    if unit in ['M']:
        # Convert months to approximate days (30.44 days per month on average)
        return pd.Timedelta(days=number * 30.44)
    elif unit in ['Y']:
        # Convert years to days (365.25 days per year on average)
        return pd.Timedelta(days=number * 365.25)
    elif unit in ['D']:
        return pd.Timedelta(days=number)
    else:
        # Fallback
        return pd.Timedelta(period)


class WalkForwardAnalyzer:
    """
    Walk-forward backtesting engine for strategy robustness validation
    """
    
    def __init__(self, train_period: str = "36M", test_period: str = "12M", 
                 config_path: str = "config/backtest.yml"):
        """
        Initialize walk-forward analyzer
        
        Args:
            train_period: Training window size (e.g., "36M", "252D")
            test_period: Test window size (e.g., "12M", "63D")
            config_path: Path to backtest configuration
        """
        self.train_period = train_period
        self.test_period = test_period
        self.config_path = config_path
        self.results = []
        self.combined_equity = None
        
    def run_walk_forward(self, start: str, end: str, signals_or_rankings: pd.DataFrame,
                        initial_cash: float = 100000, is_rankings: bool = False,
                        signal_params: dict = None) -> 'WalkForwardResults':
        """
        Run walk-forward analysis
        
        Args:
            start: Overall analysis start date
            end: Overall analysis end date  
            signals_or_rankings: Either boolean signals or ranking scores
            initial_cash: Initial cash amount
            is_rankings: Whether input is rankings (True) or signals (False)
            signal_params: Parameters for converting rankings to signals
            
        Returns:
            WalkForwardResults object
        """
        try:
            # Generate windows
            windows = make_walk_windows(start, end, self.train_period, self.test_period)
            
            if not windows:
                raise ValueError("No valid walk-forward windows generated")
            
            segment_results = []
            segment_equity_curves = []
            
            logger.info(f"Running walk-forward analysis with {len(windows)} windows")
            
            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                logger.info(f"Window {i+1}/{len(windows)}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
                
                try:
                    # Prepare signals for test period
                    if is_rankings:
                        # Convert rankings to signals for test period
                        signal_params = signal_params or {}
                        test_rankings = signals_or_rankings.loc[test_start:test_end]
                        test_signals = idea_rank_to_signals(test_rankings, **signal_params)
                    else:
                        # Use provided signals for test period
                        test_signals = signals_or_rankings.loc[test_start:test_end]
                    
                    if test_signals.empty:
                        logger.warning(f"No signals for window {i+1}, skipping")
                        continue
                    
                    # Run backtest on test period
                    backtester = Backtester(
                        start=test_start,
                        end=test_end,
                        cash=initial_cash,
                        config_path=self.config_path
                    )
                    
                    results = backtester.run(test_signals)
                    
                    # Store segment results
                    segment_metrics = results.metrics.copy()
                    segment_metrics.update({
                        'window_id': i + 1,
                        'train_start': train_start,
                        'train_end': train_end,
                        'test_start': test_start,
                        'test_end': test_end,
                        'train_period': self.train_period,
                        'test_period': self.test_period
                    })
                    
                    segment_results.append(segment_metrics)
                    
                    # Store equity curve for stitching
                    if results.portfolio is not None:
                        equity_curve = results.portfolio.value.copy()
                        equity_curve.name = f'window_{i+1}'
                        segment_equity_curves.append(equity_curve)
                    
                    logger.info(f"Window {i+1} completed: CAGR={segment_metrics.get('cagr_net', 0):.2%}, "
                              f"Sharpe={segment_metrics.get('sharpe_net', 0):.2f}")
                    
                except Exception as e:
                    logger.error(f"Window {i+1} failed: {e}")
                    continue
            
            if not segment_results:
                raise ValueError("No successful walk-forward segments")
            
            # Create combined equity curve
            combined_equity = self._stitch_equity_curves(segment_equity_curves, initial_cash)
            
            # Calculate aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics(segment_results, combined_equity)
            
            # Create results object
            wf_results = WalkForwardResults(
                segment_results=segment_results,
                combined_equity=combined_equity,
                aggregate_metrics=aggregate_metrics,
                windows=windows,
                start_date=start,
                end_date=end
            )
            
            self.results = wf_results
            
            logger.info(f"Walk-forward analysis completed: {len(segment_results)} successful segments")
            return wf_results
            
        except Exception as e:
            logger.error(f"Walk-forward analysis failed: {e}")
            raise
    
    def _stitch_equity_curves(self, equity_curves: List[pd.Series], initial_cash: float) -> pd.Series:
        """Stitch together equity curves from multiple segments"""
        if not equity_curves:
            return pd.Series(dtype=float)
        
        try:
            combined_equity = []
            cumulative_return = 1.0
            
            for i, curve in enumerate(equity_curves):
                if curve.empty:
                    continue
                
                # Normalize curve to start at cumulative_return * initial_cash
                normalized_curve = curve / curve.iloc[0] * cumulative_return * initial_cash
                
                if i == 0:
                    combined_equity.append(normalized_curve)
                else:
                    # Skip first point to avoid duplication
                    combined_equity.append(normalized_curve.iloc[1:])
                
                # Update cumulative return for next segment
                cumulative_return = normalized_curve.iloc[-1] / initial_cash
            
            if combined_equity:
                result = pd.concat(combined_equity)
                result.name = 'walk_forward_equity'
                return result
            else:
                return pd.Series(dtype=float)
                
        except Exception as e:
            logger.error(f"Failed to stitch equity curves: {e}")
            return pd.Series(dtype=float)
    
    def _calculate_aggregate_metrics(self, segment_results: List[Dict], 
                                   combined_equity: pd.Series) -> Dict:
        """Calculate aggregate metrics across all segments"""
        try:
            if not segment_results:
                return {}
            
            # Segment-level statistics
            segment_df = pd.DataFrame(segment_results)
            
            aggregate = {
                'total_segments': len(segment_results),
                'successful_segments': len([r for r in segment_results if r.get('total_trades', 0) > 0]),
                
                # Performance aggregates
                'mean_cagr_net': segment_df['cagr_net'].mean(),
                'median_cagr_net': segment_df['cagr_net'].median(),
                'std_cagr_net': segment_df['cagr_net'].std(),
                'min_cagr_net': segment_df['cagr_net'].min(),
                'max_cagr_net': segment_df['cagr_net'].max(),
                
                'mean_sharpe_net': segment_df['sharpe_net'].mean(),
                'median_sharpe_net': segment_df['sharpe_net'].median(),
                'std_sharpe_net': segment_df['sharpe_net'].std(),
                'min_sharpe_net': segment_df['sharpe_net'].min(),
                'max_sharpe_net': segment_df['sharpe_net'].max(),
                
                'mean_max_drawdown': segment_df['max_drawdown'].mean(),
                'worst_max_drawdown': segment_df['max_drawdown'].min(),  # Most negative
                
                # Trading activity
                'total_trades': segment_df['total_trades'].sum(),
                'mean_trades_per_segment': segment_df['total_trades'].mean(),
                'mean_win_rate': segment_df['win_rate'].mean(),
                
                # Cost analysis
                'total_fees': segment_df['total_fees'].sum(),
                'mean_cost_drag': segment_df['cost_drag_annual'].mean(),
            }
            
            # Combined equity curve metrics
            if not combined_equity.empty and len(combined_equity) > 1:
                returns = combined_equity.pct_change().dropna()
                
                if len(returns) > 0:
                    # Overall performance
                    total_return = (combined_equity.iloc[-1] / combined_equity.iloc[0]) - 1
                    
                    # Annualized metrics
                    trading_days = len(returns)
                    years = trading_days / 252
                    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
                    
                    # Risk metrics
                    volatility = returns.std() * np.sqrt(252)
                    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                    
                    # Drawdown
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min()
                    
                    aggregate.update({
                        'combined_total_return': total_return,
                        'combined_cagr': cagr,
                        'combined_volatility': volatility,
                        'combined_sharpe': sharpe,
                        'combined_max_drawdown': max_drawdown,
                        'combined_trading_days': trading_days,
                        'combined_years': years
                    })
            
            return aggregate
            
        except Exception as e:
            logger.error(f"Failed to calculate aggregate metrics: {e}")
            return {'total_segments': len(segment_results) if segment_results else 0}


class WalkForwardResults:
    """Container for walk-forward analysis results"""
    
    def __init__(self, segment_results: List[Dict], combined_equity: pd.Series,
                 aggregate_metrics: Dict, windows: List[Tuple], 
                 start_date: str, end_date: str):
        self.segment_results = segment_results
        self.combined_equity = combined_equity
        self.aggregate_metrics = aggregate_metrics
        self.windows = windows
        self.start_date = start_date
        self.end_date = end_date
    
    def summary_table(self) -> str:
        """Generate formatted summary table of walk-forward results"""
        try:
            from tabulate import tabulate
        except ImportError:
            # Fallback to simple formatting
            return self._simple_summary_table()
        
        if not self.segment_results:
            return "No walk-forward results available"
        
        # Prepare table data
        table_data = []
        for i, result in enumerate(self.segment_results):
            table_data.append([
                i + 1,
                result.get('test_start', ''),
                result.get('test_end', ''),
                f"{result.get('cagr_net', 0):.2%}",
                f"{result.get('sharpe_net', 0):.2f}",
                f"{result.get('max_drawdown', 0):.2%}",
                result.get('total_trades', 0),
                f"{result.get('win_rate', 0):.1%}"
            ])
        
        headers = ['Window', 'Test Start', 'Test End', 'CAGR', 'Sharpe', 'Max DD', 'Trades', 'Win Rate']
        
        table = tabulate(table_data, headers=headers, tablefmt='github')
        
        # Add aggregate summary
        agg = self.aggregate_metrics
        summary = f"""
{table}

📊 Aggregate Statistics:
   Mean CAGR:        {agg.get('mean_cagr_net', 0):.2%} ± {agg.get('std_cagr_net', 0):.2%}
   Mean Sharpe:      {agg.get('mean_sharpe_net', 0):.2f} ± {agg.get('std_sharpe_net', 0):.2f}
   Combined CAGR:    {agg.get('combined_cagr', 0):.2%}
   Combined Sharpe:  {agg.get('combined_sharpe', 0):.2f}
   Worst Drawdown:   {agg.get('worst_max_drawdown', 0):.2%}
   Total Segments:   {agg.get('total_segments', 0)}
        """
        
        return summary
    
    def _simple_summary_table(self) -> str:
        """Simple fallback table formatting"""
        if not self.segment_results:
            return "No walk-forward results available"
        
        lines = ["Walk-Forward Analysis Results:"]
        lines.append("=" * 80)
        lines.append(f"{'Window':<8} {'Test Period':<20} {'CAGR':<8} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8}")
        lines.append("-" * 80)
        
        for i, result in enumerate(self.segment_results):
            period = f"{result.get('test_start', '')[:7]} to {result.get('test_end', '')[:7]}"
            lines.append(f"{i+1:<8} {period:<20} {result.get('cagr_net', 0):>7.2%} "
                        f"{result.get('sharpe_net', 0):>7.2f} {result.get('max_drawdown', 0):>7.2%} "
                        f"{result.get('total_trades', 0):>7d}")
        
        return "\n".join(lines)
    
    def export_html(self, output_path: str, strategy_name: str = "Walk-Forward Analysis") -> str:
        """Export walk-forward results as HTML report"""
        try:
            from jinja2 import Environment, FileSystemLoader
            
            # Setup Jinja2 environment 
            template_dir = Path(__file__).parent.parent.parent / "templates"
            env = Environment(loader=FileSystemLoader(str(template_dir)))
            
            # Create walk-forward specific template
            self._create_walkforward_template(template_dir)
            
            template = env.get_template("walkforward_report.html.j2")
            
            # Prepare chart data
            chart_data = self._prepare_walkforward_charts()
            
            # Render template
            html_content = template.render(
                strategy_name=strategy_name,
                start_date=self.start_date,
                end_date=self.end_date,
                aggregate_metrics=self.aggregate_metrics,
                segment_results=self.segment_results,
                combined_equity_data=chart_data['combined_equity'],
                segment_performance_data=chart_data['segment_performance'],
                generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Write to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Walk-forward HTML report exported to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to export walk-forward HTML: {e}")
            raise
    
    def _prepare_walkforward_charts(self) -> Dict:
        """Prepare chart data for walk-forward HTML report"""
        try:
            # Combined equity curve
            if not self.combined_equity.empty:
                equity_data = {
                    'dates': [d.strftime('%Y-%m-%d') for d in self.combined_equity.index],
                    'values': self.combined_equity.values.tolist()
                }
            else:
                equity_data = {'dates': [], 'values': []}
            
            # Segment performance bar chart
            if self.segment_results:
                segment_data = {
                    'windows': [f"W{r.get('window_id', i+1)}" for i, r in enumerate(self.segment_results)],
                    'cagr': [r.get('cagr_net', 0) * 100 for r in self.segment_results],
                    'sharpe': [r.get('sharpe_net', 0) for r in self.segment_results],
                    'drawdown': [r.get('max_drawdown', 0) * 100 for r in self.segment_results]
                }
            else:
                segment_data = {'windows': [], 'cagr': [], 'sharpe': [], 'drawdown': []}
            
            return {
                'combined_equity': equity_data,
                'segment_performance': segment_data
            }
            
        except Exception as e:
            logger.warning(f"Failed to prepare walkforward chart data: {e}")
            return {
                'combined_equity': {'dates': [], 'values': []},
                'segment_performance': {'windows': [], 'cagr': [], 'sharpe': [], 'drawdown': []}
            }
    
    def _create_walkforward_template(self, template_dir: Path):
        """Create walk-forward specific HTML template"""
        template_path = template_dir / "walkforward_report.html.j2"
        
        if template_path.exists():
            return  # Template already exists
        
        template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Walk-Forward Analysis - {{ strategy_name }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        .header { text-align: center; margin-bottom: 30px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }
        .chart-container { margin-bottom: 30px; }
        .chart { width: 100%; height: 400px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚶 Walk-Forward Analysis</h1>
            <p>{{ start_date }} to {{ end_date }} | {{ strategy_name }}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Segments</h3>
                <div style="font-size: 1.5em; font-weight: bold;">{{ aggregate_metrics.total_segments }}</div>
            </div>
            <div class="metric-card">
                <h3>Mean CAGR</h3>
                <div style="font-size: 1.5em; font-weight: bold;">{{ "%.2f%%" | format(aggregate_metrics.mean_cagr_net * 100) }}</div>
            </div>
            <div class="metric-card">
                <h3>Mean Sharpe</h3>
                <div style="font-size: 1.5em; font-weight: bold;">{{ "%.2f" | format(aggregate_metrics.mean_sharpe_net) }}</div>
            </div>
            <div class="metric-card">
                <h3>Combined CAGR</h3>
                <div style="font-size: 1.5em; font-weight: bold;">{{ "%.2f%%" | format(aggregate_metrics.combined_cagr * 100) }}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Combined Equity Curve</h2>
            <div id="equity-chart" class="chart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Segment Performance</h2>
            <div id="performance-chart" class="chart"></div>
        </div>
    </div>
    
    <script>
        // Combined equity curve
        var equityData = {{ combined_equity_data | tojson }};
        var equityTrace = {
            x: equityData.dates,
            y: equityData.values,
            type: 'scatter',
            mode: 'lines',
            name: 'Portfolio Value',
            line: { color: '#007bff', width: 2 }
        };
        Plotly.newPlot('equity-chart', [equityTrace], {
            title: '',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Portfolio Value ($)' },
            margin: { l: 60, r: 30, t: 30, b: 60 }
        });
        
        // Segment performance
        var perfData = {{ segment_performance_data | tojson }};
        var cagrTrace = {
            x: perfData.windows,
            y: perfData.cagr,
            type: 'bar',
            name: 'CAGR (%)',
            marker: { color: '#28a745' }
        };
        Plotly.newPlot('performance-chart', [cagrTrace], {
            title: '',
            xaxis: { title: 'Window' },
            yaxis: { title: 'CAGR (%)' },
            margin: { l: 60, r: 30, t: 30, b: 60 }
        });
    </script>
</body>
</html>"""
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)