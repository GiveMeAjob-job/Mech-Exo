"""
Core backtesting engine using vectorbt
Provides historical strategy testing with fee-adjusted metrics
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import numpy as np
from pathlib import Path
import json

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
    VBT_Portfolio = vbt.Portfolio
except ImportError:
    VBT_AVAILABLE = False
    vbt = None
    VBT_Portfolio = type(None)  # Placeholder for type hints

from ..datasource.storage import DataStorage
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


class Backtester:
    """
    Historical backtesting engine with vectorbt integration
    
    Supports multiple strategies, realistic fees/slippage, and comprehensive metrics
    """
    
    def __init__(self, start: str, end: str, cash: float = None, 
                 commission: float = None, slippage: float = None,
                 config_path: str = "config/backtest.yml"):
        """
        Initialize backtester
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD) 
            cash: Initial cash amount (loads from config if None)
            commission: Commission per share (loads from config if None)
            slippage: Slippage percentage (loads from config if None)
            config_path: Path to backtest configuration file
        """
        if not VBT_AVAILABLE:
            raise ImportError("vectorbt is required for backtesting. Install with: pip install vectorbt")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        self.start_date = start
        self.end_date = end
        self.initial_cash = cash or self.config.get('initial_cash', 100_000)
        self.commission = commission or self.config.get('commission_per_share', 0.005)
        self.slippage = slippage or self.config.get('slippage_pct', 0.001)
        
        # Additional cost parameters from config
        self.spread_cost = self.config.get('spread_cost_pct', 0.0005)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        
        # Load price data
        self.prices = self._load_prices()
        
        # Storage for results
        self.portfolio = None
        self.results = {}
        
        logger.info(f"Backtester initialized: {start} to {end}, "
                   f"cash=${self.initial_cash:,.0f}, commission=${self.commission:.3f}, "
                   f"slippage={self.slippage:.3%}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load backtest configuration from YAML file"""
        try:
            import yaml
            from pathlib import Path
            
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded backtest config from {config_path}")
                return config
            else:
                logger.warning(f"Config file {config_path} not found, using defaults")
                return {}
        except ImportError:
            logger.warning("PyYAML not available, using default config")
            return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_prices(self) -> pd.DataFrame:
        """Load historical price data from DuckDB"""
        try:
            storage = DataStorage()
            
            # Query OHLC data for backtest period
            query = """
            SELECT 
                date,
                symbol,
                close as price,
                volume,
                returns_1d,
                volatility_20d
            FROM ohlc_data 
            WHERE date >= ? AND date <= ?
            ORDER BY date, symbol
            """
            
            df = pd.read_sql_query(
                query, 
                storage.conn, 
                params=[self.start_date, self.end_date]
            )
            storage.close()
            
            if df.empty:
                logger.warning(f"No price data found for period {self.start_date} to {self.end_date}")
                return pd.DataFrame()
            
            # Pivot to get symbols as columns
            prices = df.pivot(index='date', columns='symbol', values='price')
            prices.index = pd.to_datetime(prices.index)
            
            # Forward fill missing values, then drop rows with all NaN
            prices = prices.fillna(method='ffill').dropna(how='all')
            
            logger.info(f"Loaded price data: {len(prices)} days, {len(prices.columns)} symbols")
            return prices
            
        except Exception as e:
            logger.error(f"Failed to load price data: {e}")
            return pd.DataFrame()
    
    def run(self, signals: pd.DataFrame, **kwargs) -> 'BacktestResults':
        """
        Run backtest with trading signals
        
        Args:
            signals: DataFrame with boolean signals (index=date, columns=symbols)
            **kwargs: Additional vectorbt portfolio parameters
            
        Returns:
            BacktestResults object with metrics and portfolio
        """
        try:
            if self.prices.empty:
                raise ValueError("No price data available for backtesting")
            
            if signals.empty:
                raise ValueError("No signals provided for backtesting")
            
            # Align signals with prices
            aligned_signals, aligned_prices = signals.align(self.prices, join='inner')
            
            if aligned_signals.empty or aligned_prices.empty:
                raise ValueError("No overlapping data between signals and prices")
            
            # Create vectorbt portfolio with enhanced fee structure
            total_fee_pct = self.commission + self.slippage + self.spread_cost
            
            portfolio = vbt.Portfolio.from_signals(
                close=aligned_prices,
                entries=aligned_signals,
                exits=~aligned_signals,  # Exit when signal turns False
                init_cash=self.initial_cash,
                fees=total_fee_pct,  # Combined transaction costs
                slippage=self.slippage,
                freq='D',
                **kwargs
            )
            
            self.portfolio = portfolio
            
            # Calculate comprehensive metrics
            metrics = self._calculate_metrics(portfolio)
            
            # Store results
            results = BacktestResults(
                portfolio=portfolio,
                metrics=metrics,
                prices=aligned_prices,
                signals=aligned_signals,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            self.results = results
            
            logger.info(f"Backtest completed: CAGR={metrics['cagr']:.2%}, "
                       f"Sharpe={metrics['sharpe']:.2f}, Max DD={metrics['max_drawdown']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _calculate_metrics(self, portfolio: VBT_Portfolio) -> Dict[str, float]:
        """Calculate comprehensive backtest metrics"""
        try:
            # Basic metrics (net of fees)
            total_return = portfolio.total_return()
            cagr_net = portfolio.annualized_return()
            sharpe_net = portfolio.sharpe_ratio(risk_free=self.risk_free_rate)
            sortino = portfolio.sortino_ratio()
            max_drawdown = portfolio.max_drawdown()
            
            # Trading metrics
            win_rate = portfolio.trades.win_rate()
            profit_factor = portfolio.trades.profit_factor()
            avg_trade_duration = portfolio.trades.duration.mean()
            total_trades = portfolio.trades.count()
            
            # Fee analysis
            total_fees = portfolio.fees.sum()
            avg_fee_per_trade = total_fees / total_trades if total_trades > 0 else 0
            fee_pct_of_nav = total_fees / self.initial_cash
            
            # Gross metrics (before fees) 
            try:
                # Estimate gross returns by adding back fees
                portfolio_value = portfolio.value
                returns_net = portfolio.returns
                fee_impact = portfolio.fees / portfolio_value.shift(1).fillna(self.initial_cash)
                returns_gross = returns_net + fee_impact
                
                # Calculate gross CAGR
                cumulative_gross = (1 + returns_gross).cumprod()
                total_gross_return = cumulative_gross.iloc[-1] - 1
                
                trading_days = len(returns_gross)
                years = trading_days / 252
                cagr_gross = (1 + total_gross_return) ** (1/years) - 1 if years > 0 else 0
                
                # Gross Sharpe
                excess_returns_gross = returns_gross - self.risk_free_rate/252
                sharpe_gross = excess_returns_gross.mean() / excess_returns_gross.std() * np.sqrt(252) if excess_returns_gross.std() > 0 else 0
                
            except Exception as e:
                logger.warning(f"Could not calculate gross metrics: {e}")
                cagr_gross = cagr_net
                sharpe_gross = sharpe_net
                total_gross_return = total_return
            
            # Cost impact analysis
            cost_drag = cagr_gross - cagr_net  # Annual cost drag
            
            # Additional risk metrics
            volatility = portfolio.returns.std() * np.sqrt(252)
            calmar_ratio = abs(cagr_net / max_drawdown) if max_drawdown != 0 else 0
            
            metrics = {
                # Performance (Net)
                'total_return_net': float(total_return) if not pd.isna(total_return) else 0.0,
                'cagr_net': float(cagr_net) if not pd.isna(cagr_net) else 0.0,
                'sharpe_net': float(sharpe_net) if not pd.isna(sharpe_net) else 0.0,
                
                # Performance (Gross)
                'total_return_gross': float(total_gross_return) if not pd.isna(total_gross_return) else 0.0,
                'cagr_gross': float(cagr_gross) if not pd.isna(cagr_gross) else 0.0,
                'sharpe_gross': float(sharpe_gross) if not pd.isna(sharpe_gross) else 0.0,
                
                # Risk Metrics
                'volatility': float(volatility) if not pd.isna(volatility) else 0.0,
                'sortino': float(sortino) if not pd.isna(sortino) else 0.0,
                'max_drawdown': float(max_drawdown) if not pd.isna(max_drawdown) else 0.0,
                'calmar_ratio': float(calmar_ratio) if not pd.isna(calmar_ratio) else 0.0,
                
                # Trading Activity
                'total_trades': int(total_trades) if not pd.isna(total_trades) else 0,
                'win_rate': float(win_rate) if not pd.isna(win_rate) else 0.0,
                'profit_factor': float(profit_factor) if not pd.isna(profit_factor) else 0.0,
                'avg_trade_duration': float(avg_trade_duration) if not pd.isna(avg_trade_duration) else 0.0,
                
                # Cost Analysis
                'total_fees': float(total_fees) if not pd.isna(total_fees) else 0.0,
                'avg_fee_per_trade': float(avg_fee_per_trade) if not pd.isna(avg_fee_per_trade) else 0.0,
                'fee_pct_of_nav': float(fee_pct_of_nav) if not pd.isna(fee_pct_of_nav) else 0.0,
                'cost_drag_annual': float(cost_drag) if not pd.isna(cost_drag) else 0.0,
                
                # Meta
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_cash': self.initial_cash,
                'risk_free_rate': self.risk_free_rate
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {
                'total_return_net': 0.0, 'cagr_net': 0.0, 'cagr_gross': 0.0,
                'sharpe_net': 0.0, 'sharpe_gross': 0.0, 'volatility': 0.0,
                'sortino': 0.0, 'max_drawdown': 0.0, 'calmar_ratio': 0.0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'total_trades': 0,
                'avg_trade_duration': 0.0, 'total_fees': 0.0, 'avg_fee_per_trade': 0.0,
                'fee_pct_of_nav': 0.0, 'cost_drag_annual': 0.0,
                'start_date': self.start_date, 'end_date': self.end_date,
                'initial_cash': self.initial_cash, 'risk_free_rate': self.risk_free_rate
            }
    
    def export_html(self, output_path: str, strategy_name: str = "Backtest Strategy") -> str:
        """
        Export backtest results as interactive HTML tear-sheet
        
        Args:
            output_path: Path to save HTML file
            strategy_name: Name of the strategy for display
            
        Returns:
            Path to generated HTML file
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        try:
            from jinja2 import Environment, FileSystemLoader
            import plotly.graph_objects as go
            import plotly.express as px
            
            # Setup Jinja2 environment
            template_dir = Path(__file__).parent.parent.parent / "templates"
            env = Environment(loader=FileSystemLoader(str(template_dir)))
            template = env.get_template("tear_sheet.html.j2")
            
            # Prepare chart data
            chart_data = self._prepare_chart_data()
            
            # Render template
            html_content = template.render(
                strategy_name=strategy_name,
                start_date=self.start_date,
                end_date=self.end_date,
                metrics=self.results.metrics,
                equity_curve_data=chart_data['equity_curve'],
                monthly_returns_data=chart_data['monthly_returns'],
                annual_returns_data=chart_data['annual_returns'],
                generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Write to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML tear-sheet exported to {output_file}")
            return str(output_file)
            
        except ImportError as e:
            logger.error(f"Missing dependencies for HTML export: {e}")
            raise ImportError("jinja2 and plotly are required for HTML export")
        except Exception as e:
            logger.error(f"Failed to export HTML: {e}")
            raise
    
    def _prepare_chart_data(self) -> Dict:
        """Prepare data for Plotly charts in HTML template"""
        try:
            portfolio = self.results.portfolio
            
            # Equity curve data
            portfolio_value = portfolio.value
            equity_curve = {
                'dates': [d.strftime('%Y-%m-%d') for d in portfolio_value.index],
                'values': portfolio_value.values.tolist()
            }
            
            # Monthly returns data for heatmap
            returns = portfolio.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            returns_pct = (returns * 100).round(2)
            
            # Create monthly heatmap data
            monthly_data = []
            years = []
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            year_range = range(returns.index.year.min(), returns.index.year.max() + 1)
            
            for year in year_range:
                year_data = []
                for month in range(1, 13):
                    try:
                        value = returns_pct[f'{year}-{month:02d}']
                        year_data.append(float(value))
                    except (KeyError, IndexError):
                        year_data.append(None)
                monthly_data.append(year_data)
                years.append(str(year))
            
            monthly_returns = {
                'returns': monthly_data,
                'years': years,
                'months': months
            }
            
            # Annual returns data
            annual_returns_series = portfolio.returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
            annual_returns_pct = (annual_returns_series * 100).round(2)
            
            annual_returns = {
                'years': [str(d.year) for d in annual_returns_pct.index],
                'returns': annual_returns_pct.values.tolist()
            }
            
            return {
                'equity_curve': equity_curve,
                'monthly_returns': monthly_returns,
                'annual_returns': annual_returns
            }
            
        except Exception as e:
            logger.warning(f"Failed to prepare chart data: {e}")
            # Return empty data structure as fallback
            return {
                'equity_curve': {'dates': [], 'values': []},
                'monthly_returns': {'returns': [], 'years': [], 'months': []},
                'annual_returns': {'years': [], 'returns': []}
            }


class BacktestResults:
    """Container for backtest results and metrics"""
    
    def __init__(self, portfolio: VBT_Portfolio, metrics: Dict[str, float],
                 prices: pd.DataFrame, signals: pd.DataFrame,
                 start_date: str, end_date: str):
        self.portfolio = portfolio
        self.metrics = metrics
        self.prices = prices
        self.signals = signals
        self.start_date = start_date
        self.end_date = end_date
    
    def summary(self) -> str:
        """Get formatted summary of backtest results"""
        m = self.metrics
        
        summary = f"""
╭─ Backtest Summary ({m['start_date']} to {m['end_date']}) ─╮
│                                                          │
│  💰 Performance Metrics (Net):                          │
│     Total Return:     {m.get('total_return_net', 0):>8.2%}                      │
│     CAGR:             {m.get('cagr_net', 0):>8.2%}                      │
│     Sharpe Ratio:     {m.get('sharpe_net', 0):>8.2f}                      │
│     Volatility:       {m.get('volatility', 0):>8.2%}                      │
│                                                          │
│  💎 Performance Metrics (Gross):                        │
│     Total Return:     {m.get('total_return_gross', 0):>8.2%}                      │
│     CAGR:             {m.get('cagr_gross', 0):>8.2%}                      │
│     Sharpe Ratio:     {m.get('sharpe_gross', 0):>8.2f}                      │
│                                                          │
│  📊 Risk Metrics:                                        │
│     Max Drawdown:     {m.get('max_drawdown', 0):>8.2%}                      │
│     Sortino Ratio:    {m.get('sortino', 0):>8.2f}                      │
│     Calmar Ratio:     {m.get('calmar_ratio', 0):>8.2f}                      │
│                                                          │
│  🔄 Trading Activity:                                    │
│     Total Trades:     {m.get('total_trades', 0):>8d}                      │
│     Win Rate:         {m.get('win_rate', 0):>8.2%}                      │
│     Profit Factor:    {m.get('profit_factor', 0):>8.2f}                      │
│     Avg Duration:     {m.get('avg_trade_duration', 0):>8.1f} days                │
│                                                          │
│  💸 Cost Analysis:                                       │
│     Total Fees:       ${m.get('total_fees', 0):>8,.0f}                     │
│     Avg Fee/Trade:    ${m.get('avg_fee_per_trade', 0):>8,.2f}                   │
│     Fee % of NAV:     {m.get('fee_pct_of_nav', 0):>8.2%}                      │
│     Cost Drag:        {m.get('cost_drag_annual', 0):>8.2%} p.a.                │
│                                                          │
╰─────────────────────────────────────────────────────────╯
        """
        return summary
    
    def to_dict(self) -> Dict:
        """Export results as dictionary"""
        return {
            'metrics': self.metrics,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'prices_shape': self.prices.shape,
            'signals_shape': self.signals.shape
        }
    
    def export_html(self, output_path: str, strategy_name: str = "Backtest Strategy") -> str:
        """
        Export results as interactive HTML tear-sheet
        
        Args:
            output_path: Path to save HTML file
            strategy_name: Name of the strategy for display
            
        Returns:
            Path to generated HTML file
        """
        try:
            from jinja2 import Environment, FileSystemLoader
            
            # Setup Jinja2 environment
            template_dir = Path(__file__).parent.parent.parent / "templates"
            env = Environment(loader=FileSystemLoader(str(template_dir)))
            template = env.get_template("tear_sheet.html.j2")
            
            # Prepare chart data
            chart_data = self._prepare_chart_data()
            
            # Render template
            html_content = template.render(
                strategy_name=strategy_name,
                start_date=self.start_date,
                end_date=self.end_date,
                metrics=self.metrics,
                equity_curve_data=chart_data['equity_curve'],
                monthly_returns_data=chart_data['monthly_returns'],
                annual_returns_data=chart_data['annual_returns'],
                generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Write to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(output_file)
            
        except ImportError as e:
            raise ImportError("jinja2 is required for HTML export")
        except Exception as e:
            raise RuntimeError(f"Failed to export HTML: {e}")
    
    def _prepare_chart_data(self) -> Dict:
        """Prepare data for Plotly charts in HTML template"""
        try:
            if self.portfolio is None:
                # Return empty data if no portfolio available
                return {
                    'equity_curve': {'dates': [], 'values': []},
                    'monthly_returns': {'returns': [], 'years': [], 'months': []},
                    'annual_returns': {'years': [], 'returns': []}
                }
            
            # Equity curve data
            portfolio_value = self.portfolio.value
            equity_curve = {
                'dates': [d.strftime('%Y-%m-%d') for d in portfolio_value.index],
                'values': portfolio_value.values.tolist()
            }
            
            # Monthly returns data for heatmap
            returns = self.portfolio.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            returns_pct = (returns * 100).round(2)
            
            # Create monthly heatmap data
            monthly_data = []
            years = []
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            if len(returns) > 0:
                year_range = range(returns.index.year.min(), returns.index.year.max() + 1)
                
                for year in year_range:
                    year_data = []
                    for month in range(1, 13):
                        try:
                            value = returns_pct[f'{year}-{month:02d}']
                            year_data.append(float(value))
                        except (KeyError, IndexError):
                            year_data.append(None)
                    monthly_data.append(year_data)
                    years.append(str(year))
            
            monthly_returns = {
                'returns': monthly_data,
                'years': years,
                'months': months
            }
            
            # Annual returns data
            if len(returns) > 0:
                annual_returns_series = self.portfolio.returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
                annual_returns_pct = (annual_returns_series * 100).round(2)
                
                annual_returns = {
                    'years': [str(d.year) for d in annual_returns_pct.index],
                    'returns': annual_returns_pct.values.tolist()
                }
            else:
                annual_returns = {'years': [], 'returns': []}
            
            return {
                'equity_curve': equity_curve,
                'monthly_returns': monthly_returns,
                'annual_returns': annual_returns
            }
            
        except Exception as e:
            # Return empty data structure as fallback
            return {
                'equity_curve': {'dates': [], 'values': []},
                'monthly_returns': {'returns': [], 'years': [], 'months': []},
                'annual_returns': {'years': [], 'returns': []}
            }


def create_simple_signals(symbols: list, start: str, end: str, 
                         frequency: str = 'monthly') -> pd.DataFrame:
    """
    Create simple buy-and-hold signals for testing
    
    Args:
        symbols: List of symbols to trade
        start: Start date
        end: End date
        frequency: Rebalancing frequency ('daily', 'weekly', 'monthly')
        
    Returns:
        DataFrame with boolean trading signals
    """
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    # Create DataFrame with all False initially
    signals = pd.DataFrame(False, index=date_range, columns=symbols)
    
    if frequency == 'monthly':
        # Buy on first trading day of each month
        monthly_dates = signals.resample('MS').first().index
        signals.loc[monthly_dates, :] = True
    elif frequency == 'weekly':
        # Buy on Mondays
        signals.loc[signals.index.dayofweek == 0, :] = True
    else:  # daily
        # Buy every day (buy and hold)
        signals.loc[:, :] = True
    
    return signals