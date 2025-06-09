#!/usr/bin/env python
# coding: utf-8

# # 📊 Mech-Exo Backtesting Demo
# 
# This notebook demonstrates the comprehensive backtesting capabilities of the Mech-Exo trading system.
# 
# ## Features Covered:
# - **Signal Generation**: Converting idea rankings to trading signals
# - **Historical Backtesting**: Running vectorbt-based performance analysis
# - **Cost Modeling**: Realistic fees, slippage, and cost impact analysis  
# - **Interactive Tear-Sheets**: HTML reports with Plotly visualizations
# - **Walk-Forward Analysis**: Out-of-sample validation with rolling windows
# 
# **Estimated Runtime**: 60-90 seconds offline

# In[ ]:


# Setup Environment - Use stub mode for offline demo
get_ipython().run_line_magic('env', 'EXO_MODE=stub')

import warnings
warnings.filterwarnings('ignore')

# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# Set up paths
import sys
notebook_dir = Path().absolute()
project_root = notebook_dir.parent
sys.path.insert(0, str(project_root))

print(f"📁 Working directory: {notebook_dir}")
print(f"🏠 Project root: {project_root}")
print(f"🔧 Environment mode: {os.getenv('EXO_MODE', 'not set')}")


# ## 🎯 Step 1: Generate Sample Idea Rankings
# 
# We'll create synthetic idea rankings that simulate real factor-based scoring output.

# In[ ]:


# Generate toy ranking data for demonstration
np.random.seed(42)  # For reproducible results

# Create 6-month sample period for fast execution
start_date = '2023-01-01'
end_date = '2023-06-30'
dates = pd.date_range(start_date, end_date, freq='D')

# Sample universe of ETFs
symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'BND', 'GLD']

# Generate realistic ranking scores with trends and noise
ranking_data = {}
for i, symbol in enumerate(symbols):
    # Create trending scores with some randomness
    trend = np.sin(np.arange(len(dates)) * 0.02 + i) * 0.5
    noise = np.random.randn(len(dates)) * 0.3
    ranking_data[symbol] = trend + noise + np.random.randn() * 0.5

# Create ranking DataFrame
idea_rankings = pd.DataFrame(ranking_data, index=dates)

print(f"📊 Generated rankings for {len(symbols)} symbols over {len(dates)} days")
print(f"📅 Period: {start_date} to {end_date}")
print(f"\n🎯 Sample ranking scores (last 5 days):")
print(idea_rankings.tail().round(3))


# ## 🔄 Step 2: Convert Rankings to Trading Signals
# 
# Transform the idea rankings into boolean trading signals using our signal builder.

# In[ ]:


from mech_exo.backtest.signal_builder import idea_rank_to_signals

# Convert rankings to trading signals
signals = idea_rank_to_signals(
    rank_df=idea_rankings,
    n_top=3,                    # Hold top 3 ideas
    holding_period=14,          # Minimum 14-day holding period
    rebal_freq='weekly',        # Weekly rebalancing
    start_date=start_date,
    end_date=end_date
)

print(f"🔄 Generated trading signals:")
print(f"   Shape: {signals.shape}")
print(f"   Active positions per day: {signals.sum(axis=1).mean():.1f} average")
print(f"   Total signal changes: {signals.diff().abs().sum().sum():.0f}")

# Show sample signals
print(f"\n📋 Sample signals (last 10 days):")
sample_signals = signals.tail(10)
for date, row in sample_signals.iterrows():
    active = [symbol for symbol, active in row.items() if active]
    print(f"   {date.strftime('%Y-%m-%d')}: {active if active else 'No positions'}")


# ## 🔬 Step 3: Run Historical Backtest
# 
# Execute a comprehensive backtest with realistic cost modeling.

# In[ ]:


# Import backtesting components
try:
    from mech_exo.backtest.core import Backtester
    
    # Initialize backtester with realistic costs
    backtester = Backtester(
        start=start_date,
        end=end_date,
        cash=100000,  # $100k starting capital
        commission=0.005,  # $0.005 per share
        slippage=0.001     # 0.1% slippage
    )
    
    print(f"🔬 Running backtest...")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Initial capital: $100,000")
    print(f"   Signals shape: {signals.shape}")
    
    # Run backtest
    results = backtester.run(signals)
    
    # Display comprehensive results
    print(f"\n{results.summary()}")
    
    backtest_available = True
    
except ImportError as e:
    print(f"⚠️  Vectorbt not available for full backtesting: {e}")
    print(f"📊 Simulating backtest results for demo purposes...")
    
    # Create mock results for demonstration
    mock_metrics = {
        'total_return_net': 0.08,
        'cagr_net': 0.16,  # Annualized
        'sharpe_net': 1.45,
        'volatility': 0.12,
        'max_drawdown': -0.06,
        'total_trades': 18,
        'win_rate': 0.67,
        'total_fees': 450.0,
        'cost_drag_annual': 0.012
    }
    
    print(f"\n📈 Mock Backtest Results (6-month period):")
    print(f"   Total Return: {mock_metrics['total_return_net']:.2%}")
    print(f"   Annualized CAGR: {mock_metrics['cagr_net']:.2%}")
    print(f"   Sharpe Ratio: {mock_metrics['sharpe_net']:.2f}")
    print(f"   Max Drawdown: {mock_metrics['max_drawdown']:.2%}")
    print(f"   Total Trades: {mock_metrics['total_trades']}")
    print(f"   Win Rate: {mock_metrics['win_rate']:.1%}")
    print(f"   Total Fees: ${mock_metrics['total_fees']:.0f}")
    print(f"   Annual Cost Drag: {mock_metrics['cost_drag_annual']:.2%}")
    
    backtest_available = False


# ## 📄 Step 4: Generate Interactive Tear-Sheet
# 
# Create an HTML tear-sheet with interactive Plotly charts.

# In[ ]:


# Generate tear-sheet (with fallback for demo)
tearsheet_path = None

if backtest_available:
    try:
        # Generate real tear-sheet
        tearsheet_path = results.export_html(
            "demo_tearsheet.html", 
            strategy_name="Demo Strategy"
        )
        print(f"📄 Interactive tear-sheet generated: {tearsheet_path}")
        
    except Exception as e:
        print(f"⚠️  Tear-sheet generation failed: {e}")
else:
    # Create simple demo tear-sheet for visualization
    tearsheet_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Demo Tear-Sheet</title></head>
    <body style="font-family: Arial; padding: 20px;">
        <h1>📊 Demo Strategy Tear-Sheet</h1>
        <h2>Performance Summary</h2>
        <ul>
            <li><strong>Total Return</strong>: 8.00%</li>
            <li><strong>Annualized CAGR</strong>: 16.00%</li>
            <li><strong>Sharpe Ratio</strong>: 1.45</li>
            <li><strong>Max Drawdown</strong>: -6.00%</li>
            <li><strong>Win Rate</strong>: 66.7%</li>
        </ul>
        <p><em>Note: This is a demo tear-sheet. Full interactive charts available with vectorbt.</em></p>
    </body>
    </html>
    """
    
    tearsheet_path = "demo_tearsheet_mock.html"
    with open(tearsheet_path, 'w') as f:
        f.write(tearsheet_html)
    
    print(f"📄 Demo tear-sheet created: {tearsheet_path}")

# Display tear-sheet in iframe if available
if tearsheet_path and Path(tearsheet_path).exists():
    from IPython.display import IFrame, HTML
    
    print(f"\n🌐 Tear-sheet preview:")
    
    # Embed in iframe
    display(IFrame(tearsheet_path, width=800, height=400))
    
    print(f"\n💡 Open '{tearsheet_path}' in browser for full interactive experience")
else:
    print(f"❌ Tear-sheet file not found")


# ## 🚶 Step 5: Walk-Forward Analysis
# 
# Demonstrate out-of-sample validation with rolling windows.

# In[ ]:


try:
    from mech_exo.backtest.walk_forward import WalkForwardAnalyzer, make_walk_windows
    
    # Generate walk-forward windows (short periods for demo)
    windows = make_walk_windows(
        start='2023-01-01', 
        end='2023-06-30', 
        train='60D',  # 2-month training
        test='30D'    # 1-month testing
    )
    
    print(f"🚶 Walk-Forward Analysis Setup:")
    print(f"   Generated {len(windows)} windows")
    print(f"   Training period: 60 days")
    print(f"   Test period: 30 days")
    
    if windows:
        print(f"\n📅 Walk-Forward Windows:")
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
            print(f"   Window {i}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
    
    # Simulate walk-forward results (since vectorbt may not be available)
    print(f"\n📊 Simulated Walk-Forward Results:")
    
    # Mock segment results
    segment_results = [
        {'window': 1, 'test_period': '2023-03-01 to 2023-03-31', 'cagr': 0.18, 'sharpe': 1.6, 'max_dd': -0.05, 'trades': 8},
        {'window': 2, 'test_period': '2023-04-01 to 2023-04-30', 'cagr': 0.14, 'sharpe': 1.3, 'max_dd': -0.08, 'trades': 6},
        {'window': 3, 'test_period': '2023-05-01 to 2023-05-31', 'cagr': 0.22, 'sharpe': 1.8, 'max_dd': -0.04, 'trades': 7}
    ]
    
    # Display results table
    results_df = pd.DataFrame(segment_results)
    print("\n📋 Segment Performance:")
    print("Window | Test Period       | CAGR   | Sharpe | Max DD | Trades")
    print("-------|-------------------|--------|--------|--------|-------")
    for _, row in results_df.iterrows():
        print(f"   {row['window']:<3} | {row['test_period']:<17} | {row['cagr']:>5.1%} | {row['sharpe']:>5.1f} | {row['max_dd']:>5.1%} | {row['trades']:>5}")
    
    # Summary statistics
    print(f"\n📊 Aggregate Statistics:")
    print(f"   Mean CAGR: {results_df['cagr'].mean():.1%} ± {results_df['cagr'].std():.1%}")
    print(f"   Mean Sharpe: {results_df['sharpe'].mean():.2f} ± {results_df['sharpe'].std():.2f}")
    print(f"   Worst Max DD: {results_df['max_dd'].min():.1%}")
    print(f"   Total Trades: {results_df['trades'].sum()}")
    
    walkforward_available = True
    
except ImportError as e:
    print(f"⚠️  Walk-forward analysis not available: {e}")
    print(f"📊 This feature requires the full backtesting environment")
    walkforward_available = False
except Exception as e:
    print(f"⚠️  Walk-forward analysis error: {e}")
    walkforward_available = False


# ## 📈 Step 6: Visualize Performance
# 
# Create performance visualizations for the notebook.

# In[ ]:


# Create performance visualization
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('📊 Mech-Exo Backtesting Demo Results', fontsize=16, fontweight='bold')

# 1. Equity Curve (simulated)
dates_plot = pd.date_range(start_date, end_date, freq='D')
np.random.seed(42)
returns = np.random.randn(len(dates_plot)) * 0.01 + 0.0005  # Slight positive drift
cumulative_returns = (1 + pd.Series(returns)).cumprod()
equity_curve = cumulative_returns * 100000  # Starting with $100k

ax1.plot(dates_plot, equity_curve, color='#2E86AB', linewidth=2)
ax1.set_title('Equity Curve', fontweight='bold')
ax1.set_ylabel('Portfolio Value ($)')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Monthly Returns Heatmap (simulated)
monthly_returns = pd.Series(returns, index=dates_plot).resample('M').apply(lambda x: (1 + x).prod() - 1)
monthly_returns_matrix = monthly_returns.values.reshape(1, -1) * 100

im = ax2.imshow(monthly_returns_matrix, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=3)
ax2.set_title('Monthly Returns (%)', fontweight='bold')
ax2.set_xticks(range(len(monthly_returns)))
ax2.set_xticklabels([d.strftime('%b') for d in monthly_returns.index], rotation=45)
ax2.set_yticks([])

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.6)
cbar.set_label('Return (%)')

# 3. Signal Activity
daily_positions = signals.sum(axis=1)
ax3.plot(daily_positions.index, daily_positions.values, color='#A23B72', linewidth=1.5)
ax3.fill_between(daily_positions.index, daily_positions.values, alpha=0.3, color='#A23B72')
ax3.set_title('Daily Active Positions', fontweight='bold')
ax3.set_ylabel('Number of Positions')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Performance Metrics Bar Chart
metrics_names = ['Total Return\n(%)', 'Sharpe\nRatio', 'Max DD\n(%)', 'Win Rate\n(%)']
if backtest_available:
    metrics_values = [
        results.metrics.get('total_return_net', 0) * 100,
        results.metrics.get('sharpe_net', 0),
        abs(results.metrics.get('max_drawdown', 0)) * 100,
        results.metrics.get('win_rate', 0) * 100
    ]
else:
    metrics_values = [8.0, 1.45, 6.0, 66.7]  # Mock values

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
ax4.set_title('Key Performance Metrics', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('backtest_demo_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📊 Performance visualization completed")
print(f"💾 Chart saved as 'backtest_demo_results.png' ({os.path.getsize('backtest_demo_results.png') / 1024:.1f} KB)")


# ## 🧹 Step 7: Cleanup
# 
# Clean up temporary files created during the demo.

# In[ ]:


# Cleanup temporary files
cleanup_files = []

# Add tear-sheet files to cleanup
for filename in ['demo_tearsheet.html', 'demo_tearsheet_mock.html']:
    if Path(filename).exists():
        cleanup_files.append(filename)

# Clean up
for filename in cleanup_files:
    try:
        Path(filename).unlink()
        print(f"🗑️  Cleaned up: {filename}")
    except Exception as e:
        print(f"⚠️  Could not remove {filename}: {e}")

# Keep the chart for documentation
chart_file = Path('backtest_demo_results.png')
if chart_file.exists():
    print(f"💾 Kept chart file: {chart_file} ({chart_file.stat().st_size / 1024:.1f} KB)")

print(f"\n✅ Demo completed successfully!")
print(f"\n📚 Next Steps:")
print(f"   • Run full backtests with: poetry run exo backtest --help")
print(f"   • Try walk-forward analysis: poetry run exo walkforward --help")
print(f"   • Install vectorbt for complete functionality: pip install vectorbt")
print(f"   • Explore more examples in the documentation")

