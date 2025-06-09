"""
Tests for HTML tear-sheet export functionality
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from mech_exo.backtest.core import BacktestResults


class TestHTMLExport:
    """Test HTML export functionality"""
    
    def test_html_export_with_mock_data(self):
        """Test HTML export with mock backtest results"""
        
        # Create mock metrics
        mock_metrics = {
            'total_return_net': 0.15,
            'cagr_net': 0.12,
            'sharpe_net': 1.5,
            'total_return_gross': 0.18,
            'cagr_gross': 0.14,
            'sharpe_gross': 1.7,
            'volatility': 0.16,
            'max_drawdown': -0.08,
            'sortino': 1.8,
            'calmar_ratio': 1.5,
            'total_trades': 24,
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'avg_trade_duration': 15.5,
            'total_fees': 1200.0,
            'avg_fee_per_trade': 50.0,
            'fee_pct_of_nav': 0.012,
            'cost_drag_annual': 0.02,
            'start_date': '2020-01-01',
            'end_date': '2020-12-31'
        }
        
        # Create mock portfolio data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        mock_prices = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01),
            'MSFT': 200 + np.cumsum(np.random.randn(len(dates)) * 0.01)
        }, index=dates)
        
        mock_signals = pd.DataFrame({
            'AAPL': [True, False] * (len(dates) // 2) + [True] * (len(dates) % 2),
            'MSFT': [False, True] * (len(dates) // 2) + [False] * (len(dates) % 2)
        }, index=dates)
        
        # Create results object (without vectorbt portfolio for testing)
        results = BacktestResults(
            portfolio=None,  # Testing without actual vectorbt portfolio
            metrics=mock_metrics,
            prices=mock_prices,
            signals=mock_signals,
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Test HTML export
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            try:
                html_path = results.export_html(tmp_file.name, strategy_name="Test Strategy")
                
                # Check that file was created
                assert Path(html_path).exists()
                
                # Read file content
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Check that key elements are present
                assert "Total Return" in html_content
                assert "Test Strategy" in html_content
                assert "2020-01-01" in html_content
                assert "2020-12-31" in html_content
                assert "15.00%" in html_content  # Total return net
                assert "plotly" in html_content.lower()  # Plotly scripts
                assert "Sharpe Ratio" in html_content
                
                print(f"✅ HTML export test passed - file created at {html_path}")
                
            except ImportError:
                pytest.skip("jinja2 not available for HTML export testing")
            finally:
                # Clean up
                if Path(tmp_file.name).exists():
                    Path(tmp_file.name).unlink()
    
    def test_template_exists(self):
        """Test that the HTML template file exists"""
        template_path = Path(__file__).parent.parent / "templates" / "tear_sheet.html.j2"
        assert template_path.exists(), f"Template not found at {template_path}"
        
        # Check template contains key sections
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        assert "Total Return" in template_content
        assert "Sharpe Ratio" in template_content
        assert "equity-chart" in template_content
        assert "monthly-heatmap" in template_content
        assert "annual-returns" in template_content
        assert "plotly" in template_content.lower()
    
    def test_chart_data_preparation(self):
        """Test chart data preparation with mock data"""
        
        # Create mock results
        mock_metrics = {'total_return_net': 0.15, 'cagr_net': 0.12}
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        mock_prices = pd.DataFrame({'AAPL': [100] * len(dates)}, index=dates)
        mock_signals = pd.DataFrame({'AAPL': [True] * len(dates)}, index=dates)
        
        results = BacktestResults(
            portfolio=None,
            metrics=mock_metrics,
            prices=mock_prices,
            signals=mock_signals,
            start_date='2020-01-01',
            end_date='2020-01-31'
        )
        
        # Test chart data preparation
        chart_data = results._prepare_chart_data()
        
        # Should return proper structure even without portfolio
        assert 'equity_curve' in chart_data
        assert 'monthly_returns' in chart_data
        assert 'annual_returns' in chart_data
        
        # Check structure
        assert isinstance(chart_data['equity_curve'], dict)
        assert 'dates' in chart_data['equity_curve']
        assert 'values' in chart_data['equity_curve']


if __name__ == "__main__":
    pytest.main([__file__])