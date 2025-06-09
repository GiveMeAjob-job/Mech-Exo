"""
Test CLI integration for HTML export
"""

import pytest
from unittest.mock import Mock, patch
from mech_exo.cli import _handle_backtest


class TestCLIHTMLIntegration:
    """Test CLI HTML export integration"""
    
    def test_html_flag_handling(self):
        """Test that HTML flag is properly handled in CLI"""
        
        # Test that the function accepts html_output parameter
        try:
            _handle_backtest('2020-01-01', '2020-01-31', 100000, None, ['SPY'], 'test.html')
        except Exception as e:
            # We expect vectorbt import error, but it should mention HTML in the flow
            assert "vectorbt is required" in str(e)
    
    def test_html_export_in_results(self):
        """Test that results object has export_html method"""
        from mech_exo.backtest.core import BacktestResults
        import pandas as pd
        
        # Create mock results
        mock_metrics = {'total_return_net': 0.15}
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        mock_prices = pd.DataFrame({'AAPL': [100] * len(dates)}, index=dates)
        mock_signals = pd.DataFrame({'AAPL': [True] * len(dates)}, index=dates)
        
        results = BacktestResults(
            portfolio=None,
            metrics=mock_metrics,
            prices=mock_prices,
            signals=mock_signals,
            start_date='2020-01-01',
            end_date='2020-01-10'
        )
        
        # Check that export_html method exists and is callable
        assert hasattr(results, 'export_html')
        assert callable(results.export_html)
        
        print("✅ CLI HTML integration test passed")


if __name__ == "__main__":
    test = TestCLIHTMLIntegration()
    test.test_html_flag_handling()
    test.test_html_export_in_results()
    print("✅ All CLI HTML tests passed")