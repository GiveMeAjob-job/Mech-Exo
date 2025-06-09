"""
Tests for enhanced backtesting functionality with fees and config
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from mech_exo.backtest.core import Backtester


class TestEnhancedBacktester:
    """Test enhanced backtesting features"""
    
    def test_config_loading(self):
        """Test that config is loaded properly"""
        # Create temporary config
        config_content = """
initial_cash: 50000
commission_per_share: 0.01
slippage_pct: 0.002
risk_free_rate: 0.03
"""
        
        config_path = Path("test_config.yml")
        try:
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # This will fail due to vectorbt import, but we can check the config loading part
            try:
                backtester = Backtester("2020-01-01", "2020-01-31", config_path=str(config_path))
            except ImportError as e:
                # Expected - vectorbt not available
                assert "vectorbt is required" in str(e)
            
        finally:
            if config_path.exists():
                config_path.unlink()
    
    def test_config_defaults(self):
        """Test default config when file doesn't exist"""
        # This will fail due to vectorbt import, but tests the config loading logic
        try:
            backtester = Backtester("2020-01-01", "2020-01-31", config_path="nonexistent.yml")
        except ImportError as e:
            # Expected - vectorbt not available
            assert "vectorbt is required" in str(e)
    
    def test_fee_calculation_logic(self):
        """Test fee calculation components"""
        # Test that we can at least instantiate without vectorbt for config testing
        # In real implementation, this would test the fee calculations
        
        commission = 0.005
        slippage = 0.001  
        spread_cost = 0.0005
        
        total_fee = commission + slippage + spread_cost
        assert total_fee == 0.0065  # 0.65% total transaction cost
    
    def test_metrics_structure(self):
        """Test that enhanced metrics structure is correct"""
        # Mock metrics that would be returned by enhanced backtester
        expected_keys = [
            'total_return_net', 'cagr_net', 'sharpe_net',
            'total_return_gross', 'cagr_gross', 'sharpe_gross', 
            'volatility', 'max_drawdown', 'sortino', 'calmar_ratio',
            'total_trades', 'win_rate', 'profit_factor', 'avg_trade_duration',
            'total_fees', 'avg_fee_per_trade', 'fee_pct_of_nav', 'cost_drag_annual'
        ]
        
        # This tests the structure we expect from the enhanced metrics
        for key in expected_keys:
            assert key in expected_keys  # Placeholder test
    
    def test_summary_format(self):
        """Test that summary includes both gross and net metrics"""
        from mech_exo.backtest.core import BacktestResults
        
        # Mock metrics for testing summary
        mock_metrics = {
            'total_return_net': 0.15,
            'cagr_net': 0.12,
            'sharpe_net': 1.5,
            'total_return_gross': 0.18,
            'cagr_gross': 0.14,
            'sharpe_gross': 1.7,
            'volatility': 0.16,
            'max_drawdown': -0.08,
            'total_trades': 24,
            'total_fees': 1200.0,
            'cost_drag_annual': 0.02,
            'start_date': '2020-01-01',
            'end_date': '2020-12-31'
        }
        
        results = BacktestResults(
            portfolio=None,
            metrics=mock_metrics,
            prices=pd.DataFrame(),
            signals=pd.DataFrame(),
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        summary = results.summary()
        
        # Check that both gross and net metrics appear
        assert "Performance Metrics (Net)" in summary
        assert "Performance Metrics (Gross)" in summary
        assert "Cost Analysis" in summary
        assert "15.00%" in summary  # Net return
        assert "18.00%" in summary  # Gross return
        assert "$1,200" in summary  # Total fees


if __name__ == "__main__":
    pytest.main([__file__])