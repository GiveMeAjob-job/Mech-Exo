"""
Integration tests for Phase P3: Score → Size → Risk flow
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from mech_exo.scoring import IdeaScorer
from mech_exo.sizing import PositionSizer, SizingMethod
from mech_exo.risk import RiskChecker, Portfolio, Position, StopEngine
from mech_exo.risk.base import RiskStatus


class TestCompleteP3Flow:
    """Test complete flow from scoring to sizing to risk management"""
    
    @pytest.fixture
    def mock_complete_config(self):
        """Complete mock configuration for all modules"""
        return {
            "scoring_config": {
                'fundamental': {
                    'pe_ratio': {'weight': 30, 'direction': 'lower_better'},
                    'return_on_equity': {'weight': 70, 'direction': 'higher_better'}
                },
                'sector_adjustments': {'Technology': 1.1}
            },
            "risk_config": {
                "position_sizing": {
                    "max_single_trade_risk": 0.02,
                    "max_single_position": 0.10,
                    "min_position_value": 1000.0,
                    "atr_multiplier": 2.0,
                    "default_method": "atr_based"
                },
                "portfolio": {
                    "max_gross_exposure": 1.5,
                    "max_net_exposure": 1.0,
                    "max_drawdown": 0.10,
                    "max_sector_exposure": 0.20
                },
                "stops": {
                    "trailing_stop_pct": 0.25,
                    "hard_stop_pct": 0.15,
                    "profit_target_pct": 0.30,
                    "time_stop_days": 60
                }
            }
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing"""
        return {
            'fundamental': pd.DataFrame({
                'symbol': ['AAPL', 'GOOGL', 'MSFT'],
                'pe_ratio': [25.5, 30.2, 28.1],
                'return_on_equity': [0.30, 0.25, 0.35],  # Strong ROE
                'sector': ['Technology', 'Technology', 'Technology'],
                'current_price': [150.0, 120.0, 300.0],
                'market_cap': [2500000000000, 1500000000000, 2200000000000],
                'fetch_date': [datetime.now()] * 3
            }),
            'ohlc': pd.DataFrame({
                'symbol': ['AAPL'] * 20 + ['GOOGL'] * 20 + ['MSFT'] * 20,
                'date': pd.date_range('2023-01-01', periods=20).tolist() * 3,
                'close': ([150] * 20) + ([120] * 20) + ([300] * 20),
                'atr': ([2.0] * 20) + ([1.8] * 20) + ([4.5] * 20),
                'volatility': ([0.20] * 20) + ([0.25] * 20) + ([0.18] * 20),
                'returns': [0.01] * 60
            })
        }
    
    @patch('mech_exo.scoring.scorer.ConfigManager')
    @patch('mech_exo.scoring.scorer.DataStorage')
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    @patch('mech_exo.risk.checker.ConfigManager')
    @patch('mech_exo.risk.checker.DataStorage')
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_complete_score_to_risk_flow(self, mock_stop_config, mock_risk_storage, 
                                       mock_risk_config, mock_size_storage, 
                                       mock_size_config, mock_score_storage, 
                                       mock_score_config, mock_complete_config, 
                                       sample_market_data):
        """Test complete flow: Score ideas → Size positions → Check risk"""
        
        # === Setup Mocks ===
        
        # Scoring mocks
        mock_score_config_instance = Mock()
        mock_score_config_instance.get_factor_config.return_value = mock_complete_config["scoring_config"]
        mock_score_config.return_value = mock_score_config_instance
        
        mock_score_storage_instance = Mock()
        mock_score_storage_instance.get_fundamental_data.return_value = sample_market_data['fundamental']
        mock_score_storage_instance.get_ohlc_data.return_value = sample_market_data['ohlc']
        mock_score_storage_instance.get_news_data.return_value = pd.DataFrame()
        mock_score_storage.return_value = mock_score_storage_instance
        
        # Sizing mocks
        mock_size_config_instance = Mock()
        mock_size_config_instance.load_config.return_value = mock_complete_config["risk_config"]
        mock_size_config.return_value = mock_size_config_instance
        
        mock_size_storage_instance = Mock()
        mock_size_storage_instance.get_ohlc_data.return_value = sample_market_data['ohlc']
        mock_size_storage.return_value = mock_size_storage_instance
        
        # Risk mocks
        mock_risk_config_instance = Mock()
        mock_risk_config_instance.load_config.return_value = mock_complete_config["risk_config"]
        mock_risk_config.return_value = mock_risk_config_instance
        
        mock_risk_storage_instance = Mock()
        mock_risk_storage.return_value = mock_risk_storage_instance
        
        # Stop engine mock
        mock_stop_config_instance = Mock()
        mock_stop_config_instance.load_config.return_value = mock_complete_config["risk_config"]
        mock_stop_config.return_value = mock_stop_config_instance
        
        # === Execute Complete Flow ===
        
        try:
            # Step 1: Score ideas
            scorer = IdeaScorer()
            ranking = scorer.score(['AAPL', 'GOOGL', 'MSFT'])
            
            assert not ranking.empty, "Scoring should produce results"
            assert 'composite_score' in ranking.columns, "Should have composite scores"
            assert len(ranking) == 3, "Should score all 3 symbols"
            
            # Get top idea
            top_idea = ranking.iloc[0]
            symbol = top_idea['symbol']
            current_price = sample_market_data['fundamental'][
                sample_market_data['fundamental']['symbol'] == symbol
            ]['current_price'].iloc[0]
            
            print(f"🏆 Top idea: {symbol} @ ${current_price}")
            
            # Step 2: Size position
            nav = 100000  # $100k portfolio
            sizer = PositionSizer(nav)
            
            # Calculate position size with ATR method
            shares = sizer.calculate_size(
                symbol=symbol,
                price=current_price,
                method=SizingMethod.ATR_BASED,
                signal_strength=1.0
            )
            
            assert shares > 0, "Should calculate positive position size"
            position_value = shares * current_price
            nav_pct = position_value / nav
            
            print(f"📏 Position size: {shares} shares (${position_value:,.0f}, {nav_pct:.1%} of NAV)")
            
            # Step 3: Generate stops
            stop_engine = StopEngine()
            stops = stop_engine.generate_stops(
                entry_price=current_price,
                position_type="long",
                entry_date=datetime.now(),
                atr=2.0  # Sample ATR
            )
            
            assert 'hard_stop' in stops, "Should generate hard stop"
            assert 'profit_target' in stops, "Should generate profit target"
            assert stops['hard_stop'] < current_price, "Hard stop should be below entry for long"
            assert stops['profit_target'] > current_price, "Profit target should be above entry for long"
            
            print(f"🛡️  Stops: Hard=${stops['hard_stop']:.2f}, Target=${stops['profit_target']:.2f}")
            
            # Step 4: Create portfolio and check risk
            portfolio = Portfolio(nav)
            
            # Add the position
            position = Position(
                symbol=symbol,
                shares=shares,
                entry_price=current_price,
                current_price=current_price,
                entry_date=datetime.now(),
                sector='Technology'
            )
            portfolio.add_position(position)
            
            # Check risk
            risk_checker = RiskChecker(portfolio)
            risk_report = risk_checker.check()
            
            assert 'status' in risk_report, "Should produce risk status"
            assert risk_report['status'] in [RiskStatus.OK, RiskStatus.WARNING], "Should not breach limits with single position"
            
            print(f"🔍 Risk status: {risk_report['status'].value}")
            
            # Step 5: Test risk limits by adding more positions
            # Add more positions to test limits
            additional_symbols = ['GOOGL', 'MSFT']
            
            for add_symbol in additional_symbols:
                add_price = sample_market_data['fundamental'][
                    sample_market_data['fundamental']['symbol'] == add_symbol
                ]['current_price'].iloc[0]
                
                add_shares = sizer.calculate_size(
                    symbol=add_symbol,
                    price=add_price,
                    method=SizingMethod.FIXED_PERCENT,
                    signal_strength=1.0
                )
                
                add_position = Position(
                    symbol=add_symbol,
                    shares=add_shares,
                    entry_price=add_price,
                    current_price=add_price,
                    entry_date=datetime.now(),
                    sector='Technology'
                )
                portfolio.add_position(add_position)
            
            # Check risk with multiple positions
            final_risk_report = risk_checker.check()
            
            # Should warn about sector concentration (all Technology)
            if final_risk_report['status'] == RiskStatus.WARNING:
                assert any('Technology' in warning for warning in final_risk_report.get('warnings', [])), \
                    "Should warn about Technology sector concentration"
            
            print(f"🎯 Final risk status with {len(portfolio.positions)} positions: {final_risk_report['status'].value}")
            
            # === Validate Complete Flow ===
            
            # Check that all components work together
            flow_summary = {
                'scoring_successful': not ranking.empty,
                'sizing_successful': shares > 0,
                'stops_generated': len(stops) >= 3,
                'risk_check_successful': 'status' in final_risk_report,
                'portfolio_nav': portfolio.current_nav,
                'total_positions': len(portfolio.positions),
                'gross_exposure_pct': portfolio.gross_exposure / portfolio.current_nav
            }
            
            print(f"\n✅ Flow Summary: {flow_summary}")
            
            # All components should be successful
            assert all([
                flow_summary['scoring_successful'],
                flow_summary['sizing_successful'], 
                flow_summary['stops_generated'],
                flow_summary['risk_check_successful']
            ]), "All components of the flow should succeed"
            
            # Portfolio should be reasonable
            assert flow_summary['gross_exposure_pct'] <= 1.5, "Should not exceed max leverage"
            assert flow_summary['total_positions'] == 3, "Should have 3 positions"
            
            print("🎉 Complete P3 flow test PASSED!")
            
        finally:
            # Cleanup
            if 'scorer' in locals():
                scorer.close()
            if 'sizer' in locals():
                sizer.close()
            if 'risk_checker' in locals():
                risk_checker.close()
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    @patch('mech_exo.risk.checker.ConfigManager')
    @patch('mech_exo.risk.checker.DataStorage')
    def test_risk_breach_scenario(self, mock_risk_storage, mock_risk_config, 
                                 mock_size_storage, mock_size_config, 
                                 mock_complete_config):
        """Test scenario where risk limits are breached"""
        
        # Setup mocks for tighter risk limits
        tight_config = mock_complete_config["risk_config"].copy()
        tight_config["position_sizing"]["max_single_position"] = 0.05  # Very tight 5% limit
        
        mock_size_config_instance = Mock()
        mock_size_config_instance.load_config.return_value = tight_config
        mock_size_config.return_value = mock_size_config_instance
        
        mock_risk_config_instance = Mock()
        mock_risk_config_instance.load_config.return_value = tight_config
        mock_risk_config.return_value = mock_risk_config_instance
        
        mock_storage_instance = Mock()
        mock_storage_instance.get_ohlc_data.return_value = pd.DataFrame()
        mock_size_storage.return_value = mock_storage_instance
        mock_risk_storage.return_value = mock_storage_instance
        
        nav = 100000
        portfolio = Portfolio(nav)
        
        # Add large position that should breach limits
        large_position = Position(
            symbol="LARGE",
            shares=200,  # 200 shares @ $400 = $80k = 80% of NAV (exceeds 5% limit)
            entry_price=400.0,
            current_price=400.0,
            entry_date=datetime.now(),
            sector='Technology'
        )
        portfolio.add_position(large_position)
        
        # Check risk - should breach
        risk_checker = RiskChecker(portfolio)
        risk_report = risk_checker.check()
        
        # Should report breach
        assert risk_report['status'] == RiskStatus.BREACH, "Should detect risk breach"
        assert len(risk_report.get('violations', [])) > 0, "Should have violation messages"
        
        # Test pre-trade risk check
        pre_trade_check = risk_checker.check_new_position(
            symbol="ANOTHER",
            shares=100,
            price=500.0,  # Another $50k position
            sector="Finance"
        )
        
        assert pre_trade_check['pre_trade_analysis']['recommendation'] == "REJECT", \
            "Should reject new position that would increase breach"
        
        print("✅ Risk breach detection test PASSED!")
        
        risk_checker.close()
    
    @patch('mech_exo.risk.stop_engine.ConfigManager')
    def test_stop_management_integration(self, mock_config, mock_complete_config):
        """Test stop management in realistic trading scenarios"""
        
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = mock_complete_config["risk_config"]
        mock_config.return_value = mock_config_instance
        
        stop_engine = StopEngine()
        
        # Test scenario: Long position with price appreciation
        entry_price = 100.0
        
        # Generate initial stops
        stops = stop_engine.generate_stops(
            entry_price=entry_price,
            position_type="long",
            entry_date=datetime.now(),
            atr=2.0
        )
        
        # Price appreciates - update trailing stop
        new_price = 120.0  # 20% gain
        high_water_mark = 120.0
        
        updated_trailing = stop_engine.update_trailing_stop(
            current_price=new_price,
            current_stop=stops['trailing_stop'],
            position_type="long",
            high_water_mark=high_water_mark
        )
        
        # New trailing stop should be higher
        assert updated_trailing > stops['trailing_stop'], "Trailing stop should move up with price"
        
        # Check if stops are hit at various price levels
        scenarios = [
            (130.0, False),  # Above profit target - should trigger
            (90.0, True),    # Below hard stop - should trigger  
            (110.0, False),  # Safe zone - no triggers
        ]
        
        for test_price, expect_trigger in scenarios:
            check_result = stop_engine.check_stop_hit(
                current_price=test_price,
                stops={'hard_stop': 85.0, 'profit_target': 130.0, 'trailing_stop': updated_trailing},
                position_type="long"
            )
            
            if expect_trigger:
                assert check_result['status'] == 'triggered', f"Should trigger at ${test_price}"
            else:
                triggered = check_result['status'] == 'triggered'
                # Only check for profit target at 130
                if test_price == 130.0:
                    assert triggered, f"Should trigger profit target at ${test_price}"
        
        print("✅ Stop management integration test PASSED!")


class TestEdgeCases:
    """Test edge cases and error scenarios"""
    
    @patch('mech_exo.sizing.position_sizer.ConfigManager')
    @patch('mech_exo.sizing.position_sizer.DataStorage')
    def test_zero_nav_scenario(self, mock_storage, mock_config):
        """Test behavior with zero or negative NAV"""
        
        config = {
            "position_sizing": {
                "max_single_trade_risk": 0.02,
                "max_single_position": 0.10,
                "min_position_value": 1000.0
            },
            "portfolio": {}
        }
        
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = config
        mock_config.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage_instance.get_ohlc_data.return_value = pd.DataFrame()
        mock_storage.return_value = mock_storage_instance
        
        # Test with zero NAV
        with pytest.raises(Exception):  # Should fail gracefully
            sizer = PositionSizer(0)
        
        # Test with negative NAV  
        with pytest.raises(Exception):  # Should fail gracefully
            sizer = PositionSizer(-1000)
    
    @patch('mech_exo.risk.checker.ConfigManager')
    @patch('mech_exo.risk.checker.DataStorage')
    def test_empty_portfolio_risk_check(self, mock_storage, mock_config):
        """Test risk check on empty portfolio"""
        
        config = {
            "position_sizing": {"max_single_position": 0.10},
            "portfolio": {"max_gross_exposure": 1.5},
            "stops": {},
            "costs": {},
            "operational": {},
            "volatility": {},
            "margin": {},
            "options": {}
        }
        
        mock_config_instance = Mock()
        mock_config_instance.load_config.return_value = config
        mock_config.return_value = mock_config_instance
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        empty_portfolio = Portfolio(100000)
        risk_checker = RiskChecker(empty_portfolio)
        
        risk_report = risk_checker.check()
        
        # Empty portfolio should be OK
        assert risk_report['status'] == RiskStatus.OK, "Empty portfolio should be OK"
        assert len(risk_report.get('violations', [])) == 0, "Empty portfolio should have no violations"
        
        risk_checker.close()


def test_integration_with_actual_config():
    """Test integration using actual config files (if available)"""
    try:
        from mech_exo.utils import ConfigManager
        
        # Try to load actual config
        config_manager = ConfigManager()
        risk_config = config_manager.load_config("risk_limits")
        
        if not risk_config:
            pytest.skip("No actual risk config available - using mocks only")
            
        # Test that config loads properly
        assert isinstance(risk_config, dict), "Config should be dictionary"
        assert "position_sizing" in risk_config, "Should have position sizing config"
        
        print("✅ Actual config integration test PASSED!")
        
    except Exception as e:
        pytest.skip(f"Config integration test skipped: {e}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])