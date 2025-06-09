"""
Unit tests for daily reporting functionality
"""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mech_exo.reporting.daily import DailyReport, generate_daily_report


class TestDailyReport:
    """Test cases for DailyReport class"""

    def test_parse_date_today(self):
        """Test parsing 'today' date"""
        report = DailyReport(date="today")
        expected_date = datetime.now(UTC).strftime("%Y-%m-%d")
        assert report.date_str == expected_date

    def test_parse_date_specific(self):
        """Test parsing specific date"""
        test_date = "2025-06-08"
        report = DailyReport(date=test_date)
        assert report.date_str == test_date

    def test_parse_date_invalid(self):
        """Test parsing invalid date format"""
        with pytest.raises(ValueError, match="Invalid date format"):
            DailyReport(date="invalid-date")

    @patch('mech_exo.reporting.daily.FillStore')
    def test_empty_fills_summary(self, mock_fill_store):
        """Test summary with no fills"""
        # Mock empty fills
        mock_store = MagicMock()
        mock_store.conn = MagicMock()
        mock_fill_store.return_value = mock_store
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            
            report = DailyReport(date="2025-06-08")
            summary = report.summary()
            
            expected = {
                "date": "2025-06-08",
                "daily_pnl": 0.0,
                "fees": 0.0,
                "max_dd": 0.0,
                "trade_count": 0,
                "volume": 0.0,
                "avg_slippage_bps": 0.0,
                "avg_routing_latency_ms": 0.0,
                "strategies": [],
                "symbols": []
            }
            
            assert summary == expected

    @patch('mech_exo.reporting.daily.FillStore')
    def test_fills_summary(self, mock_fill_store):
        """Test summary with sample fills"""
        # Mock fills data
        mock_store = MagicMock()
        mock_store.conn = MagicMock()
        mock_fill_store.return_value = mock_store
        
        sample_fills = pd.DataFrame({
            'fill_id': ['fill1', 'fill2'],
            'order_id': ['order1', 'order2'],
            'symbol': ['AAPL', 'GOOGL'],
            'quantity': [100, -50],
            'price': [150.0, 120.0],
            'commission': [1.0, 0.5],
            'timestamp': [
                datetime.now(UTC),
                datetime.now(UTC) + timedelta(hours=1)
            ],
            'strategy': ['momentum', 'mean_reversion'],
            'signal_strength': [0.8, 0.6],
            'slippage_bps': [2.5, 1.8],
            'routing_latency_ms': [10.0, 8.0],
            'broker_latency_ms': [50.0, 45.0]
        })
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = sample_fills
            
            report = DailyReport(date="2025-06-08")
            summary = report.summary()
            
            # Verify key metrics
            assert summary['date'] == "2025-06-08"
            assert summary['trade_count'] == 2
            assert summary['fees'] == 1.5
            assert summary['avg_slippage_bps'] == 2.15  # (2.5 + 1.8) / 2
            assert summary['avg_routing_latency_ms'] == 9.0  # (10.0 + 8.0) / 2
            assert set(summary['strategies']) == {'momentum', 'mean_reversion'}
            assert set(summary['symbols']) == {'AAPL', 'GOOGL'}

    @patch('mech_exo.reporting.daily.FillStore')
    def test_max_drawdown_calculation(self, mock_fill_store):
        """Test maximum drawdown calculation"""
        mock_store = MagicMock()
        mock_store.conn = MagicMock()
        mock_fill_store.return_value = mock_store
        
        # Create fills with varying P&L to test drawdown
        sample_fills = pd.DataFrame({
            'fill_id': ['fill1', 'fill2', 'fill3'],
            'order_id': ['order1', 'order2', 'order3'],
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'quantity': [100, 100, 100],
            'price': [150.0, 151.0, 149.0],
            'commission': [1.0, 1.0, 1.0],
            'timestamp': [
                datetime(2025, 6, 8, 9, 0, tzinfo=UTC),
                datetime(2025, 6, 8, 10, 0, tzinfo=UTC),
                datetime(2025, 6, 8, 11, 0, tzinfo=UTC)
            ],
            'strategy': ['test', 'test', 'test'],
            'signal_strength': [1.0, 1.0, 1.0],
            'slippage_bps': [0.0, 0.0, 0.0],
            'routing_latency_ms': [0.0, 0.0, 0.0],
            'broker_latency_ms': [0.0, 0.0, 0.0]
        })
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = sample_fills
            
            report = DailyReport(date="2025-06-08")
            max_dd = report._calculate_max_drawdown()
            
            # Should have some drawdown from the P&L pattern
            assert isinstance(max_dd, float)

    @patch('mech_exo.reporting.daily.FillStore')
    def test_detailed_breakdown(self, mock_fill_store):
        """Test detailed breakdown functionality"""
        mock_store = MagicMock()
        mock_store.conn = MagicMock()
        mock_fill_store.return_value = mock_store
        
        sample_fills = pd.DataFrame({
            'fill_id': ['fill1', 'fill2'],
            'order_id': ['order1', 'order2'],
            'symbol': ['AAPL', 'GOOGL'],
            'quantity': [100, -50],
            'price': [150.0, 120.0],
            'commission': [1.0, 0.5],
            'timestamp': [
                datetime(2025, 6, 8, 9, 0, tzinfo=UTC),
                datetime(2025, 6, 8, 10, 0, tzinfo=UTC)
            ],
            'strategy': ['momentum', 'mean_reversion'],
            'signal_strength': [0.8, 0.6],
            'slippage_bps': [2.5, 1.8],
            'routing_latency_ms': [10.0, 8.0],
            'broker_latency_ms': [50.0, 45.0]
        })
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = sample_fills
            
            report = DailyReport(date="2025-06-08")
            breakdown = report.detailed_breakdown()
            
            # Check structure
            assert 'date' in breakdown
            assert 'by_strategy' in breakdown
            assert 'by_symbol' in breakdown
            assert 'hourly_pnl' in breakdown
            
            # Check strategy breakdown
            assert 'momentum' in breakdown['by_strategy']
            assert 'mean_reversion' in breakdown['by_strategy']
            
            # Check symbol breakdown
            assert 'AAPL' in breakdown['by_symbol']
            assert 'GOOGL' in breakdown['by_symbol']

    @patch('mech_exo.reporting.daily.FillStore')
    def test_to_json(self, mock_fill_store):
        """Test JSON export functionality"""
        mock_store = MagicMock()
        mock_store.conn = MagicMock()
        mock_fill_store.return_value = mock_store
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            
            report = DailyReport(date="2025-06-08")
            
            # Test JSON string output
            json_str = report.to_json()
            data = json.loads(json_str)
            
            assert 'summary' in data
            assert 'breakdown' in data
            assert 'metadata' in data
            assert data['summary']['date'] == "2025-06-08"

    @patch('mech_exo.reporting.daily.FillStore')
    def test_to_json_file(self, mock_fill_store):
        """Test JSON export to file"""
        mock_store = MagicMock()
        mock_store.conn = MagicMock()
        mock_fill_store.return_value = mock_store
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            
            report = DailyReport(date="2025-06-08")
            
            # Test file output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_path = Path(f.name)
            
            try:
                json_str = report.to_json(temp_path)
                
                # Verify file was created and contains valid JSON
                assert temp_path.exists()
                data = json.loads(temp_path.read_text())
                assert data['summary']['date'] == "2025-06-08"
                
            finally:
                if temp_path.exists():
                    temp_path.unlink()

    @patch('mech_exo.reporting.daily.FillStore')
    def test_to_csv(self, mock_fill_store):
        """Test CSV export functionality"""
        mock_store = MagicMock()
        mock_store.conn = MagicMock()
        mock_fill_store.return_value = mock_store
        
        sample_fills = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'quantity': [100, -50],
            'price': [150.0, 120.0]
        })
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = sample_fills
            
            report = DailyReport(date="2025-06-08")
            
            # Test CSV export
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_path = Path(f.name)
            
            try:
                report.to_csv(temp_path)
                
                # Verify file was created
                assert temp_path.exists()
                csv_content = temp_path.read_text()
                assert 'symbol' in csv_content
                assert 'AAPL' in csv_content
                
            finally:
                if temp_path.exists():
                    temp_path.unlink()


def test_generate_daily_report():
    """Test convenience function"""
    with patch('mech_exo.reporting.daily.DailyReport') as mock_report:
        generate_daily_report("2025-06-08")
        mock_report.assert_called_once_with(date="2025-06-08")