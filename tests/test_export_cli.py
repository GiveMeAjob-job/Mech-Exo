"""
Unit tests for export CLI functionality
"""

import pytest
import tempfile
import os
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from mech_exo.cli.export import DataExporter


class TestDataExporter:
    """Test cases for DataExporter class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Use temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter()
        self.exporter.exports_dir = Path(self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_parse_date_range_keywords(self):
        """Test parsing of date range keywords"""
        today = date.today()
        
        # Test last7d
        start, end = self.exporter.parse_date_range("last7d")
        assert end == today
        assert start == today - timedelta(days=7)
        
        # Test last30d
        start, end = self.exporter.parse_date_range("last30d")
        assert end == today
        assert start == today - timedelta(days=30)
        
        # Test ytd
        start, end = self.exporter.parse_date_range("ytd")
        assert end == today
        assert start == date(today.year, 1, 1)
    
    def test_parse_date_range_explicit(self):
        """Test parsing of explicit date ranges"""
        start, end = self.exporter.parse_date_range("2024-01-01:2024-12-31")
        
        assert start == date(2024, 1, 1)
        assert end == date(2024, 12, 31)
    
    def test_parse_date_range_invalid(self):
        """Test invalid date range handling"""
        # Missing colon
        with pytest.raises(ValueError, match="Date range must be in format"):
            self.exporter.parse_date_range("2024-01-01")
        
        # Invalid date format
        with pytest.raises(ValueError, match="Invalid date format"):
            self.exporter.parse_date_range("invalid:2024-01-01")
        
        # Start after end
        with pytest.raises(ValueError, match="Start date must be before"):
            self.exporter.parse_date_range("2024-12-31:2024-01-01")
    
    def test_get_table_data_unsupported_table(self):
        """Test error handling for unsupported table"""
        with pytest.raises(ValueError, match="Unsupported table"):
            self.exporter.get_table_data("invalid_table", date(2024, 1, 1), date(2024, 1, 31))
    
    def test_get_fills_data_empty(self):
        """Test fills data retrieval when no data exists"""
        with patch.object(self.exporter.fill_store, 'get_fills_df') as mock_get_fills:
            mock_get_fills.return_value = pd.DataFrame()
            
            result = self.exporter._get_fills_data(date(2024, 1, 1), date(2024, 1, 31))
            
            assert result.empty
            mock_get_fills.assert_called_once()
    
    def test_get_fills_data_with_data(self):
        """Test fills data retrieval with sample data"""
        # Create sample fills data
        sample_fills = pd.DataFrame({
            'filled_at': pd.to_datetime(['2024-01-15 10:00:00', '2024-01-16 11:00:00']),
            'symbol': ['AAPL', 'MSFT'],
            'quantity': [100, -50],
            'price': [150.0, 300.0],
            'commission': [1.0, 1.5],
            'fees': [0.0, 0.0]
        })
        
        with patch.object(self.exporter.fill_store, 'get_fills_df') as mock_get_fills:
            mock_get_fills.return_value = sample_fills
            
            result = self.exporter._get_fills_data(date(2024, 1, 1), date(2024, 1, 31))
            
            assert len(result) == 2
            assert 'gross_value' in result.columns
            assert 'net_value' in result.columns
            
            # Check calculated fields
            assert result.iloc[0]['gross_value'] == 15000.0  # 100 * 150
            assert result.iloc[0]['net_value'] == 14999.0    # 15000 - 1.0 commission
    
    def test_get_positions_data(self):
        """Test positions data calculation from fills"""
        # Create sample fills that result in positions
        sample_fills = pd.DataFrame({
            'filled_at': pd.to_datetime(['2024-01-15 10:00:00', '2024-01-16 11:00:00', '2024-01-17 12:00:00']),
            'symbol': ['AAPL', 'AAPL', 'MSFT'],
            'quantity': [100, 50, -25],  # Net: AAPL +150, MSFT -25
            'price': [150.0, 155.0, 300.0],
            'commission': [1.0, 1.5, 0.5],
            'fees': [0.0, 0.0, 0.0]
        })
        
        with patch.object(self.exporter.fill_store, 'get_fills_df') as mock_get_fills:
            mock_get_fills.return_value = sample_fills
            
            result = self.exporter._get_positions_data(date(2024, 1, 1), date(2024, 1, 31))
            
            assert len(result) == 2  # Two symbols with non-zero positions
            
            # Check AAPL position
            aapl_pos = result[result['symbol'] == 'AAPL'].iloc[0]
            assert aapl_pos['quantity'] == 150  # 100 + 50
            assert aapl_pos['avg_price'] == (100*150 + 50*155)/(100+50)  # Weighted average
            assert aapl_pos['trades_count'] == 2
            
            # Check MSFT position (short)
            msft_pos = result[result['symbol'] == 'MSFT'].iloc[0]
            assert msft_pos['quantity'] == -25
    
    def test_export_data_csv_format(self):
        """Test CSV export functionality"""
        # Create sample data
        sample_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'quantity': [100, 50],
            'price': [150.0, 300.0]
        })
        
        with patch.object(self.exporter, 'get_table_data') as mock_get_data:
            mock_get_data.return_value = sample_data
            
            result = self.exporter.export_data(
                'fills', date(2024, 1, 1), date(2024, 1, 31), 
                format='csv', gzip_compress=False
            )
            
            assert result['success'] == True
            assert result['rows'] == 2
            assert result['file_path'].endswith('.csv')
            assert os.path.exists(result['file_path'])
            
            # Verify file content
            exported_df = pd.read_csv(result['file_path'])
            assert len(exported_df) == 2
            assert list(exported_df.columns) == ['symbol', 'quantity', 'price']
    
    def test_export_data_parquet_format(self):
        """Test Parquet export functionality"""
        # Create sample data
        sample_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'quantity': [100, 50],
            'price': [150.0, 300.0]
        })
        
        with patch.object(self.exporter, 'get_table_data') as mock_get_data:
            mock_get_data.return_value = sample_data
            
            result = self.exporter.export_data(
                'fills', date(2024, 1, 1), date(2024, 1, 31), 
                format='parquet', gzip_compress=True
            )
            
            assert result['success'] == True
            assert result['rows'] == 2
            assert result['file_path'].endswith('.parquet.gz')
            assert os.path.exists(result['file_path'])
    
    def test_export_data_empty_dataset(self):
        """Test export with empty dataset"""
        with patch.object(self.exporter, 'get_table_data') as mock_get_data:
            mock_get_data.return_value = pd.DataFrame()
            
            result = self.exporter.export_data(
                'fills', date(2024, 1, 1), date(2024, 1, 31)
            )
            
            assert result['success'] == False
            assert result['rows'] == 0
            assert "No data found" in result['message']
    
    def test_export_data_invalid_format(self):
        """Test export with invalid format"""
        with pytest.raises(ValueError, match="Unsupported format"):
            self.exporter.export_data(
                'fills', date(2024, 1, 1), date(2024, 1, 31), 
                format='invalid'
            )
    
    def test_export_data_invalid_table(self):
        """Test export with invalid table"""
        with pytest.raises(ValueError, match="Unsupported table"):
            self.exporter.export_data(
                'invalid', date(2024, 1, 1), date(2024, 1, 31)
            )
    
    def test_filename_generation(self):
        """Test filename generation logic"""
        sample_data = pd.DataFrame({'col1': [1, 2]})
        
        with patch.object(self.exporter, 'get_table_data') as mock_get_data:
            mock_get_data.return_value = sample_data
            
            # Test CSV without compression
            result = self.exporter.export_data(
                'fills', date(2024, 1, 15), date(2024, 1, 31), 
                format='csv', gzip_compress=False
            )
            assert 'fills_20240115_20240131.csv' in result['file_path']
            
            # Test Parquet with compression
            result = self.exporter.export_data(
                'positions', date(2024, 1, 15), date(2024, 1, 31), 
                format='parquet', gzip_compress=True
            )
            assert 'positions_20240115_20240131.parquet.gz' in result['file_path']


class TestDateRangeParsing:
    """Test date range parsing edge cases"""
    
    def test_all_keyword_shortcuts(self):
        """Test all supported keyword shortcuts"""
        exporter = DataExporter()
        today = date.today()
        
        keywords = {
            'last7d': 7,
            'last30d': 30,
            'last90d': 90,
            'last365d': 365
        }
        
        for keyword, days in keywords.items():
            start, end = exporter.parse_date_range(keyword)
            assert end == today
            assert start == today - timedelta(days=days)
    
    def test_case_insensitive_keywords(self):
        """Test that keywords are case insensitive"""
        exporter = DataExporter()
        
        start1, end1 = exporter.parse_date_range("LAST7D")
        start2, end2 = exporter.parse_date_range("last7d")
        
        assert start1 == start2
        assert end1 == end2


def test_integration_export_workflow():
    """Integration test for complete export workflow"""
    exporter = DataExporter()
    
    # Test with mock data to ensure the full workflow works
    with patch.object(exporter.fill_store, 'get_fills_df') as mock_fills:
        mock_fills.return_value = pd.DataFrame({
            'filled_at': pd.to_datetime(['2024-01-15']),
            'symbol': ['TEST'],
            'quantity': [100],
            'price': [50.0],
            'commission': [1.0],
            'fees': [0.0]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter.exports_dir = Path(temp_dir)
            
            result = exporter.export_data(
                'fills', date(2024, 1, 1), date(2024, 1, 31),
                format='csv'
            )
            
            assert result['success'] == True
            assert result['rows'] == 1
            assert os.path.exists(result['file_path'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])