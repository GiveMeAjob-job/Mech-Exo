"""
Tests for IB Statement Parser
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime

from mech_exo.reconciliation.ib_statement_parser import (
    IBStatementParser, 
    StatementFormat,
    parse_ib_statement,
    validate_ib_statement
)


class TestIBStatementParser:
    """Test cases for IBStatementParser"""
    
    def setup_method(self):
        """Set up test environment"""
        self.parser = IBStatementParser()
    
    def test_parser_initialization(self):
        """Test parser initialization"""
        assert isinstance(self.parser.supported_formats, list)
        assert StatementFormat.CSV in self.parser.supported_formats
        assert StatementFormat.OFX in self.parser.supported_formats
    
    def test_format_detection_csv(self):
        """Test CSV format detection"""
        # Create temporary CSV file
        csv_content = "Symbol,Date/Time,Quantity,Price,Commission,Amount\nAAPL,2024-12-06,100,150.00,1.00,14999.00"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            detected_format = self.parser._detect_format(temp_path)
            assert detected_format == StatementFormat.CSV
        finally:
            temp_path.unlink()
    
    def test_format_detection_ofx(self):
        """Test OFX format detection"""
        ofx_content = """<?xml version="1.0" encoding="UTF-8"?>
        <OFX>
            <INVSTMTMSGSRSV1>
                <INVSTMTTRNRS>
                    <INVSTMTRS>
                        <INVPOSLIST>
                            <INVPOS>
                                <SECID>AAPL</SECID>
                            </INVPOS>
                        </INVPOSLIST>
                    </INVSTMTRS>
                </INVSTMTTRNRS>
            </INVSTMTMSGSRSV1>
        </OFX>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
            f.write(ofx_content)
            temp_path = Path(f.name)
        
        try:
            detected_format = self.parser._detect_format(temp_path)
            assert detected_format == StatementFormat.OFX
        finally:
            temp_path.unlink()
    
    def test_parse_csv_statement(self):
        """Test parsing CSV statement"""
        csv_content = """Account Information
Account,U1234567,USD

Trades,Header,DataDiscriminator,Asset Category,Currency,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code
Trades,Data,Order,Stocks,USD,AAPL,2024-12-06 09:30:00,100,150.00,150.00,-15000.00,1.00,-15001.00,0,0,O
Trades,Data,Order,Stocks,USD,MSFT,2024-12-06 10:15:00,-50,300.00,300.00,15000.00,1.50,14998.50,0,0,C
Trades,Total,,,,,,,,-15000.00,2.50,-15001.00,0,0,

Portfolio Summary"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            df = self.parser._parse_csv_statement(temp_path)
            
            # Validate results
            assert not df.empty
            assert len(df) == 2  # Two trades
            assert 'symbol' in df.columns
            assert 'qty' in df.columns
            assert 'price' in df.columns
            
            # Check specific data
            assert 'AAPL' in df['symbol'].values
            assert 'MSFT' in df['symbol'].values
            
            # Check data types
            assert pd.api.types.is_numeric_dtype(df['qty'])
            assert pd.api.types.is_numeric_dtype(df['price'])
            
        finally:
            temp_path.unlink()
    
    def test_parse_simple_csv(self):
        """Test parsing simple CSV format"""
        csv_content = """Symbol,Quantity,Price,Commission,Net Cash
AAPL,100,150.00,1.00,-15001.00
TSLA,-25,800.00,2.00,19998.00
MSFT,50,300.00,1.50,-15001.50"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            df = self.parser.parse_statement(str(temp_path))
            
            # Validate results
            assert len(df) == 3
            assert list(df['symbol']) == ['AAPL', 'TSLA', 'MSFT']
            assert list(df['qty']) == [100, -25, 50]
            assert list(df['price']) == [150.00, 800.00, 300.00]
            
        finally:
            temp_path.unlink()
    
    def test_parse_ofx_statement(self):
        """Test parsing OFX statement"""
        ofx_content = """<?xml version="1.0" encoding="UTF-8"?>
        <OFX>
            <INVSTMTMSGSRSV1>
                <INVSTMTTRNRS>
                    <INVSTMTRS>
                        <INVTRANLIST>
                            <BUYSTOCK>
                                <INVTRAN>
                                    <FITID>12345</FITID>
                                    <DTTRADE>20241206</DTTRADE>
                                    <DTSETTLE>20241207</DTSETTLE>
                                </INVTRAN>
                                <SECID>AAPL</SECID>
                                <UNITS>100</UNITS>
                                <UNITPRICE>150.00</UNITPRICE>
                                <COMMISSION>1.00</COMMISSION>
                                <TOTAL>-15001.00</TOTAL>
                            </BUYSTOCK>
                            <SELLSTOCK>
                                <INVTRAN>
                                    <FITID>12346</FITID>
                                    <DTTRADE>20241206</DTTRADE>
                                </INVTRAN>
                                <SECID>MSFT</SECID>
                                <UNITS>-50</UNITS>
                                <UNITPRICE>300.00</UNITPRICE>
                                <COMMISSION>1.50</COMMISSION>
                                <TOTAL>14998.50</TOTAL>
                            </SELLSTOCK>
                        </INVTRANLIST>
                    </INVSTMTRS>
                </INVSTMTTRNRS>
            </INVSTMTMSGSRSV1>
        </OFX>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
            f.write(ofx_content)
            temp_path = Path(f.name)
        
        try:
            df = self.parser._parse_ofx_statement(temp_path)
            
            # Validate results
            assert not df.empty
            assert len(df) == 2
            assert 'symbol' in df.columns
            assert 'qty' in df.columns
            assert 'price' in df.columns
            
            # Check specific data
            symbols = set(df['symbol'])
            assert 'AAPL' in symbols
            assert 'MSFT' in symbols
            
        finally:
            temp_path.unlink()
    
    def test_standardize_dataframe(self):
        """Test DataFrame standardization"""
        # Create test DataFrame with various column names
        raw_data = {
            'Ticker': ['AAPL', 'MSFT'],
            'Shares': [100, -50],
            'Exec_Price': [150.00, 300.00],
            'Fees': [1.00, 1.50],
            'Amount': [-15001.00, 14998.50],
            'Date': ['2024-12-06', '2024-12-06'],
            'Order_ID': ['12345', '12346']
        }
        raw_df = pd.DataFrame(raw_data)
        
        result = self.parser._standardize_dataframe(raw_df, StatementFormat.CSV)
        
        # Check standardized columns
        expected_cols = ['symbol', 'qty', 'price', 'commission', 'net_cash', 'trade_date', 'trade_id', 'currency']
        for col in expected_cols:
            assert col in result.columns
        
        # Check data mapping
        assert list(result['symbol']) == ['AAPL', 'MSFT']
        assert list(result['qty']) == [100, -50]
        assert list(result['price']) == [150.00, 300.00]
    
    def test_clean_dataframe(self):
        """Test DataFrame cleaning"""
        # Create test DataFrame with messy data
        messy_data = {
            'symbol': [' aapl ', 'NASDAQ:MSFT', 'tsla'],
            'qty': ['100', '-50.0', '25'],
            'price': ['150.00', '300', 'invalid'],
            'commission': ['1.00', '', '2.00'],
            'currency': ['USD', None, 'USD']
        }
        messy_df = pd.DataFrame(messy_data)
        
        result = self.parser._clean_dataframe(messy_df)
        
        # Check cleaned data
        assert list(result['symbol']) == ['AAPL', 'MSFT', 'TSLA']
        assert list(result['qty']) == [100, -50.0, 25]
        assert result['currency'].fillna('USD').tolist() == ['USD', 'USD', 'USD']
    
    def test_validate_statement(self):
        """Test statement validation"""
        # Valid statement
        valid_data = {
            'symbol': ['AAPL', 'MSFT'],
            'qty': [100, -50],
            'price': [150.00, 300.00],
            'commission': [1.00, 1.50]
        }
        valid_df = pd.DataFrame(valid_data)
        
        validation = self.parser.validate_statement(valid_df)
        assert validation['is_valid'] is True
        assert validation['row_count'] == 2
        
        # Invalid statement (missing symbol)
        invalid_data = {
            'qty': [100, -50],
            'price': [150.00, 300.00]
        }
        invalid_df = pd.DataFrame(invalid_data)
        
        validation = self.parser.validate_statement(invalid_df)
        assert validation['is_valid'] is False
        assert len(validation['issues']) > 0
    
    def test_empty_statement(self):
        """Test handling of empty statements"""
        empty_content = "Symbol,Quantity,Price\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(empty_content)
            temp_path = Path(f.name)
        
        try:
            df = self.parser.parse_statement(str(temp_path))
            assert df.empty
            
            validation = self.parser.validate_statement(df)
            assert validation['is_valid'] is False
            
        finally:
            temp_path.unlink()
    
    def test_file_not_found(self):
        """Test handling of missing files"""
        with pytest.raises(FileNotFoundError):
            self.parser.parse_statement('/nonexistent/file.csv')
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        csv_content = """Symbol,Quantity,Price
AAPL,100,150.00
MSFT,-50,300.00"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            # Test parse function
            df = parse_ib_statement(str(temp_path))
            assert len(df) == 2
            
            # Test validate function
            validation = validate_ib_statement(df)
            assert 'is_valid' in validation
            assert 'row_count' in validation
            
        finally:
            temp_path.unlink()


class TestStatementFormats:
    """Test different statement format scenarios"""
    
    def test_csv_with_sections(self):
        """Test CSV with multiple sections (typical IB format)"""
        csv_content = """Statement,Data,BrokerageAccount,U1234567,USD,2024-12-06,2024-12-06

Account Information,Header,Field,Value
Account Information,Data,Name,Test Account
Account Information,Data,Account ID,U1234567

Trades,Header,DataDiscriminator,Asset Category,Currency,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code
Trades,Data,Order,Stocks,USD,AAPL,2024-12-06 09:30:00,100,150.00,150.00,-15000.00,1.00,-15001.00,0,0,O
Trades,Data,Order,Stocks,USD,MSFT,2024-12-06 10:15:00,-50,300.00,300.00,15000.00,1.50,14998.50,0,0,C
Trades,Total,,,,,,,-,,-,2.50,-,0,0,

Portfolio Summary,Header,Field,Value"""
        
        parser = IBStatementParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            df = parser.parse_statement(str(temp_path))
            
            assert len(df) == 2
            assert 'AAPL' in df['symbol'].values
            assert 'MSFT' in df['symbol'].values
            
            # Validate commission data is preserved
            assert df[df['symbol'] == 'AAPL']['commission'].iloc[0] == 1.00
            assert df[df['symbol'] == 'MSFT']['commission'].iloc[0] == 1.50
            
        finally:
            temp_path.unlink()
    
    def test_malformed_csv_handling(self):
        """Test handling of malformed CSV data"""
        malformed_csv = """This is not a proper CSV
Some random text
Symbol,Quantity,Price,Commission
AAPL,100,150.00,1.00
Invalid row with wrong number of columns
MSFT,-50,300.00,1.50
Another invalid row"""
        
        parser = IBStatementParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(malformed_csv)
            temp_path = Path(f.name)
        
        try:
            df = parser.parse_statement(str(temp_path))
            
            # Should still extract valid rows
            assert len(df) >= 1  # At least AAPL should be parsed
            if len(df) > 0:
                assert 'AAPL' in df['symbol'].values
            
        finally:
            temp_path.unlink()
    
    def test_different_ofx_structures(self):
        """Test different OFX structures"""
        # Test minimal OFX structure
        minimal_ofx = """<?xml version="1.0"?>
        <OFX>
            <INVTRANLIST>
                <INVTRAN>
                    <FITID>123</FITID>
                    <SYMBOL>AAPL</SYMBOL>
                    <UNITS>100</UNITS>
                    <UNITPRICE>150.00</UNITPRICE>
                </INVTRAN>
            </INVTRANLIST>
        </OFX>"""
        
        parser = IBStatementParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ofx', delete=False) as f:
            f.write(minimal_ofx)
            temp_path = Path(f.name)
        
        try:
            df = parser.parse_statement(str(temp_path))
            
            # Should handle minimal structure
            if not df.empty:
                assert 'symbol' in df.columns
                
        finally:
            temp_path.unlink()