"""
Interactive Brokers Statement Parser

Parses IB T+1 statements in both CSV and OFX formats.
Maps trade data to standardized DataFrame format for reconciliation.
"""

import logging
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

logger = logging.getLogger(__name__)


class StatementFormat(Enum):
    """Statement format types"""
    CSV = "csv"
    OFX = "ofx"
    XML = "xml"


class IBStatementParser:
    """
    Parser for Interactive Brokers statements
    
    Supports both CSV and OFX/XML formats commonly provided by IB.
    Standardizes trade data for reconciliation against internal fills.
    """
    
    def __init__(self):
        """Initialize the statement parser"""
        self.supported_formats = [StatementFormat.CSV, StatementFormat.OFX, StatementFormat.XML]
    
    def parse_statement(self, file_path: str, format_hint: Optional[StatementFormat] = None) -> pd.DataFrame:
        """
        Parse statement file and return standardized DataFrame
        
        Args:
            file_path: Path to statement file
            format_hint: Optional format hint, will auto-detect if None
            
        Returns:
            DataFrame with columns: symbol, qty, price, commission, net_cash, trade_date, trade_id, currency
            
        Raises:
            ValueError: If file format not supported or parsing fails
            FileNotFoundError: If statement file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Statement file not found: {file_path}")
        
        # Auto-detect format if not provided
        if format_hint is None:
            format_hint = self._detect_format(file_path)
        
        logger.info(f"Parsing IB statement: {file_path.name} (format: {format_hint.value})")
        
        if format_hint == StatementFormat.CSV:
            return self._parse_csv_statement(file_path)
        elif format_hint in [StatementFormat.OFX, StatementFormat.XML]:
            return self._parse_ofx_statement(file_path)
        else:
            raise ValueError(f"Unsupported statement format: {format_hint}")
    
    def _detect_format(self, file_path: Path) -> StatementFormat:
        """
        Auto-detect statement format from file extension and content
        
        Args:
            file_path: Path to statement file
            
        Returns:
            Detected StatementFormat
        """
        # Check file extension first
        ext = file_path.suffix.lower()
        if ext == '.csv':
            return StatementFormat.CSV
        elif ext in ['.ofx', '.xml']:
            # Check content to distinguish OFX from generic XML
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1KB
                    if 'OFX' in content.upper() or 'OFXHEADER' in content.upper():
                        return StatementFormat.OFX
                    else:
                        return StatementFormat.XML
            except Exception:
                return StatementFormat.XML
        
        # Fallback: try to detect from content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if ',' in first_line and ('Symbol' in first_line or 'Date' in first_line):
                    return StatementFormat.CSV
                elif first_line.startswith('<?xml') or 'OFX' in first_line.upper():
                    return StatementFormat.OFX
        except Exception:
            pass
        
        # Default to CSV
        logger.warning(f"Could not detect format for {file_path.name}, defaulting to CSV")
        return StatementFormat.CSV
    
    def _parse_csv_statement(self, file_path: Path) -> pd.DataFrame:
        """
        Parse IB CSV statement format
        
        IB CSV typically has sections like:
        - Account Information
        - Trades
        - Dividends
        - etc.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Standardized DataFrame
        """
        try:
            # Read entire CSV to find trades section
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find trades section
            trades_start = None
            trades_end = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                if 'Trades' in line and ('Header' in line or 'Data' in line):
                    if trades_start is None:
                        trades_start = i + 1  # Start after header
                elif trades_start is not None and line.startswith('Total'):
                    trades_end = i
                    break
                elif trades_start is not None and (line == '' or line.startswith(',')):
                    trades_end = i
                    break
            
            if trades_start is None:
                # Try alternative approach - look for common column headers
                for i, line in enumerate(lines):
                    if any(col in line.upper() for col in ['SYMBOL', 'QUANTITY', 'PRICE', 'COMMISSION']):
                        trades_start = i
                        break
            
            if trades_start is None:
                raise ValueError("Could not find trades section in CSV statement")
            
            # Extract trades data
            if trades_end is None:
                trades_lines = lines[trades_start:]
            else:
                trades_lines = lines[trades_start:trades_end]
            
            # Parse trades section
            trades_data = []
            header_line = None
            
            for line in trades_lines:
                line = line.strip()
                if not line or line.startswith('Total'):
                    continue
                
                parts = [part.strip('"').strip() for part in line.split(',')]
                
                # Skip obvious header or separator lines
                if 'Symbol' in line or 'Currency' in line or len(parts) < 5:
                    if header_line is None and 'Symbol' in line:
                        header_line = parts
                    continue
                
                # Skip empty or malformed lines
                if len(parts) < 5 or not any(parts):
                    continue
                
                trades_data.append(parts)
            
            if not trades_data:
                logger.warning("No trade data found in CSV statement")
                return self._create_empty_dataframe()
            
            # Create DataFrame
            if header_line and len(header_line) == len(trades_data[0]):
                df = pd.DataFrame(trades_data, columns=header_line)
            else:
                # Use standard IB column names
                expected_cols = ['Symbol', 'Date/Time', 'Quantity', 'Price', 'Commission', 'Amount', 'Currency']
                num_cols = len(trades_data[0])
                if num_cols >= len(expected_cols):
                    df = pd.DataFrame(trades_data, columns=expected_cols[:num_cols])
                else:
                    # Minimal columns
                    cols = expected_cols[:num_cols] if num_cols <= len(expected_cols) else [f'Col{i}' for i in range(num_cols)]
                    df = pd.DataFrame(trades_data, columns=cols)
            
            # Standardize DataFrame
            return self._standardize_dataframe(df, StatementFormat.CSV)
            
        except Exception as e:
            logger.error(f"Failed to parse CSV statement {file_path}: {e}")
            raise ValueError(f"CSV parsing failed: {e}")
    
    def _parse_ofx_statement(self, file_path: Path) -> pd.DataFrame:
        """
        Parse IB OFX/XML statement format
        
        OFX format contains structured XML with transaction data.
        
        Args:
            file_path: Path to OFX/XML file
            
        Returns:
            Standardized DataFrame
        """
        try:
            # Read and parse XML content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Handle OFX format quirks
            if 'OFXHEADER' in content:
                # OFX files often have a header section before XML
                xml_start = content.find('<')
                if xml_start > 0:
                    content = content[xml_start:]
            
            # Parse XML
            root = ET.fromstring(content)
            
            # Find transaction records
            trades_data = []
            
            # Look for common OFX transaction elements
            transaction_tags = [
                './/BUYMF', './/SELLMF',  # Mutual fund transactions
                './/BUYSTOCK', './/SELLSTOCK',  # Stock transactions
                './/BUYOPT', './/SELLOPT',  # Options transactions
                './/INVTRAN', './/POSTRAN'  # Generic investment transactions
            ]
            
            for tag in transaction_tags:
                transactions = root.findall(tag)
                for txn in transactions:
                    trade_data = self._extract_ofx_transaction(txn)
                    if trade_data:
                        trades_data.append(trade_data)
            
            # If no specific transaction types found, look for generic patterns
            if not trades_data:
                # Look for any elements that might contain trade data
                for elem in root.iter():
                    if any(keyword in elem.tag.upper() for keyword in ['TXN', 'TRADE', 'TRANS']):
                        trade_data = self._extract_ofx_transaction(elem)
                        if trade_data:
                            trades_data.append(trade_data)
            
            if not trades_data:
                logger.warning("No trade data found in OFX statement")
                return self._create_empty_dataframe()
            
            # Create DataFrame
            df = pd.DataFrame(trades_data)
            
            # Standardize DataFrame
            return self._standardize_dataframe(df, StatementFormat.OFX)
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error in {file_path}: {e}")
            raise ValueError(f"OFX/XML parsing failed: {e}")
        except Exception as e:
            logger.error(f"Failed to parse OFX statement {file_path}: {e}")
            raise ValueError(f"OFX parsing failed: {e}")
    
    def _extract_ofx_transaction(self, txn_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """
        Extract transaction data from OFX XML element
        
        Args:
            txn_elem: XML element containing transaction data
            
        Returns:
            Dictionary with transaction data or None if invalid
        """
        try:
            trade_data = {}
            
            # Map OFX field names to standard names
            field_mapping = {
                'SECID': 'symbol',
                'TICKER': 'symbol', 
                'SYMBOL': 'symbol',
                'UNITS': 'qty',
                'QUANTITY': 'qty',
                'UNITPRICE': 'price',
                'PRICE': 'price',
                'COMMISSION': 'commission',
                'FEES': 'commission',
                'TOTAL': 'net_cash',
                'AMOUNT': 'net_cash',
                'DTTRADE': 'trade_date',
                'DTSETTLE': 'settle_date',
                'FITID': 'trade_id',
                'UNIQUEID': 'trade_id',
                'CURDEF': 'currency'
            }
            
            # Extract data from XML element and children
            for child in txn_elem.iter():
                tag = child.tag.upper()
                if tag in field_mapping and child.text:
                    field_name = field_mapping[tag]
                    trade_data[field_name] = child.text.strip()
            
            # Validate required fields
            if not trade_data.get('symbol'):
                return None
            
            # Set defaults
            trade_data.setdefault('currency', 'USD')
            trade_data.setdefault('commission', '0.0')
            
            return trade_data
            
        except Exception as e:
            logger.debug(f"Could not extract transaction from OFX element: {e}")
            return None
    
    def _standardize_dataframe(self, df: pd.DataFrame, source_format: StatementFormat) -> pd.DataFrame:
        """
        Standardize DataFrame to common format for reconciliation
        
        Args:
            df: Raw DataFrame from parsing
            source_format: Original format of the data
            
        Returns:
            Standardized DataFrame with consistent columns and types
        """
        if df.empty:
            return self._create_empty_dataframe()
        
        # Create standardized DataFrame
        result = pd.DataFrame()
        
        # Map columns to standard names (case-insensitive)
        column_mapping = {
            'symbol': ['symbol', 'ticker', 'instrument', 'secid'],
            'qty': ['qty', 'quantity', 'units', 'shares'],
            'price': ['price', 'unitprice', 'exec_price', 'fill_price'],
            'commission': ['commission', 'fees', 'comm'],
            'net_cash': ['net_cash', 'amount', 'total', 'net_amount'],
            'trade_date': ['trade_date', 'date', 'datetime', 'date/time', 'dttrade'],
            'trade_id': ['trade_id', 'exec_id', 'fitid', 'uniqueid', 'order_id'],
            'currency': ['currency', 'curdef', 'ccy']
        }
        
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        for std_col, possible_names in column_mapping.items():
            mapped_col = None
            for name in possible_names:
                if name.lower() in df_columns_lower:
                    mapped_col = df_columns_lower[name.lower()]
                    break
            
            if mapped_col:
                result[std_col] = df[mapped_col]
            else:
                # Set default values
                if std_col == 'currency':
                    result[std_col] = 'USD'
                elif std_col == 'commission':
                    result[std_col] = 0.0
                elif std_col == 'trade_id':
                    # Generate synthetic trade ID if missing
                    result[std_col] = range(len(df))
                else:
                    result[std_col] = None
        
        # Clean and convert data types
        result = self._clean_dataframe(result)
        
        logger.info(f"Standardized {len(result)} trades from {source_format.value} format")
        return result
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert data types in standardized DataFrame
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame with proper data types
        """
        if df.empty:
            return df
        
        # Clean symbol
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
            # Remove common prefixes/suffixes
            df['symbol'] = df['symbol'].str.replace(r'^\w+:', '', regex=True)  # Remove exchange prefixes
        
        # Convert numeric columns
        numeric_columns = ['qty', 'price', 'commission', 'net_cash']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert trade_date
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        
        # Clean trade_id
        if 'trade_id' in df.columns:
            df['trade_id'] = df['trade_id'].astype(str).str.strip()
        
        # Fill missing currency
        if 'currency' in df.columns:
            df['currency'] = df['currency'].fillna('USD')
        
        # Remove rows with critical missing data
        initial_count = len(df)
        df = df.dropna(subset=['symbol'])
        
        if len(df) < initial_count:
            logger.warning(f"Dropped {initial_count - len(df)} rows with missing symbols")
        
        return df
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """
        Create empty DataFrame with standard columns
        
        Returns:
            Empty DataFrame with reconciliation columns
        """
        return pd.DataFrame(columns=[
            'symbol', 'qty', 'price', 'commission', 'net_cash', 
            'trade_date', 'trade_id', 'currency'
        ])
    
    def validate_statement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate parsed statement data
        
        Args:
            df: Parsed statement DataFrame
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'is_valid': True,
            'row_count': len(df),
            'issues': [],
            'warnings': []
        }
        
        if df.empty:
            validation['is_valid'] = False
            validation['issues'].append('Statement contains no trade data')
            return validation
        
        # Check for required columns
        required_cols = ['symbol', 'qty', 'price']
        missing_cols = [col for col in required_cols if col not in df.columns or df[col].isna().all()]
        if missing_cols:
            validation['is_valid'] = False
            validation['issues'].append(f'Missing required columns: {missing_cols}')
        
        # Check for null values in critical columns
        for col in ['symbol', 'qty']:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    validation['warnings'].append(f'{null_count} null values in {col}')
        
        # Check for reasonable price values
        if 'price' in df.columns and not df['price'].isna().all():
            negative_prices = (df['price'] < 0).sum()
            if negative_prices > 0:
                validation['warnings'].append(f'{negative_prices} negative prices found')
            
            zero_prices = (df['price'] == 0).sum()
            if zero_prices > 0:
                validation['warnings'].append(f'{zero_prices} zero prices found')
        
        # Check commission data
        if 'commission' in df.columns:
            null_commission = df['commission'].isna().sum()
            if null_commission > 0:
                validation['warnings'].append(f'{null_commission} trades missing commission data')
        
        return validation


# Convenience functions
def parse_ib_statement(file_path: str, format_hint: Optional[StatementFormat] = None) -> pd.DataFrame:
    """
    Convenience function to parse IB statement
    
    Args:
        file_path: Path to statement file
        format_hint: Optional format hint
        
    Returns:
        Standardized DataFrame
    """
    parser = IBStatementParser()
    return parser.parse_statement(file_path, format_hint)


def validate_ib_statement(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to validate statement DataFrame
    
    Args:
        df: Statement DataFrame to validate
        
    Returns:
        Validation results
    """
    parser = IBStatementParser()
    return parser.validate_statement(df)