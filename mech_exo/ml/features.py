"""
Feature engineering for ML alpha signals.

Creates feature matrices combining price, fundamental, and sentiment data
for training machine learning models to predict forward returns.
"""

import logging
import os
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from mech_exo.datasource import DataStorage

logger = logging.getLogger(__name__)

# Suppress pandas performance warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class FeatureBuilder:
    """
    Builds feature matrices for ML alpha signal training.
    
    Combines multiple data sources:
    - Price data (OHLC, returns, volatility)
    - Fundamental data (ratios, growth metrics)  
    - Sentiment data (news scores, analyst revisions)
    
    Output format: "wide" Parquet files per date for efficient loading.
    """
    
    def __init__(self, storage: Optional[DataStorage] = None, output_dir: str = "data/features"):
        """
        Initialize feature builder.
        
        Args:
            storage: DataStorage instance for data access
            output_dir: Directory to save feature files
        """
        self.storage = storage or DataStorage()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature engineering parameters
        self.price_lags = [1, 5, 20]  # Return lookback periods in days
        self.volatility_windows = [5, 20]  # Volatility calculation windows
        self.z_score_clip = 5.0  # Clip extreme z-scores to ±5
        
    def build_features(self, start_date: str, end_date: str, symbols: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Build feature matrix for date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            symbols: List of symbols to process (None = all available)
            
        Returns:
            Dict mapping date strings to Parquet file paths
        """
        logger.info(f"Building features from {start_date} to {end_date}")
        
        # Load all required data upfront
        data_dict = self._load_data(start_date, end_date, symbols)
        
        # Generate feature matrices by date
        feature_files = {}
        date_range = pd.bdate_range(start_date, end_date)
        
        for business_date in date_range:
            date_str = business_date.strftime('%Y-%m-%d')
            
            try:
                # Build features for this date
                features_df = self._build_features_for_date(data_dict, business_date)
                
                if features_df is not None and len(features_df) > 0:
                    # Save to CSV (fallback if parquet not available)
                    output_file = self.output_dir / f"features_{date_str}.csv"
                    try:
                        # Try parquet first
                        parquet_file = self.output_dir / f"features_{date_str}.parquet"
                        features_df.to_parquet(parquet_file, index=False, compression='snappy')
                        output_file = parquet_file
                    except ImportError:
                        # Fall back to CSV
                        features_df.to_csv(output_file, index=False)
                    
                    feature_files[date_str] = output_file
                    logger.debug(f"Saved {len(features_df)} feature rows for {date_str}")
                    
            except Exception as e:
                logger.warning(f"Failed to build features for {date_str}: {e}")
                continue
                
        logger.info(f"Built features for {len(feature_files)} dates")
        return feature_files
        
    def _load_data(self, start_date: str, end_date: str, symbols: Optional[List[str]]) -> Dict[str, pd.DataFrame]:
        """Load all required data sources."""
        logger.info("Loading data sources...")
        
        # Extend date range for lag calculation
        extended_start = (pd.to_datetime(start_date) - BDay(30)).strftime('%Y-%m-%d')
        
        data_dict = {}
        
        try:
            # Load OHLC data using storage methods
            data_dict['ohlc'] = self.storage.get_ohlc_data(
                symbols=symbols,
                start_date=extended_start,
                end_date=end_date
            )
            logger.info(f"Loaded {len(data_dict['ohlc'])} OHLC records")
            
            # Load fundamental data
            data_dict['fundamentals'] = self.storage.get_fundamental_data(
                symbols=symbols,
                latest_only=False
            )
            # Filter by date range and handle column naming
            if len(data_dict['fundamentals']) > 0:
                # Check for date column (could be 'date' or 'fetch_date')
                date_col = 'date' if 'date' in data_dict['fundamentals'].columns else 'fetch_date'
                
                if date_col in data_dict['fundamentals'].columns:
                    data_dict['fundamentals'][date_col] = pd.to_datetime(data_dict['fundamentals'][date_col])
                    data_dict['fundamentals'] = data_dict['fundamentals'][
                        (data_dict['fundamentals'][date_col] >= extended_start) &
                        (data_dict['fundamentals'][date_col] <= end_date)
                    ].copy()
                    
                    # Rename to 'date' for consistency
                    if date_col != 'date':
                        data_dict['fundamentals'] = data_dict['fundamentals'].rename(columns={date_col: 'date'})
            
            logger.info(f"Loaded {len(data_dict['fundamentals'])} fundamental records")
            
            # Load news/sentiment data (limited by days_back parameter)
            data_dict['news'] = self.storage.get_news_data(
                symbols=symbols,
                days_back=30  # Load last 30 days of news
            )
            # Rename published_at to date for consistency
            if len(data_dict['news']) > 0 and 'published_at' in data_dict['news'].columns:
                data_dict['news'] = data_dict['news'].rename(columns={'published_at': 'date'})
                # Add article_count if not present
                if 'article_count' not in data_dict['news'].columns:
                    data_dict['news']['article_count'] = 1
                    
                # Filter by date range manually
                data_dict['news']['date'] = pd.to_datetime(data_dict['news']['date'])
                data_dict['news'] = data_dict['news'][
                    (data_dict['news']['date'] >= extended_start) &
                    (data_dict['news']['date'] <= end_date)
                ].copy()
            
            logger.info(f"Loaded {len(data_dict['news'])} news records")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
        return data_dict
        
    def _build_features_for_date(self, data_dict: Dict[str, pd.DataFrame], target_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Build feature matrix for a specific date."""
        date_str = target_date.strftime('%Y-%m-%d')
        
        # Get OHLC data up to target date
        ohlc_df = data_dict['ohlc'].copy()
        ohlc_df['date'] = pd.to_datetime(ohlc_df['date'])
        ohlc_subset = ohlc_df[ohlc_df['date'] <= target_date].copy()
        
        if len(ohlc_subset) == 0:
            return None
            
        # Build price features
        price_features = self._build_price_features(ohlc_subset, target_date)
        
        # Build fundamental features
        fund_features = self._build_fundamental_features(data_dict['fundamentals'], target_date)
        
        # Build sentiment features
        sentiment_features = self._build_sentiment_features(data_dict['news'], target_date)
        
        # Merge all features
        features_df = price_features
        
        if fund_features is not None:
            features_df = features_df.merge(fund_features, on='symbol', how='left')
            
        if sentiment_features is not None:
            features_df = features_df.merge(sentiment_features, on='symbol', how='left')
            
        # Add metadata
        features_df['feature_date'] = date_str
        features_df['feature_count'] = len(features_df.columns) - 2  # Exclude symbol and feature_date
        
        # Clean and validate
        features_df = self._clean_features(features_df)
        
        return features_df
        
    def _build_price_features(self, ohlc_df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
        """Build price-based features for all symbols."""
        features_list = []
        
        for symbol in ohlc_df['symbol'].unique():
            symbol_data = ohlc_df[ohlc_df['symbol'] == symbol].sort_values('date')
            
            # Get latest available data point
            latest_data = symbol_data[symbol_data['date'] <= target_date]
            if len(latest_data) == 0:
                continue
                
            latest_idx = len(latest_data) - 1
            feature_row = {'symbol': symbol}
            
            # Current price features
            if latest_idx >= 0:
                current = latest_data.iloc[latest_idx]
                feature_row.update({
                    'price': current['close'],
                    'volume': current['volume'],
                    'atr_20': current.get('atr_20', np.nan),
                })
                
            # Lagged returns
            for lag in self.price_lags:
                if latest_idx >= lag:
                    past_price = latest_data.iloc[latest_idx - lag]['close']
                    current_price = latest_data.iloc[latest_idx]['close']
                    returns = (current_price / past_price - 1) if past_price > 0 else 0
                    feature_row[f'return_{lag}d'] = returns
                else:
                    feature_row[f'return_{lag}d'] = np.nan
                    
            # Volatility features
            for window in self.volatility_windows:
                if len(latest_data) >= window:
                    recent_data = latest_data.tail(window)
                    recent_data = recent_data.copy()
                    recent_data['daily_return'] = recent_data['close'].pct_change()
                    vol = recent_data['daily_return'].std() * np.sqrt(252)  # Annualized
                    feature_row[f'volatility_{window}d'] = vol
                else:
                    feature_row[f'volatility_{window}d'] = np.nan
                    
            # Technical indicators
            if len(latest_data) >= 14:
                # RSI approximation using recent price changes
                recent_14 = latest_data.tail(14).copy()
                recent_14['price_change'] = recent_14['close'].diff()
                gains = recent_14['price_change'].where(recent_14['price_change'] > 0, 0).mean()
                losses = -recent_14['price_change'].where(recent_14['price_change'] < 0, 0).mean()
                rs = gains / losses if losses != 0 else 100
                rsi = 100 - (100 / (1 + rs))
                feature_row['rsi_14'] = rsi
            else:
                feature_row['rsi_14'] = np.nan
                
            features_list.append(feature_row)
            
        return pd.DataFrame(features_list)
        
    def _build_fundamental_features(self, fund_df: pd.DataFrame, target_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Build fundamental features."""
        if len(fund_df) == 0:
            return None
            
        fund_df = fund_df.copy()
        fund_df['date'] = pd.to_datetime(fund_df['date'])
        
        # Get most recent fundamental data for each symbol
        latest_fund = fund_df[fund_df['date'] <= target_date].copy()
        if len(latest_fund) == 0:
            return None
            
        # Get latest record per symbol
        latest_fund = latest_fund.sort_values('date').groupby('symbol').tail(1)
        
        feature_cols = [
            'pe_ratio', 'return_on_equity', 'revenue_growth', 'earnings_growth',
            'debt_to_equity', 'current_ratio', 'gross_margin', 'profit_margin'
        ]
        
        # Select available columns
        available_cols = ['symbol'] + [col for col in feature_cols if col in latest_fund.columns]
        fund_features = latest_fund[available_cols].copy()
        
        # Add fundamental ratios prefix
        for col in fund_features.columns:
            if col != 'symbol':
                fund_features = fund_features.rename(columns={col: f'fund_{col}'})
                
        return fund_features
        
    def _build_sentiment_features(self, news_df: pd.DataFrame, target_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Build sentiment features."""
        if len(news_df) == 0:
            return None
            
        news_df = news_df.copy()
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # Get recent sentiment data (last 5 days)
        start_sentiment = target_date - timedelta(days=5)
        recent_news = news_df[
            (news_df['date'] >= start_sentiment) & 
            (news_df['date'] <= target_date)
        ].copy()
        
        if len(recent_news) == 0:
            return None
            
        # Aggregate sentiment by symbol
        sentiment_agg = recent_news.groupby('symbol').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'article_count': 'sum'
        }).reset_index()
        
        # Flatten column names
        sentiment_agg.columns = [
            'symbol', 'sent_score_mean', 'sent_score_std', 'sent_score_count', 'total_articles'
        ]
        
        # Fill NaN std with 0 (single article cases)
        sentiment_agg['sent_score_std'] = sentiment_agg['sent_score_std'].fillna(0)
        
        return sentiment_agg
        
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate feature matrix."""
        # Clip extreme z-scores
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['feature_count']]
        
        for col in numeric_cols:
            if features_df[col].notna().sum() > 1:  # Need at least 2 non-null values
                mean_val = features_df[col].mean()
                std_val = features_df[col].std()
                
                if std_val > 0:
                    z_scores = (features_df[col] - mean_val) / std_val
                    features_df[col] = features_df[col].where(
                        z_scores.abs() <= self.z_score_clip,
                        mean_val + np.sign(z_scores) * self.z_score_clip * std_val
                    )
                    
        # Forward fill missing values within reasonable limits
        for col in numeric_cols:
            features_df[col] = features_df[col].ffill(limit=5)
            
        # Fill remaining NaN with 0 for tree models
        features_df = features_df.fillna(0)
        
        return features_df
        
    def get_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """Get summary statistics for feature matrix."""
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'total_symbols': len(features_df),
            'total_features': len(numeric_cols),
            'feature_names': list(numeric_cols),
            'missing_data_pct': (features_df[numeric_cols].isnull().sum() / len(features_df) * 100).to_dict(),
            'feature_date': features_df['feature_date'].iloc[0] if 'feature_date' in features_df.columns else None
        }
        
        return summary
        
    def to_training_matrix(self, start_date: str, end_date: str, 
                          symbols: Optional[List[str]] = None,
                          forward_days: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training matrix with features and forward return labels.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: Optional list of symbols
            forward_days: Days ahead for forward return calculation
            
        Returns:
            (X, y): Feature matrix and binary labels (forward return > 0%)
        """
        logger.info(f"Creating training matrix for {start_date} to {end_date}")
        
        # Build features first
        feature_files = self.build_features(start_date, end_date, symbols)
        
        if not feature_files:
            raise ValueError("No feature files generated")
        
        # Load all features
        all_features = []
        
        for file_path in feature_files.values():
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_parquet(file_path)
            all_features.append(df)
        
        features_df = pd.concat(all_features, ignore_index=True)
        
        # Prepare training data with forward returns
        X, y = self._prepare_training_labels(features_df, forward_days)
        
        logger.info(f"Training matrix: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Labels: {y.sum()} positive, {len(y) - y.sum()} negative")
        
        return X, y
        
    def _prepare_training_labels(self, features_df: pd.DataFrame, 
                               forward_days: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels with no look-ahead bias."""
        
        # Sort by symbol and date
        df = features_df.copy()
        df['feature_date'] = pd.to_datetime(df['feature_date'])
        df = df.sort_values(['symbol', 'feature_date']).reset_index(drop=True)
        
        # Calculate forward returns for each symbol
        forward_returns = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            prices = symbol_data['price'].values
            
            # Calculate forward returns
            symbol_forward_returns = np.full(len(prices), np.nan)
            
            for i in range(len(prices) - forward_days):
                current_price = prices[i]
                future_price = prices[i + forward_days]
                
                if current_price > 0 and future_price > 0:
                    symbol_forward_returns[i] = (future_price / current_price - 1)
            
            forward_returns.extend(symbol_forward_returns)
        
        df['forward_return'] = forward_returns
        
        # Remove rows with missing data
        df = df.dropna().reset_index(drop=True)
        
        # Create binary labels (positive forward return)
        y = (df['forward_return'] > 0.0).astype(int)
        
        # Feature matrix (exclude metadata and labels)
        feature_cols = [col for col in df.columns 
                       if col not in ['symbol', 'feature_date', 'feature_count', 'forward_return']]
        X = df[feature_cols].copy()
        
        return X, pd.Series(y, index=X.index)


def build_features_cli(start_date: str, end_date: str, symbols: Optional[List[str]] = None, 
                      output_dir: str = "data/features") -> None:
    """
    CLI function for building features.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbols: Optional list of symbols
        output_dir: Output directory for feature files
    """
    builder = FeatureBuilder(output_dir=output_dir)
    
    logger.info(f"🏗️ Building ML features...")
    logger.info(f"   Date range: {start_date} to {end_date}")
    logger.info(f"   Output dir: {output_dir}")
    
    if symbols:
        logger.info(f"   Symbols: {len(symbols)} specified")
    else:
        logger.info("   Symbols: All available")
        
    feature_files = builder.build_features(start_date, end_date, symbols)
    
    if feature_files:
        # Load a sample to show summary
        sample_file = list(feature_files.values())[0]
        sample_df = pd.read_parquet(sample_file)
        summary = builder.get_feature_summary(sample_df)
        
        logger.info(f"✅ Feature building completed:")
        logger.info(f"   Files created: {len(feature_files)}")
        logger.info(f"   Sample symbols: {summary['total_symbols']}")
        logger.info(f"   Feature count: {summary['total_features']}")
        logger.info(f"   Feature names: {summary['feature_names'][:5]}..." if len(summary['feature_names']) > 5 else f"   Feature names: {summary['feature_names']}")
        
    else:
        logger.warning("❌ No feature files created - check data availability")


if __name__ == "__main__":
    # Test with sample data
    import sys
    
    if len(sys.argv) >= 3:
        start = sys.argv[1]
        end = sys.argv[2]
        build_features_cli(start, end)
    else:
        # Test with recent dates
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        build_features_cli(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))