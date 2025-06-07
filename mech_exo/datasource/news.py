"""
News and sentiment data fetcher
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import time
import logging
from datetime import datetime, timedelta
from .base import BaseDataFetcher, DataValidationError

logger = logging.getLogger(__name__)


class NewsScraper(BaseDataFetcher):
    """Fetches news and sentiment data"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.finnhub_api_key = config.get("finnhub", {}).get("api_key")
        self.news_api_key = config.get("news_api", {}).get("api_key")
        self.finnhub_base_url = config.get("finnhub", {}).get("base_url", "https://finnhub.io/api/v1")
        self.news_api_base_url = config.get("news_api", {}).get("base_url", "https://newsapi.org/v2")
        self.max_retries = config.get("max_retries", 3)
        
    def fetch(self, symbols: List[str], days_back: int = 7, **kwargs) -> pd.DataFrame:
        """Fetch news data for given symbols"""
        all_news = []
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching news for {symbol}")
                
                # Get news from Finnhub if available
                if self.finnhub_api_key:
                    finnhub_news = self._fetch_finnhub_news(symbol, days_back)
                    all_news.extend(finnhub_news)
                
                # Get news from NewsAPI if available
                if self.news_api_key:
                    newsapi_news = self._fetch_newsapi_news(symbol, days_back)
                    all_news.extend(newsapi_news)
                    
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to fetch news for {symbol}: {e}")
                continue
                
        if not all_news:
            logger.warning("No news data fetched")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_news)
        
        # Remove duplicates based on URL
        if not df.empty and 'url' in df.columns:
            df = df.drop_duplicates(subset=['url'])
            
        if not df.empty and not self.validate_data(df):
            raise DataValidationError("News data validation failed")
            
        return df
    
    def _fetch_finnhub_news(self, symbol: str, days_back: int) -> List[Dict[str, Any]]:
        """Fetch news from Finnhub"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f"{self.finnhub_base_url}/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            news_data = response.json()
            
            processed_news = []
            for article in news_data:
                processed_article = {
                    'symbol': symbol,
                    'headline': article.get('headline', ''),
                    'summary': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', 'finnhub'),
                    'published_at': pd.to_datetime(article.get('datetime'), unit='s'),
                    'image_url': article.get('image', ''),
                    'category': article.get('category', ''),
                    'sentiment_score': self._calculate_simple_sentiment(
                        article.get('headline', '') + ' ' + article.get('summary', '')
                    ),
                    'fetch_date': pd.Timestamp.now(),
                    'data_source': 'finnhub'
                }
                processed_news.append(processed_article)
                
            return processed_news
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub news for {symbol}: {e}")
            return []
    
    def _fetch_newsapi_news(self, symbol: str, days_back: int) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search for company news
            url = f"{self.news_api_base_url}/everything"
            params = {
                'q': f'"{symbol}" OR "{self._get_company_name(symbol)}"',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.news_api_key,
                'pageSize': 100
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            processed_news = []
            for article in articles:
                processed_article = {
                    'symbol': symbol,
                    'headline': article.get('title', ''),
                    'summary': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', 'newsapi'),
                    'published_at': pd.to_datetime(article.get('publishedAt')),
                    'image_url': article.get('urlToImage', ''),
                    'category': 'general',
                    'sentiment_score': self._calculate_simple_sentiment(
                        article.get('title', '') + ' ' + article.get('description', '')
                    ),
                    'fetch_date': pd.Timestamp.now(),
                    'data_source': 'newsapi'
                }
                processed_news.append(processed_article)
                
            return processed_news
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI data for {symbol}: {e}")
            return []
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol (simplified mapping)"""
        # This could be enhanced with a proper mapping database
        company_mapping = {
            'AAPL': 'Apple Inc',
            'GOOGL': 'Alphabet Google',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'META': 'Meta Facebook',
            'NVDA': 'NVIDIA',
            'FXI': 'China ETF',
            'SPY': 'S&P 500 ETF',
            'QQQ': 'Nasdaq ETF'
        }
        return company_mapping.get(symbol, symbol)
    
    def _calculate_simple_sentiment(self, text: str) -> float:
        """Calculate simple rule-based sentiment score"""
        if not text:
            return 0.0
            
        text = text.lower()
        
        # Simple positive/negative word lists
        positive_words = [
            'gain', 'gains', 'up', 'rise', 'rises', 'bullish', 'positive', 'good', 'great',
            'excellent', 'strong', 'growth', 'profit', 'profits', 'beat', 'beats', 'exceeds',
            'outperform', 'success', 'successful', 'boost', 'surge', 'rally', 'upgrade'
        ]
        
        negative_words = [
            'loss', 'losses', 'down', 'fall', 'falls', 'bearish', 'negative', 'bad', 'poor',
            'weak', 'decline', 'miss', 'misses', 'underperform', 'failure', 'fail', 'drop',
            'crash', 'plunge', 'downgrade', 'concern', 'concerns', 'risk', 'risks'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Normalize to -1 to 1 scale
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
            
        sentiment = (positive_count - negative_count) / max(total_words * 0.1, 1)
        return max(-1.0, min(1.0, sentiment))
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate news data quality"""
        if data.empty:
            logger.warning("News data is empty")
            return True  # Empty news is okay
            
        required_columns = ['symbol', 'headline', 'published_at', 'sentiment_score']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check sentiment scores are in valid range
        if 'sentiment_score' in data.columns:
            invalid_sentiment = (data['sentiment_score'] < -1) | (data['sentiment_score'] > 1)
            if invalid_sentiment.any():
                logger.error("Found sentiment scores outside [-1, 1] range")
                return False
                
        # Check published dates are reasonable (not in future, not too old)
        if 'published_at' in data.columns:
            now = pd.Timestamp.now()
            future_dates = data['published_at'] > now
            very_old_dates = data['published_at'] < (now - pd.Timedelta(days=365))
            
            if future_dates.any():
                logger.warning("Found future publication dates")
                
            if very_old_dates.any():
                logger.warning("Found very old publication dates")
                
        logger.info(f"News data validation passed for {len(data)} articles")
        return True
    
    def fetch_sentiment_summary(self, symbols: List[str], days_back: int = 7) -> pd.DataFrame:
        """Fetch and summarize sentiment for symbols"""
        news_data = self.fetch(symbols, days_back)
        
        if news_data.empty:
            return pd.DataFrame()
            
        # Calculate sentiment summary by symbol
        sentiment_summary = news_data.groupby('symbol').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'published_at': 'max'
        }).round(4)
        
        # Flatten column names
        sentiment_summary.columns = ['avg_sentiment', 'sentiment_volatility', 'news_count', 'latest_news']
        sentiment_summary = sentiment_summary.reset_index()
        
        # Add sentiment categories
        sentiment_summary['sentiment_category'] = pd.cut(
            sentiment_summary['avg_sentiment'],
            bins=[-1, -0.2, 0.2, 1],
            labels=['bearish', 'neutral', 'bullish']
        )
        
        return sentiment_summary