"""
Main idea scoring and ranking system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from mech_exo.datasource import DataStorage
from mech_exo.utils import ConfigManager
from .base import BaseScorer, ScoringError
from .factors import FactorFactory

logger = logging.getLogger(__name__)


class IdeaScorer(BaseScorer):
    """Main idea scoring and ranking system"""
    
    def __init__(self, config_path: str = "config/factors.yml", use_ml: bool = False):
        # Load configuration
        config_manager = ConfigManager()
        self.factor_config = config_manager.get_factor_config()
        
        if not self.factor_config:
            raise ValueError(f"Could not load factor configuration from {config_path}")
            
        super().__init__(self.factor_config)
        
        # Initialize data storage
        self.storage = DataStorage()
        
        # Market regime adjustments
        self.market_regime_config = self.factor_config.get('market_regime', {})
        self.sector_adjustments = self.factor_config.get('sector_adjustments', {})
        
        # ML integration settings
        self.use_ml = use_ml
        self.ml_weight = self.factor_config.get('ml_weight', 0.3)
        self.ml_scores = None
        
        logger.info(f"IdeaScorer initialized with ML: {use_ml}, ML weight: {self.ml_weight}")
        
    def _initialize_factors(self):
        """Initialize factor models from configuration"""
        # Combine all factor categories
        all_factors = {}
        
        for category in ['fundamental', 'technical', 'sentiment', 'quality']:
            category_factors = self.factor_config.get(category, {})
            all_factors.update(category_factors)
        
        # Create factor instances
        self.factors = FactorFactory.create_all_factors(all_factors)
        
        if not self.factors:
            raise ScoringError("No factors were successfully initialized")
            
        logger.info(f"Initialized {len(self.factors)} factors")
    
    def score(self, symbols: List[str], ml_scores_file: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Score symbols and return ranked DataFrame"""
        try:
            logger.info(f"Scoring {len(symbols)} symbols")
            
            # Get data for scoring
            data = self._prepare_data(symbols)
            
            if data.empty:
                raise ScoringError("No data available for scoring")
            
            # Calculate factor scores
            factor_scores = self._calculate_factor_scores(data)
            
            # Combine scores
            combined_scores = self._combine_factor_scores(factor_scores)
            
            # Apply adjustments
            adjusted_scores = self._apply_adjustments(combined_scores, data)
            
            # Load and integrate ML scores if enabled
            if self.use_ml:
                ml_scores = self._load_ml_scores(symbols, ml_scores_file)
                if ml_scores is not None:
                    self.ml_scores = ml_scores  # Store for use in ranking
                    adjusted_scores = self._integrate_ml_scores(adjusted_scores, ml_scores, data)
                else:
                    logger.warning("ML scores not available - using traditional scores only")
            
            # Rank and format results
            ranked_results = self._rank_results(adjusted_scores, data)
            
            logger.info(f"Successfully scored {len(ranked_results)} symbols")
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            raise ScoringError(f"Scoring failed: {e}")
    
    def rank_universe(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Rank entire universe or specified symbols"""
        if symbols is None:
            # Get symbols from universe
            universe = self.storage.get_universe(active_only=True)
            if universe.empty:
                raise ScoringError("No symbols in universe")
            symbols = universe['symbol'].tolist()
        
        return self.score(symbols)
    
    def _prepare_data(self, symbols: List[str]) -> pd.DataFrame:
        """Prepare data for scoring"""
        logger.info("Preparing data for scoring")
        
        # Get fundamental data
        fundamental_data = self.storage.get_fundamental_data(symbols, latest_only=True)
        
        # Get OHLC data (last 252 trading days for technical indicators)
        ohlc_data = self.storage.get_ohlc_data(symbols, limit=252 * len(symbols))
        
        # Get news sentiment
        news_data = self.storage.get_news_data(symbols, days_back=30)
        
        if fundamental_data.empty and ohlc_data.empty:
            logger.error("No fundamental or OHLC data available")
            return pd.DataFrame()
        
        # Start with fundamental data as base
        if not fundamental_data.empty:
            base_data = fundamental_data.copy()
        else:
            # Create base from OHLC data if no fundamental data
            base_data = pd.DataFrame({'symbol': symbols})
        
        # Add latest OHLC data for each symbol
        if not ohlc_data.empty:
            latest_ohlc = ohlc_data.groupby('symbol').last().reset_index()
            base_data = base_data.merge(latest_ohlc, on='symbol', how='left')
        
        # Add aggregated news sentiment
        if not news_data.empty:
            news_sentiment = self._aggregate_news_sentiment(news_data)
            base_data = base_data.merge(news_sentiment, on='symbol', how='left')
        
        # Add technical indicators
        if not ohlc_data.empty:
            technical_indicators = self._calculate_technical_indicators(ohlc_data)
            base_data = base_data.merge(technical_indicators, on='symbol', how='left')
        
        logger.info(f"Prepared data for {len(base_data)} symbols with {len(base_data.columns)} features")
        
        return base_data
    
    def _aggregate_news_sentiment(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news sentiment by symbol"""
        sentiment_agg = news_data.groupby('symbol').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).round(4)
        
        sentiment_agg.columns = ['news_sentiment_avg', 'news_sentiment_vol', 'news_count']
        sentiment_agg = sentiment_agg.reset_index()
        
        # Add sentiment category
        sentiment_agg['news_sentiment_category'] = pd.cut(
            sentiment_agg['news_sentiment_avg'],
            bins=[-1, -0.2, 0.2, 1],
            labels=['bearish', 'neutral', 'bullish']
        )
        
        return sentiment_agg
    
    def _calculate_technical_indicators(self, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for scoring"""
        technical_data = []
        
        for symbol in ohlc_data['symbol'].unique():
            symbol_data = ohlc_data[ohlc_data['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) < 20:  # Need minimum data for indicators
                continue
                
            # Calculate additional technical indicators
            tech_indicators = {
                'symbol': symbol,
                'current_price': symbol_data['close'].iloc[-1],
                'sma_20': symbol_data['close'].rolling(20).mean().iloc[-1],
                'sma_50': symbol_data['close'].rolling(50).mean().iloc[-1] if len(symbol_data) >= 50 else np.nan,
                'price_vs_sma20': (symbol_data['close'].iloc[-1] / symbol_data['close'].rolling(20).mean().iloc[-1] - 1) * 100,
                'volume_ratio': symbol_data['volume'].iloc[-5:].mean() / symbol_data['volume'].iloc[-20:].mean() if len(symbol_data) >= 20 else 1,
                'volatility_20d': symbol_data['returns'].iloc[-20:].std() * np.sqrt(252) if 'returns' in symbol_data.columns else np.nan
            }
            
            technical_data.append(tech_indicators)
        
        return pd.DataFrame(technical_data)
    
    def _calculate_factor_scores(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate scores for each factor"""
        factor_scores = {}
        
        for factor_name, factor in self.factors.items():
            try:
                logger.debug(f"Calculating factor: {factor_name}")
                
                # Calculate raw factor values
                raw_values = factor.calculate(data)
                
                if raw_values.empty:
                    logger.warning(f"No values calculated for factor {factor_name}")
                    continue
                
                # Normalize values
                normalized_values = factor.normalize(raw_values.dropna(), method="rank")
                
                # Apply direction preference
                directional_values = factor.apply_direction(normalized_values)
                
                # Store scores
                factor_scores[factor_name] = directional_values
                
                logger.debug(f"Factor {factor_name}: {len(directional_values)} scores calculated")
                
            except Exception as e:
                logger.error(f"Failed to calculate factor {factor_name}: {e}")
                continue
        
        return factor_scores
    
    def _combine_factor_scores(self, factor_scores: Dict[str, pd.Series]) -> pd.Series:
        """Combine individual factor scores into composite score"""
        if not factor_scores:
            raise ScoringError("No factor scores to combine")
        
        # Create DataFrame with all factor scores
        scores_df = pd.DataFrame(factor_scores)
        
        # Calculate weighted composite score
        total_weight = 0
        composite_score = pd.Series(0.0, index=scores_df.index)
        
        for factor_name, scores in factor_scores.items():
            if factor_name in self.factors:
                weight = self.factors[factor_name].weight
                # Normalize weights to percentages
                weight_pct = weight / 100.0
                
                composite_score += scores * weight_pct
                total_weight += weight_pct
        
        # Normalize by total weight
        if total_weight > 0:
            composite_score = composite_score / total_weight
        
        return composite_score
    
    def _apply_adjustments(self, scores: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply market regime and sector adjustments"""
        adjusted_scores = scores.copy()
        
        # Apply sector adjustments
        if 'sector' in data.columns and self.sector_adjustments:
            for idx in adjusted_scores.index:
                if idx < len(data):
                    sector = data.iloc[idx].get('sector')
                    if sector and sector in self.sector_adjustments:
                        multiplier = self.sector_adjustments[sector]
                        adjusted_scores.iloc[idx] *= multiplier
                        logger.debug(f"Applied sector adjustment {multiplier} for {sector}")
        
        # Apply market regime adjustments (simplified - could be enhanced)
        market_multiplier = self.market_regime_config.get('bull_market_multiplier', 1.0)
        adjusted_scores *= market_multiplier
        
        return adjusted_scores
    
    def _load_ml_scores(self, symbols: List[str], ml_scores_file: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load ML scores from file or database"""
        try:
            if ml_scores_file and Path(ml_scores_file).exists():
                # Load from specified file
                logger.info(f"Loading ML scores from {ml_scores_file}")
                ml_scores = pd.read_csv(ml_scores_file)
                
                if 'symbol' not in ml_scores.columns or 'ml_score' not in ml_scores.columns:
                    logger.error("ML scores file must contain 'symbol' and 'ml_score' columns")
                    return None
                
                # Filter to requested symbols
                ml_scores = ml_scores[ml_scores['symbol'].isin(symbols)]
                
                logger.info(f"Loaded ML scores for {len(ml_scores)} symbols")
                return ml_scores
                
            else:
                # Try to load from database (latest ML scores)
                try:
                    query = """
                    SELECT symbol, ml_score, prediction_date 
                    FROM ml_scores 
                    WHERE prediction_date = (SELECT MAX(prediction_date) FROM ml_scores)
                    AND symbol IN ({})
                    """.format(','.join([f"'{s}'" for s in symbols]))
                    
                    ml_scores = self.storage.execute_query(query)
                    
                    if not ml_scores.empty:
                        logger.info(f"Loaded ML scores from database for {len(ml_scores)} symbols")
                        return ml_scores
                    else:
                        logger.warning("No ML scores found in database")
                        
                except Exception as e:
                    logger.warning(f"Failed to load ML scores from database: {e}")
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to load ML scores: {e}")
            return None
    
    def _integrate_ml_scores(self, traditional_scores: pd.Series, 
                           ml_scores: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Integrate ML scores with traditional factor scores"""
        logger.info(f"Integrating ML scores with weight {self.ml_weight}")
        
        # Create traditional ranks (lower rank = better)
        traditional_ranks = traditional_scores.rank(ascending=False, method='min')
        
        # Create ML scores DataFrame aligned with data
        ml_scores_aligned = pd.Series(index=data.index, dtype=float)
        
        for idx, row in data.iterrows():
            symbol = row['symbol']
            ml_row = ml_scores[ml_scores['symbol'] == symbol]
            
            if not ml_row.empty:
                ml_scores_aligned.loc[idx] = ml_row['ml_score'].iloc[0]
            else:
                # Use median ML score for missing symbols
                ml_scores_aligned.loc[idx] = ml_scores['ml_score'].median()
        
        # Convert ML scores to ranks (higher score = better, so ascending=False)
        ml_ranks = ml_scores_aligned.rank(ascending=False, method='min')
        
        # Combine ranks using weighted average
        # final_score = (1 - ml_weight) * traditional_rank + ml_weight * ml_rank
        combined_ranks = (1 - self.ml_weight) * traditional_ranks + self.ml_weight * ml_ranks
        
        # Convert back to scores (lower rank = higher score)
        max_rank = combined_ranks.max()
        final_scores = max_rank - combined_ranks + 1
        
        # Normalize to 0-100 scale
        final_scores = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min()) * 100
        
        logger.info(f"Combined traditional and ML scores successfully")
        
        return final_scores
    
    def _rank_results(self, scores: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
        """Rank results and format output"""
        # Create results DataFrame
        results = data.copy()
        
        # Add scores and ranks
        results['composite_score'] = scores
        results['rank'] = scores.rank(ascending=False, method='min')
        results['percentile'] = scores.rank(pct=True)
        
        # Add ML-specific columns if ML is enabled
        if self.use_ml and self.ml_scores is not None:
            results['ml_rank'] = self.ml_scores.set_index('symbol')['ml_score'].rank(ascending=False, method='min')
            results['final_score'] = scores  # This is already the combined score
            results['ml_weight_used'] = self.ml_weight
        
        # Sort by rank
        results = results.sort_values('rank')
        
        # Add scoring metadata
        results['scoring_date'] = datetime.now()
        results['total_factors'] = len(self.factors)
        results['uses_ml'] = self.use_ml
        
        # Select and order key columns
        key_columns = [
            'rank', 'symbol', 'composite_score', 'percentile',
            'current_price', 'market_cap', 'pe_ratio', 'revenue_growth',
            'sector', 'scoring_date'
        ]
        
        # Add ML columns if applicable
        if self.use_ml:
            key_columns.extend(['ml_rank', 'final_score', 'ml_weight_used', 'uses_ml'])
        
        # Include only existing columns
        available_columns = [col for col in key_columns if col in results.columns]
        
        return results[available_columns].reset_index(drop=True)
    
    def save_ranking(self, ranking: pd.DataFrame, output_path: str = "data/latest_ranking.csv") -> bool:
        """Save ranking to CSV file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            ranking.to_csv(output_file, index=False)
            logger.info(f"Saved ranking to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ranking: {e}")
            return False
    
    def get_top_ideas(self, n: int = 10, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get top N investment ideas"""
        ranking = self.rank_universe(symbols)
        return ranking.head(n)
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'storage'):
            self.storage.close()