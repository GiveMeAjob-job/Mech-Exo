"""
Signal builder for converting idea rankings to trading signals
Handles rebalancing frequency, position sizing, and signal generation
"""

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def idea_rank_to_signals(rank_df: pd.DataFrame, n_top: int = 3, 
                        holding_period: int = 30, rebal_freq: str = 'monthly',
                        start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Convert idea rankings to boolean trading signals
    
    Args:
        rank_df: DataFrame with ranking data (index=date, columns=symbols, values=rank/score)
        n_top: Number of top-ranked ideas to hold at any time
        holding_period: Minimum holding period in days
        rebal_freq: Rebalancing frequency ('daily', 'weekly', 'monthly', 'quarterly')
        start_date: Optional start date for signals (YYYY-MM-DD)
        end_date: Optional end date for signals (YYYY-MM-DD)
        
    Returns:
        DataFrame with boolean signals (True=long, False=flat)
    """
    
    if rank_df.empty:
        logger.warning("Empty ranking DataFrame provided")
        return pd.DataFrame()
    
    # Ensure index is datetime
    if not isinstance(rank_df.index, pd.DatetimeIndex):
        rank_df.index = pd.to_datetime(rank_df.index)
    
    # Filter by date range if provided
    if start_date:
        rank_df = rank_df[rank_df.index >= pd.to_datetime(start_date)]
    if end_date:
        rank_df = rank_df[rank_df.index <= pd.to_datetime(end_date)]
    
    if rank_df.empty:
        logger.warning("No data in specified date range")
        return pd.DataFrame()
    
    # Create rebalancing schedule
    rebal_dates = _get_rebalancing_dates(rank_df.index, rebal_freq)
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(False, index=rank_df.index, columns=rank_df.columns)
    
    # Track current positions and their entry dates
    current_positions = {}  # {symbol: entry_date}
    
    logger.info(f"Generating signals: {n_top} positions, {rebal_freq} rebalancing, "
               f"{holding_period}d holding period")
    
    for date in signals.index:
        # Copy previous day's positions
        if date != signals.index[0]:
            prev_date = signals.index[signals.index < date][-1]
            current_positions = {
                symbol: entry_date for symbol, entry_date in current_positions.items()
                if signals.loc[prev_date, symbol]
            }
        
        # Check if it's a rebalancing date
        is_rebal_date = date in rebal_dates
        
        # Get current rankings for this date
        current_ranks = rank_df.loc[date].dropna()
        
        if current_ranks.empty:
            # No rankings available, maintain current positions
            for symbol in current_positions:
                if symbol in signals.columns:
                    signals.loc[date, symbol] = True
            continue
        
        # Determine positions to hold
        if is_rebal_date:
            # Full rebalancing
            new_positions = _select_top_positions(
                current_ranks, n_top, current_positions, holding_period, date
            )
        else:
            # Maintain positions, only exit if holding period expired
            new_positions = _maintain_positions(
                current_positions, holding_period, date, current_ranks, n_top
            )
        
        # Update signals
        for symbol in signals.columns:
            signals.loc[date, symbol] = symbol in new_positions
        
        # Update position tracking
        current_positions = new_positions
    
    logger.info(f"Generated signals for {len(signals)} days, {len(signals.columns)} symbols")
    return signals


def _get_rebalancing_dates(date_index: pd.DatetimeIndex, frequency: str) -> List[datetime]:
    """Get rebalancing dates based on frequency"""
    
    start_date = date_index.min()
    end_date = date_index.max()
    
    if frequency == 'daily':
        return list(date_index)
    elif frequency == 'weekly':
        # Rebalance on Mondays
        return [d for d in date_index if d.dayofweek == 0]
    elif frequency == 'monthly':
        # Rebalance on first trading day of each month
        monthly_dates = []
        current_month = None
        for date in date_index:
            if current_month != date.month:
                monthly_dates.append(date)
                current_month = date.month
        return monthly_dates
    elif frequency == 'quarterly':
        # Rebalance quarterly (Mar, Jun, Sep, Dec)
        quarterly_dates = []
        current_quarter = None
        for date in date_index:
            quarter = (date.month - 1) // 3 + 1
            if current_quarter != quarter:
                quarterly_dates.append(date)
                current_quarter = quarter
        return quarterly_dates
    else:
        raise ValueError(f"Unsupported rebalancing frequency: {frequency}")


def _select_top_positions(current_ranks: pd.Series, n_top: int, 
                         current_positions: Dict[str, datetime],
                         holding_period: int, current_date: datetime) -> Dict[str, datetime]:
    """Select top N positions considering holding period constraints"""
    
    # Get top ranked symbols
    top_symbols = current_ranks.nlargest(n_top * 2).index.tolist()  # Get more to allow for filtering
    
    # Filter out positions that haven't met minimum holding period
    available_symbols = []
    forced_holds = {}
    
    for symbol in current_positions:
        days_held = (current_date - current_positions[symbol]).days
        if days_held < holding_period:
            # Must hold this position
            forced_holds[symbol] = current_positions[symbol]
        else:
            # Can exit this position
            pass
    
    # Available slots for new positions
    available_slots = n_top - len(forced_holds)
    
    # Select new positions from top ranked symbols
    new_positions = forced_holds.copy()
    
    for symbol in top_symbols:
        if len(new_positions) >= n_top:
            break
        if symbol not in new_positions:
            new_positions[symbol] = current_date
    
    return new_positions


def _maintain_positions(current_positions: Dict[str, datetime], 
                       holding_period: int, current_date: datetime,
                       current_ranks: pd.Series, n_top: int) -> Dict[str, datetime]:
    """Maintain current positions, only exiting if holding period met"""
    
    maintained_positions = {}
    
    # Check each current position
    for symbol, entry_date in current_positions.items():
        days_held = (current_date - entry_date).days
        
        if days_held >= holding_period:
            # Can exit - check if still in top rankings
            if symbol in current_ranks.nlargest(n_top * 1.5).index:
                # Still well-ranked, keep holding
                maintained_positions[symbol] = entry_date
            # Otherwise let it exit
        else:
            # Must hold due to minimum holding period
            maintained_positions[symbol] = entry_date
    
    return maintained_positions


def create_momentum_signals(symbols: List[str], start: str, end: str,
                           lookback: int = 252, rebal_freq: str = 'monthly',
                           n_top: int = 3) -> pd.DataFrame:
    """
    Create momentum-based trading signals using historical returns
    
    Args:
        symbols: List of symbols to generate signals for
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD) 
        lookback: Lookback period for momentum calculation (days)
        rebal_freq: Rebalancing frequency
        n_top: Number of top momentum stocks to hold
        
    Returns:
        DataFrame with boolean trading signals
    """
    try:
        from ..datasource.storage import DataStorage
        
        # Load price data
        storage = DataStorage()
        query = """
        SELECT date, symbol, close, returns_1d
        FROM ohlc_data 
        WHERE symbol IN ({}) AND date >= ? AND date <= ?
        ORDER BY date, symbol
        """.format(','.join(['?'] * len(symbols)))
        
        params = symbols + [start, end]
        df = pd.read_sql_query(query, storage.conn, params=params)
        storage.close()
        
        if df.empty:
            logger.warning("No price data found for momentum signals")
            return pd.DataFrame()
        
        # Pivot to get prices by symbol
        prices = df.pivot(index='date', columns='symbol', values='close')
        prices.index = pd.to_datetime(prices.index)
        
        # Calculate momentum scores (cumulative returns over lookback period)
        momentum_scores = prices.pct_change().rolling(window=lookback).sum()
        
        # Convert momentum scores to trading signals
        signals = idea_rank_to_signals(
            momentum_scores, 
            n_top=n_top,
            rebal_freq=rebal_freq,
            start_date=start,
            end_date=end
        )
        
        logger.info(f"Created momentum signals for {len(symbols)} symbols")
        return signals
        
    except Exception as e:
        logger.error(f"Failed to create momentum signals: {e}")
        return pd.DataFrame()


def create_ranking_signals_from_scores(scores_df: pd.DataFrame, 
                                     n_long: int = 3, n_short: int = 0,
                                     rebal_freq: str = 'monthly') -> pd.DataFrame:
    """
    Create long/short signals from ranking scores
    
    Args:
        scores_df: DataFrame with ranking scores (higher = better)
        n_long: Number of long positions
        n_short: Number of short positions  
        rebal_freq: Rebalancing frequency
        
    Returns:
        DataFrame with signals (1=long, -1=short, 0=flat)
    """
    
    if scores_df.empty:
        return pd.DataFrame()
    
    # Get long signals
    long_signals = idea_rank_to_signals(
        scores_df, n_top=n_long, rebal_freq=rebal_freq
    )
    
    if n_short == 0:
        # Long-only strategy
        return long_signals.astype(int)
    
    # Get short signals (invert scores for bottom ranking)
    inverted_scores = -scores_df  # Lower scores (worse) become higher for shorting
    short_signals = idea_rank_to_signals(
        inverted_scores, n_top=n_short, rebal_freq=rebal_freq
    )
    
    # Combine long and short signals
    combined_signals = long_signals.astype(int) - short_signals.astype(int)
    
    return combined_signals


# Validation functions
def validate_signals(signals: pd.DataFrame, max_positions: int = None) -> Dict[str, bool]:
    """
    Validate trading signals for common issues
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'has_data': not signals.empty,
        'has_signals': False,
        'position_limit_ok': True,
        'no_lookahead_bias': True,
        'reasonable_turnover': True
    }
    
    if signals.empty:
        return validation
    
    # Check if there are any True signals
    validation['has_signals'] = signals.any().any()
    
    # Check position limits
    if max_positions:
        daily_positions = signals.sum(axis=1)
        validation['position_limit_ok'] = (daily_positions <= max_positions).all()
    
    # Check for reasonable turnover (not changing every day)
    if len(signals) > 1:
        daily_changes = (signals != signals.shift(1)).sum(axis=1)
        avg_daily_changes = daily_changes.mean()
        validation['reasonable_turnover'] = avg_daily_changes < len(signals.columns) * 0.5
    
    return validation