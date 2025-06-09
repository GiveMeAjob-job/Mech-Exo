"""
Backtesting engine for historical strategy validation

This module provides comprehensive backtesting capabilities using vectorbt,
including walk-forward analysis, realistic fees/slippage, and tear-sheet reports.
"""

from .core import Backtester, BacktestResults, create_simple_signals
from .signal_builder import (
    idea_rank_to_signals, 
    create_momentum_signals,
    create_ranking_signals_from_scores,
    validate_signals
)

__all__ = [
    'Backtester',
    'BacktestResults', 
    'create_simple_signals',
    'idea_rank_to_signals',
    'create_momentum_signals', 
    'create_ranking_signals_from_scores',
    'validate_signals'
]