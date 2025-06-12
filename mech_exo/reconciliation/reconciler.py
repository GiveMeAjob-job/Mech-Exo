"""
Trade Reconciliation Engine

Matches internal fills against broker statements and produces diff reports.
Supports tolerance-based matching and generates alerts for discrepancies.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MatchType(Enum):
    """Trade matching types"""
    EXACT_ID = "exact_id"
    FUZZY_MATCH = "fuzzy_match"
    NO_MATCH = "no_match"


class ReconciliationStatus(Enum):
    """Reconciliation status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class TradeMatch:
    """Represents a matched trade pair"""
    internal_trade: Optional[Dict[str, Any]]
    broker_trade: Optional[Dict[str, Any]]
    match_type: MatchType
    match_score: float
    differences: Dict[str, Any]
    
    @property
    def is_matched(self) -> bool:
        return self.match_type != MatchType.NO_MATCH
    
    @property
    def has_differences(self) -> bool:
        return bool(self.differences)


@dataclass
class ReconciliationResult:
    """Results of trade reconciliation"""
    trade_matches: List[TradeMatch]
    unmatched_internal: List[Dict[str, Any]]
    unmatched_broker: List[Dict[str, Any]]
    total_commission_diff: float
    total_net_cash_diff: float
    status: ReconciliationStatus
    summary: Dict[str, Any]
    alerts: List[str]
    
    @property
    def total_diff_bps(self) -> float:
        """Total difference in basis points"""
        if self.summary.get('total_broker_value', 0) == 0:
            return 0.0
        return abs(self.total_net_cash_diff) / abs(self.summary['total_broker_value']) * 10000
    
    @property
    def is_pass(self) -> bool:
        return self.status == ReconciliationStatus.PASS


class TradeReconciler:
    """
    Trade reconciliation engine
    
    Matches internal fill data against broker statements using configurable
    matching criteria and tolerance thresholds.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reconciler with configuration
        
        Args:
            config: Configuration dict with matching parameters
        """
        self.config = config or self._default_config()
        
        # Extract tolerances
        self.price_tolerance = self.config['tolerances']['price_tolerance']
        self.commission_tolerance = self.config['tolerances']['commission_tolerance']
        self.net_cash_tolerance = self.config['tolerances']['net_cash_tolerance']
        self.pass_threshold_bps = self.config['thresholds']['pass_threshold_bps']
        
        logger.info(f"Reconciler initialized with {self.pass_threshold_bps}bp pass threshold")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default reconciliation configuration"""
        return {
            'tolerances': {
                'price_tolerance': 0.01,      # $0.01 price tolerance
                'commission_tolerance': 0.01,  # $0.01 commission tolerance  
                'net_cash_tolerance': 0.05,    # $0.05 net cash tolerance
                'quantity_tolerance': 0        # Exact quantity match required
            },
            'thresholds': {
                'pass_threshold_bps': 5,       # 5 basis points = PASS threshold
                'warning_threshold_bps': 2     # 2 basis points = WARNING threshold
            },
            'matching': {
                'prefer_trade_id': True,       # Prefer trade ID matching
                'fuzzy_match_enabled': True,   # Enable fuzzy matching
                'max_time_diff_hours': 24      # Max time difference for matching
            },
            'alerts': {
                'enabled': True,
                'channels': ['telegram'],
                'critical_threshold_bps': 10
            }
        }
    
    def reconcile(self, 
                  internal_fills: pd.DataFrame, 
                  broker_statement: pd.DataFrame,
                  trade_date: Optional[date] = None) -> ReconciliationResult:
        """
        Perform trade reconciliation
        
        Args:
            internal_fills: DataFrame with internal fill data
            broker_statement: DataFrame with broker statement data
            trade_date: Optional trade date for filtering
            
        Returns:
            ReconciliationResult with matching and difference analysis
        """
        logger.info(f"Starting reconciliation: {len(internal_fills)} internal fills vs "
                   f"{len(broker_statement)} broker trades")
        
        # Validate input data
        self._validate_dataframes(internal_fills, broker_statement)
        
        # Filter by trade date if specified
        if trade_date:
            internal_fills = self._filter_by_date(internal_fills, trade_date, 'internal')
            broker_statement = self._filter_by_date(broker_statement, trade_date, 'broker')
        
        # Standardize data formats
        internal_std = self._standardize_internal_fills(internal_fills)
        broker_std = self._standardize_broker_statement(broker_statement)
        
        # Perform matching
        matches = self._match_trades(internal_std, broker_std)
        
        # Calculate differences
        total_commission_diff, total_net_cash_diff = self._calculate_total_differences(matches)
        
        # Find unmatched trades
        unmatched_internal = self._find_unmatched_internal(internal_std, matches)
        unmatched_broker = self._find_unmatched_broker(broker_std, matches)
        
        # Generate summary
        summary = self._generate_summary(internal_std, broker_std, matches)
        
        # Determine status
        total_diff_bps = abs(total_net_cash_diff) / abs(summary['total_broker_value']) * 10000 if summary['total_broker_value'] != 0 else 0
        status = self._determine_status(total_diff_bps, len(unmatched_internal), len(unmatched_broker))
        
        # Generate alerts
        alerts = self._generate_alerts(total_diff_bps, unmatched_internal, unmatched_broker)
        
        result = ReconciliationResult(
            trade_matches=matches,
            unmatched_internal=unmatched_internal,
            unmatched_broker=unmatched_broker,
            total_commission_diff=total_commission_diff,
            total_net_cash_diff=total_net_cash_diff,
            status=status,
            summary=summary,
            alerts=alerts
        )
        
        logger.info(f"Reconciliation complete: {status.value} ({total_diff_bps:.1f}bp difference)")
        return result
    
    def _validate_dataframes(self, internal_fills: pd.DataFrame, broker_statement: pd.DataFrame):
        """Validate input DataFrames"""
        # Check internal fills format
        required_internal = ['symbol', 'quantity', 'fill_price']
        missing_internal = [col for col in required_internal if col not in internal_fills.columns]
        if missing_internal:
            raise ValueError(f"Internal fills missing columns: {missing_internal}")
        
        # Check broker statement format
        required_broker = ['symbol', 'qty', 'price']
        missing_broker = [col for col in required_broker if col not in broker_statement.columns]
        if missing_broker:
            raise ValueError(f"Broker statement missing columns: {missing_broker}")
    
    def _filter_by_date(self, df: pd.DataFrame, trade_date: date, source: str) -> pd.DataFrame:
        """Filter DataFrame by trade date"""
        date_col = 'fill_time' if source == 'internal' else 'trade_date'
        
        if date_col not in df.columns:
            logger.warning(f"No date column ({date_col}) in {source} data, skipping date filter")
            return df
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Filter by date
        start_time = pd.Timestamp(trade_date)
        end_time = start_time + pd.Timedelta(days=1)
        
        filtered = df[(df[date_col] >= start_time) & (df[date_col] < end_time)]
        logger.info(f"Filtered {source} data: {len(df)} -> {len(filtered)} trades for {trade_date}")
        
        return filtered
    
    def _standardize_internal_fills(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize internal fills format"""
        if df.empty:
            return df
        
        # Create standardized copy
        std = df.copy()
        
        # Ensure required columns
        if 'trade_id' not in std.columns:
            std['trade_id'] = std.get('fill_id', range(len(std)))
        
        if 'commission' not in std.columns:
            std['commission'] = std.get('commission_usd', 0.0)
        
        # Calculate net cash if not present
        if 'net_cash' not in std.columns:
            qty = std.get('quantity', 0)
            price = std.get('fill_price', 0)
            commission = std.get('commission', 0)
            std['net_cash'] = -(qty * price) - commission  # Negative for buys
        
        # Standardize column names
        column_mapping = {
            'quantity': 'qty',
            'fill_price': 'price'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in std.columns and new_col not in std.columns:
                std[new_col] = std[old_col]
        
        return std
    
    def _standardize_broker_statement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize broker statement format"""
        if df.empty:
            return df
        
        # Already standardized by statement parser
        std = df.copy()
        
        # Ensure trade_id exists
        if 'trade_id' not in std.columns or std['trade_id'].isna().all():
            std['trade_id'] = range(len(std))
        
        # Ensure commission exists
        if 'commission' not in std.columns:
            std['commission'] = 0.0
        
        return std
    
    def _match_trades(self, internal: pd.DataFrame, broker: pd.DataFrame) -> List[TradeMatch]:
        """Match internal fills with broker trades"""
        matches = []
        used_internal = set()
        used_broker = set()
        
        # Convert to list of dicts for easier processing
        internal_trades = internal.to_dict('records') if not internal.empty else []
        broker_trades = broker.to_dict('records') if not broker.empty else []
        
        # Phase 1: Exact ID matching
        if self.config['matching']['prefer_trade_id']:
            for i, internal_trade in enumerate(internal_trades):
                if i in used_internal:
                    continue
                
                internal_id = str(internal_trade.get('trade_id', ''))
                if not internal_id or internal_id == 'nan':
                    continue
                
                for j, broker_trade in enumerate(broker_trades):
                    if j in used_broker:
                        continue
                    
                    broker_id = str(broker_trade.get('trade_id', ''))
                    if broker_id == internal_id:
                        match = self._create_trade_match(internal_trade, broker_trade, MatchType.EXACT_ID)
                        matches.append(match)
                        used_internal.add(i)
                        used_broker.add(j)
                        break
        
        # Phase 2: Fuzzy matching
        if self.config['matching']['fuzzy_match_enabled']:
            for i, internal_trade in enumerate(internal_trades):
                if i in used_internal:
                    continue
                
                best_match = None
                best_score = 0
                best_j = -1
                
                for j, broker_trade in enumerate(broker_trades):
                    if j in used_broker:
                        continue
                    
                    score = self._calculate_match_score(internal_trade, broker_trade)
                    if score > 0.8 and score > best_score:  # 80% similarity threshold
                        best_match = broker_trade
                        best_score = score
                        best_j = j
                
                if best_match:
                    match = self._create_trade_match(internal_trade, best_match, MatchType.FUZZY_MATCH)
                    match.match_score = best_score
                    matches.append(match)
                    used_internal.add(i)
                    used_broker.add(best_j)
        
        logger.info(f"Matched {len(matches)} trades ({len(used_internal)} internal, {len(used_broker)} broker)")
        return matches
    
    def _calculate_match_score(self, internal: Dict[str, Any], broker: Dict[str, Any]) -> float:
        """Calculate similarity score between trades"""
        score = 0.0
        total_weight = 0.0
        
        # Symbol match (high weight)
        if internal.get('symbol', '').upper() == broker.get('symbol', '').upper():
            score += 0.4
        total_weight += 0.4
        
        # Quantity match (high weight)
        internal_qty = float(internal.get('qty', 0))
        broker_qty = float(broker.get('qty', 0))
        if abs(internal_qty - broker_qty) <= self.config['tolerances']['quantity_tolerance']:
            score += 0.3
        total_weight += 0.3
        
        # Price match (medium weight)
        internal_price = float(internal.get('price', 0))
        broker_price = float(broker.get('price', 0))
        if internal_price > 0 and broker_price > 0:
            price_diff = abs(internal_price - broker_price)
            if price_diff <= self.price_tolerance:
                score += 0.2
            elif price_diff <= self.price_tolerance * 2:
                score += 0.1  # Partial credit
        total_weight += 0.2
        
        # Time proximity (low weight)
        # TODO: Implement time-based matching
        score += 0.1  # Default time score
        total_weight += 0.1
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _create_trade_match(self, internal: Dict[str, Any], broker: Dict[str, Any], match_type: MatchType) -> TradeMatch:
        """Create TradeMatch with difference analysis"""
        differences = {}
        
        # Price difference
        internal_price = float(internal.get('price', 0))
        broker_price = float(broker.get('price', 0))
        price_diff = internal_price - broker_price
        if abs(price_diff) > self.price_tolerance:
            differences['price'] = {
                'internal': internal_price,
                'broker': broker_price,
                'difference': price_diff
            }
        
        # Commission difference
        internal_commission = float(internal.get('commission', 0))
        broker_commission = float(broker.get('commission', 0))
        commission_diff = internal_commission - broker_commission
        if abs(commission_diff) > self.commission_tolerance:
            differences['commission'] = {
                'internal': internal_commission,
                'broker': broker_commission,
                'difference': commission_diff
            }
        
        # Net cash difference
        internal_net = float(internal.get('net_cash', 0))
        broker_net = float(broker.get('net_cash', 0))
        net_diff = internal_net - broker_net
        if abs(net_diff) > self.net_cash_tolerance:
            differences['net_cash'] = {
                'internal': internal_net,
                'broker': broker_net,
                'difference': net_diff
            }
        
        return TradeMatch(
            internal_trade=internal,
            broker_trade=broker,
            match_type=match_type,
            match_score=1.0 if match_type == MatchType.EXACT_ID else 0.0,
            differences=differences
        )
    
    def _calculate_total_differences(self, matches: List[TradeMatch]) -> Tuple[float, float]:
        """Calculate total commission and net cash differences"""
        total_commission_diff = 0.0
        total_net_cash_diff = 0.0
        
        for match in matches:
            if 'commission' in match.differences:
                total_commission_diff += match.differences['commission']['difference']
            
            if 'net_cash' in match.differences:
                total_net_cash_diff += match.differences['net_cash']['difference']
        
        return total_commission_diff, total_net_cash_diff
    
    def _find_unmatched_internal(self, internal: pd.DataFrame, matches: List[TradeMatch]) -> List[Dict[str, Any]]:
        """Find unmatched internal trades"""
        if internal.empty:
            return []
        
        matched_indices = set()
        for match in matches:
            if match.internal_trade:
                # Find index by trade_id or other unique identifier
                trade_id = match.internal_trade.get('trade_id')
                for i, row in internal.iterrows():
                    if str(row.get('trade_id', '')) == str(trade_id):
                        matched_indices.add(i)
                        break
        
        unmatched = []
        for i, row in internal.iterrows():
            if i not in matched_indices:
                unmatched.append(row.to_dict())
        
        return unmatched
    
    def _find_unmatched_broker(self, broker: pd.DataFrame, matches: List[TradeMatch]) -> List[Dict[str, Any]]:
        """Find unmatched broker trades"""
        if broker.empty:
            return []
        
        matched_indices = set()
        for match in matches:
            if match.broker_trade:
                # Find index by trade_id or other unique identifier
                trade_id = match.broker_trade.get('trade_id')
                for i, row in broker.iterrows():
                    if str(row.get('trade_id', '')) == str(trade_id):
                        matched_indices.add(i)
                        break
        
        unmatched = []
        for i, row in broker.iterrows():
            if i not in matched_indices:
                unmatched.append(row.to_dict())
        
        return unmatched
    
    def _generate_summary(self, internal: pd.DataFrame, broker: pd.DataFrame, matches: List[TradeMatch]) -> Dict[str, Any]:
        """Generate reconciliation summary"""
        summary = {
            'internal_trades': len(internal),
            'broker_trades': len(broker),
            'matched_trades': len(matches),
            'match_rate': len(matches) / max(len(internal), len(broker)) if max(len(internal), len(broker)) > 0 else 0,
            'total_internal_value': internal['net_cash'].sum() if not internal.empty and 'net_cash' in internal.columns else 0,
            'total_broker_value': broker['net_cash'].sum() if not broker.empty and 'net_cash' in broker.columns else 0,
            'total_internal_commission': internal['commission'].sum() if not internal.empty and 'commission' in internal.columns else 0,
            'total_broker_commission': broker['commission'].sum() if not broker.empty and 'commission' in broker.columns else 0,
            'reconciliation_time': datetime.now().isoformat()
        }
        
        return summary
    
    def _determine_status(self, total_diff_bps: float, unmatched_internal: int, unmatched_broker: int) -> ReconciliationStatus:
        """Determine reconciliation status"""
        # Check for unmatched trades
        if unmatched_internal > 0 or unmatched_broker > 0:
            return ReconciliationStatus.FAIL
        
        # Check total difference threshold
        if total_diff_bps <= self.pass_threshold_bps:
            return ReconciliationStatus.PASS
        elif total_diff_bps <= self.config['thresholds']['warning_threshold_bps'] * 2:
            return ReconciliationStatus.WARNING
        else:
            return ReconciliationStatus.FAIL
    
    def _generate_alerts(self, total_diff_bps: float, unmatched_internal: List, unmatched_broker: List) -> List[str]:
        """Generate alerts based on reconciliation results"""
        alerts = []
        
        if total_diff_bps > self.pass_threshold_bps:
            alerts.append(f"Total difference {total_diff_bps:.1f}bp exceeds {self.pass_threshold_bps}bp threshold")
        
        if unmatched_internal:
            alerts.append(f"{len(unmatched_internal)} unmatched internal trade(s)")
        
        if unmatched_broker:
            alerts.append(f"{len(unmatched_broker)} unmatched broker trade(s)")
        
        return alerts


# Convenience functions
def reconcile_trades(internal_fills: pd.DataFrame, 
                    broker_statement: pd.DataFrame,
                    config: Optional[Dict[str, Any]] = None) -> ReconciliationResult:
    """
    Convenience function for trade reconciliation
    
    Args:
        internal_fills: DataFrame with internal fill data
        broker_statement: DataFrame with broker statement data  
        config: Optional reconciliation configuration
        
    Returns:
        ReconciliationResult
    """
    reconciler = TradeReconciler(config)
    return reconciler.reconcile(internal_fills, broker_statement)