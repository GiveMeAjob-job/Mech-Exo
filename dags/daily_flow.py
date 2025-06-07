"""
Complete daily flow: Data Pipeline + Execution Engine
Flow: get_signals → size_positions → risk_precheck → route_orders → wait_fills → snapshot
"""

import asyncio
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mech_exo.datasource import DataStorage
from mech_exo.scoring import IdeaScorer
from mech_exo.sizing import PositionSizer, SizingMethod
from mech_exo.risk import RiskChecker, Portfolio, Position
from mech_exo.execution.broker_adapter import create_broker_adapter
from mech_exo.execution.order_router import OrderRouter
from mech_exo.execution.fill_store import FillStore
from mech_exo.execution.models import create_market_order, create_limit_order, OrderStatus
from mech_exo.utils import ConfigManager

# Import data pipeline components
from .data_pipeline import (
    load_config, 
    daily_data_pipeline,
    get_universe_symbols,
    fetch_ohlc_data,
    fetch_fundamental_data, 
    fetch_news_data,
    store_ohlc_data,
    store_fundamental_data,
    store_news_data,
    run_data_quality_checks
)


@task(retries=2, retry_delay_seconds=30)
def generate_trading_signals(config: Dict[str, Any]) -> pd.DataFrame:
    """Generate trading signals using IdeaScorer"""
    logger = get_run_logger()
    
    try:
        scorer = IdeaScorer()
        
        # Score the universe
        ranking = scorer.rank_universe()
        
        if ranking.empty:
            logger.warning("No ranking results - scoring may have failed")
            return pd.DataFrame()
        
        # Filter for investable signals (top quartile, minimum score threshold)
        min_score = config.get('trading', {}).get('min_signal_score', 0.6)
        max_positions = config.get('trading', {}).get('max_positions', 10)
        
        # Get top signals
        signals = ranking[
            (ranking['composite_score'] >= min_score) & 
            (ranking['percentile'] >= 0.75)
        ].head(max_positions).copy()
        
        # Add signal metadata
        signals['signal_timestamp'] = datetime.now()
        signals['signal_strength'] = signals['composite_score']
        
        logger.info(f"Generated {len(signals)} trading signals")
        logger.info(f"Top signal: {signals.iloc[0]['symbol']} (score: {signals.iloc[0]['composite_score']:.3f})")
        
        scorer.close()
        return signals
        
    except Exception as e:
        logger.error(f"Failed to generate trading signals: {e}")
        if 'scorer' in locals():
            scorer.close()
        raise


@task(retries=2, retry_delay_seconds=30)
def size_positions(signals: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Calculate position sizes for signals"""
    logger = get_run_logger()
    
    if signals.empty:
        logger.info("No signals to size")
        return pd.DataFrame()
    
    try:
        nav = config.get('trading', {}).get('nav', 100000)
        sizing_method = config.get('trading', {}).get('sizing_method', 'atr_based')
        
        sizer = PositionSizer(nav)
        sized_positions = []
        
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            current_price = signal.get('current_price', 100.0)  # Fallback price
            signal_strength = signal['signal_strength']
            
            # Determine sizing method
            method = SizingMethod.ATR_BASED if sizing_method == 'atr_based' else SizingMethod.FIXED_PERCENT
            
            # Calculate size
            shares = sizer.calculate_size(
                symbol=symbol,
                price=current_price,
                method=method,
                signal_strength=signal_strength
            )
            
            if shares > 0:
                sized_positions.append({
                    'symbol': symbol,
                    'signal_score': signal['composite_score'],
                    'signal_strength': signal_strength,
                    'current_price': current_price,
                    'target_shares': shares,
                    'target_value': shares * current_price,
                    'sizing_method': sizing_method,
                    'sizing_timestamp': datetime.now()
                })
        
        sized_df = pd.DataFrame(sized_positions)
        
        if not sized_df.empty:
            total_value = sized_df['target_value'].sum()
            logger.info(f"Sized {len(sized_df)} positions, total value: ${total_value:,.0f}")
        
        sizer.close()
        return sized_df
        
    except Exception as e:
        logger.error(f"Failed to size positions: {e}")
        if 'sizer' in locals():
            sizer.close()
        raise


@task(retries=2, retry_delay_seconds=30)
def precheck_risk(sized_positions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-trade risk checks on proposed positions"""
    logger = get_run_logger()
    
    if sized_positions.empty:
        logger.info("No positions to risk check")
        return {'status': 'ok', 'approved_positions': pd.DataFrame(), 'violations': []}
    
    try:
        nav = config.get('trading', {}).get('nav', 100000)
        portfolio = Portfolio(nav)
        
        # Load existing positions (if any)
        # In production, this would load from broker or position tracking system
        
        risk_checker = RiskChecker(portfolio)
        
        approved_positions = []
        violations = []
        
        for _, position in sized_positions.iterrows():
            symbol = position['symbol']
            shares = position['target_shares']
            price = position['current_price']
            
            # Perform pre-trade risk check
            risk_analysis = risk_checker.check_new_position(
                symbol=symbol,
                shares=shares,
                price=price,
                sector='Unknown'  # Would be looked up in production
            )
            
            recommendation = risk_analysis['pre_trade_analysis']['recommendation']
            
            if recommendation == 'APPROVE':
                approved_positions.append(position.to_dict())
            elif recommendation == 'APPROVE_WITH_CAUTION':
                approved_positions.append(position.to_dict())
                violations.extend(risk_analysis['pre_trade_analysis'].get('warnings', []))
            else:  # REJECT
                violations.extend(risk_analysis['pre_trade_analysis'].get('violations', []))
                logger.warning(f"Position {symbol} rejected by risk check: {risk_analysis['pre_trade_analysis']['violations']}")
        
        approved_df = pd.DataFrame(approved_positions)
        
        risk_result = {
            'status': 'warning' if violations else 'ok',
            'approved_positions': approved_df,
            'violations': violations,
            'positions_approved': len(approved_positions),
            'positions_rejected': len(sized_positions) - len(approved_positions),
            'risk_check_timestamp': datetime.now()
        }
        
        logger.info(f"Risk check: {len(approved_positions)}/{len(sized_positions)} positions approved")
        if violations:
            logger.warning(f"Risk violations: {violations}")
        
        risk_checker.close()
        return risk_result
        
    except Exception as e:
        logger.error(f"Failed to perform risk checks: {e}")
        if 'risk_checker' in locals():
            risk_checker.close()
        raise


@task(retries=2, retry_delay_seconds=30)
async def route_orders(approved_positions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Route approved positions as orders to broker"""
    logger = get_run_logger()
    
    if approved_positions.empty:
        logger.info("No approved positions to route")
        return {'status': 'no_orders', 'orders_submitted': 0, 'orders_rejected': 0}
    
    try:
        # Get trading mode from environment or config
        trading_mode = os.getenv('EXO_MODE', config.get('trading', {}).get('mode', 'stub'))
        
        # Setup broker adapter
        broker_config = config.get('broker', {})
        if trading_mode.lower() == 'paper':
            broker_config['port'] = 4002  # Paper trading port
        elif trading_mode.lower() == 'live':
            broker_config['port'] = 4001  # Live trading port
        else:
            trading_mode = 'stub'  # Default to stub for safety
        
        broker = create_broker_adapter(trading_mode, broker_config)
        await broker.connect()
        
        # Setup portfolio and risk checker for OrderRouter
        nav = config.get('trading', {}).get('nav', 100000)
        portfolio = Portfolio(nav)
        risk_checker = RiskChecker(portfolio)
        
        # Setup OrderRouter
        router_config = config.get('order_router', {})
        router = OrderRouter(broker, risk_checker, router_config)
        
        # Track results
        orders_submitted = 0
        orders_rejected = 0
        routing_results = []
        
        # Route each position as an order
        for _, position in approved_positions.iterrows():
            symbol = position['symbol']
            shares = int(position['target_shares'])
            
            # Create market order for simplicity (could be enhanced to use limit orders)
            order = create_market_order(
                symbol=symbol,
                quantity=shares,
                strategy=config.get('trading', {}).get('strategy_name', 'systematic')
            )
            
            # Route the order
            routing_result = await router.route_order(order)
            routing_results.append({
                'symbol': symbol,
                'order_id': order.order_id,
                'decision': routing_result.decision.value,
                'rejection_reason': routing_result.rejection_reason,
                'routing_notes': routing_result.routing_notes
            })
            
            if routing_result.decision.value == 'APPROVE':
                orders_submitted += 1
                logger.info(f"Order submitted: {symbol} {shares} shares")
            else:
                orders_rejected += 1
                logger.warning(f"Order rejected: {symbol} - {routing_result.rejection_reason}")
        
        routing_summary = {
            'status': 'completed',
            'trading_mode': trading_mode,
            'orders_submitted': orders_submitted,
            'orders_rejected': orders_rejected,
            'routing_results': routing_results,
            'routing_timestamp': datetime.now()
        }
        
        logger.info(f"Order routing complete: {orders_submitted} submitted, {orders_rejected} rejected")
        
        # Cleanup
        await broker.disconnect()
        risk_checker.close()
        
        return routing_summary
        
    except Exception as e:
        logger.error(f"Failed to route orders: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'orders_submitted': 0,
            'orders_rejected': 0
        }


@task(retries=1, retry_delay_seconds=30)
async def monitor_fills(routing_summary: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Monitor for order fills and update fill store"""
    logger = get_run_logger()
    
    if routing_summary.get('orders_submitted', 0) == 0:
        logger.info("No orders to monitor")
        return {'status': 'no_fills', 'fills_received': 0}
    
    try:
        # Setup fill monitoring
        max_wait_time = config.get('trading', {}).get('fill_timeout_minutes', 5)
        fills_received = 0
        
        # In a real implementation, this would:
        # 1. Connect to broker and listen for fill callbacks
        # 2. Wait for fills with timeout
        # 3. Store fills to FillStore
        # 4. Update position tracking
        
        # For now, simulate some monitoring time
        await asyncio.sleep(2)  # Brief monitoring period
        
        # Setup FillStore for when we do get fills
        fill_store = FillStore()
        
        fill_summary = {
            'status': 'monitoring_complete',
            'fills_received': fills_received,
            'orders_monitored': routing_summary.get('orders_submitted', 0),
            'monitoring_duration_seconds': 2,
            'fill_timeout_minutes': max_wait_time,
            'monitoring_timestamp': datetime.now()
        }
        
        logger.info(f"Fill monitoring complete: {fills_received} fills received")
        
        fill_store.close()
        return fill_summary
        
    except Exception as e:
        logger.error(f"Failed to monitor fills: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'fills_received': 0
        }


@task
def create_daily_snapshot(pipeline_result: Dict[str, Any], 
                         signals: pd.DataFrame,
                         sized_positions: pd.DataFrame,
                         risk_result: Dict[str, Any],
                         routing_summary: Dict[str, Any],
                         fill_summary: Dict[str, Any],
                         config: Dict[str, Any]) -> Dict[str, Any]:
    """Create daily trading snapshot/summary"""
    logger = get_run_logger()
    
    try:
        snapshot = {
            'date': datetime.now().date(),
            'timestamp': datetime.now(),
            'system_status': 'operational',
            
            # Data pipeline summary
            'data_pipeline': {
                'status': pipeline_result.get('status', 'unknown'),
                'symbols_processed': pipeline_result.get('symbols_processed', 0),
                'ohlc_records': pipeline_result.get('ohlc_records', 0),
                'fundamental_records': pipeline_result.get('fundamental_records', 0),
                'news_articles': pipeline_result.get('news_articles', 0)
            },
            
            # Signal generation summary
            'signal_generation': {
                'signals_generated': len(signals),
                'avg_signal_score': signals['composite_score'].mean() if not signals.empty else 0,
                'top_signal': signals.iloc[0]['symbol'] if not signals.empty else None,
                'top_signal_score': signals.iloc[0]['composite_score'] if not signals.empty else 0
            },
            
            # Position sizing summary
            'position_sizing': {
                'positions_sized': len(sized_positions),
                'total_target_value': sized_positions['target_value'].sum() if not sized_positions.empty else 0,
                'avg_position_size': sized_positions['target_value'].mean() if not sized_positions.empty else 0
            },
            
            # Risk management summary
            'risk_management': {
                'status': risk_result.get('status', 'unknown'),
                'positions_approved': risk_result.get('positions_approved', 0),
                'positions_rejected': risk_result.get('positions_rejected', 0),
                'violations_count': len(risk_result.get('violations', []))
            },
            
            # Execution summary
            'execution': {
                'trading_mode': routing_summary.get('trading_mode', 'unknown'),
                'orders_submitted': routing_summary.get('orders_submitted', 0),
                'orders_rejected': routing_summary.get('orders_rejected', 0),
                'fills_received': fill_summary.get('fills_received', 0)
            },
            
            # System health
            'system_health': {
                'data_quality_score': 1.0 if pipeline_result.get('status') == 'success' else 0.5,
                'execution_rate': routing_summary.get('orders_submitted', 0) / max(len(sized_positions), 1),
                'risk_compliance': 1.0 if risk_result.get('status') in ['ok', 'warning'] else 0.0
            }
        }
        
        # Store snapshot to database (optional)
        try:
            storage = DataStorage()
            # Could store snapshot to a daily_snapshots table
            logger.info("Daily snapshot created successfully")
            storage.close()
        except Exception as e:
            logger.warning(f"Failed to store daily snapshot: {e}")
        
        logger.info(f"Daily snapshot: {snapshot['signal_generation']['signals_generated']} signals → "
                   f"{snapshot['execution']['orders_submitted']} orders → "
                   f"{snapshot['execution']['fills_received']} fills")
        
        return snapshot
        
    except Exception as e:
        logger.error(f"Failed to create daily snapshot: {e}")
        return {
            'date': datetime.now().date(),
            'timestamp': datetime.now(),
            'system_status': 'error',
            'error': str(e)
        }


@flow(
    name="Daily Trading Flow",
    description="Complete daily trading flow: data → signals → sizing → risk → execution → monitoring",
    task_runner=ConcurrentTaskRunner()
)
async def daily_trading_flow(config_path: str = "config") -> Dict[str, Any]:
    """Main daily trading flow"""
    logger = get_run_logger()
    
    try:
        logger.info("🚀 Starting daily trading flow")
        
        # Load configuration
        config = load_config(config_path)
        
        # Phase 1: Data Pipeline (run data pipeline first)
        logger.info("📊 Phase 1: Running data pipeline")
        pipeline_result = await daily_data_pipeline.fn(config_path)
        
        if pipeline_result.get('status') != 'success':
            logger.error("Data pipeline failed - aborting trading flow")
            return {
                'status': 'aborted',
                'reason': 'data_pipeline_failed',
                'timestamp': datetime.now(),
                'pipeline_result': pipeline_result
            }
        
        # Phase 2: Signal Generation
        logger.info("🎯 Phase 2: Generating trading signals")
        signals = generate_trading_signals(config)
        
        if signals.empty:
            logger.warning("No trading signals generated - completing flow without trades")
            return {
                'status': 'completed_no_trades',
                'reason': 'no_signals',
                'timestamp': datetime.now(),
                'pipeline_result': pipeline_result
            }
        
        # Phase 3: Position Sizing
        logger.info("📏 Phase 3: Sizing positions")
        sized_positions = size_positions(signals, config)
        
        # Phase 4: Risk Pre-check
        logger.info("🛡️ Phase 4: Pre-trade risk checks")
        risk_result = precheck_risk(sized_positions, config)
        
        if risk_result['status'] == 'violation':
            logger.error("Critical risk violations - aborting trading")
            return {
                'status': 'aborted',
                'reason': 'risk_violation',
                'timestamp': datetime.now(),
                'risk_violations': risk_result['violations']
            }
        
        # Phase 5: Order Routing
        logger.info("🔀 Phase 5: Routing orders")
        routing_summary = await route_orders(risk_result['approved_positions'], config)
        
        # Phase 6: Fill Monitoring
        logger.info("👀 Phase 6: Monitoring fills")
        fill_summary = await monitor_fills(routing_summary, config)
        
        # Phase 7: Daily Snapshot
        logger.info("📸 Phase 7: Creating daily snapshot")
        snapshot = create_daily_snapshot(
            pipeline_result, signals, sized_positions, 
            risk_result, routing_summary, fill_summary, config
        )
        
        # Final summary
        flow_result = {
            'status': 'completed',
            'timestamp': datetime.now(),
            'daily_snapshot': snapshot,
            'performance_summary': {
                'signals_generated': len(signals),
                'positions_approved': risk_result.get('positions_approved', 0),
                'orders_submitted': routing_summary.get('orders_submitted', 0),
                'fills_received': fill_summary.get('fills_received', 0)
            }
        }
        
        logger.info(f"✅ Daily trading flow completed successfully: {flow_result['performance_summary']}")
        return flow_result
        
    except Exception as e:
        logger.error(f"❌ Daily trading flow failed: {e}")
        return {
            'status': 'failed',
            'timestamp': datetime.now(),
            'error': str(e)
        }


# Convenience flow for data-only runs
@flow(name="Data Only Flow")
async def data_only_flow(config_path: str = "config") -> Dict[str, Any]:
    """Run just the data pipeline without trading"""
    return await daily_data_pipeline.fn(config_path)


if __name__ == "__main__":
    # Run the complete trading flow
    import asyncio
    
    # Check trading mode
    trading_mode = os.getenv('EXO_MODE', 'data_only')
    
    if trading_mode == 'data_only':
        print("Running data-only pipeline...")
        result = asyncio.run(data_only_flow())
    else:
        print(f"Running full trading flow in {trading_mode} mode...")
        result = asyncio.run(daily_trading_flow())
    
    print(f"Flow result: {result['status']}")
    if 'daily_snapshot' in result:
        snapshot = result['daily_snapshot']
        print(f"Summary: {snapshot['signal_generation']['signals_generated']} signals → "
              f"{snapshot['execution']['orders_submitted']} orders → "
              f"{snapshot['execution']['fills_received']} fills")