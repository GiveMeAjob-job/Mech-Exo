"""
Daily Canary Performance Tracker

Flow: pull_fills_today → calc_canary_pnl → update_canary_performance → compute_rolling_sharpe
Scheduled at 23:30 UTC (after market close) to capture complete daily performance.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mech_exo.execution.fill_store import FillStore
from mech_exo.reporting.pnl import (
    compute_tag_based_nav, 
    compute_daily_pnl,
    compute_rolling_sharpe,
    store_daily_performance
)
from mech_exo.datasource.storage import DataStorage


@task(retries=2, retry_delay_seconds=30)
def pull_fills_today(target_date: Optional[date] = None) -> Dict[str, Any]:
    """Fetch today's fills and group by allocation tag"""
    logger = get_run_logger()
    
    if target_date is None:
        target_date = date.today()
    
    try:
        fill_store = FillStore()
        
        # Get all fills for target date
        fills = fill_store.get_fills(
            start_date=datetime.combine(target_date, datetime.min.time()),
            end_date=datetime.combine(target_date, datetime.max.time())
        )
        
        # Group fills by tag
        base_fills = [f for f in fills if f.tag == 'base']
        canary_fills = [f for f in fills if f.tag == 'ml_canary']
        
        # Calculate basic metrics
        total_fills = len(fills)
        base_volume = sum(abs(f.quantity) for f in base_fills)
        canary_volume = sum(abs(f.quantity) for f in canary_fills)
        
        base_notional = sum(f.gross_value for f in base_fills)
        canary_notional = sum(f.gross_value for f in canary_fills)
        
        result = {
            'status': 'success',
            'target_date': target_date,
            'total_fills': total_fills,
            'base_fills': len(base_fills),
            'canary_fills': len(canary_fills),
            'base_volume': base_volume,
            'canary_volume': canary_volume,
            'base_notional': base_notional,
            'canary_notional': canary_notional,
            'canary_allocation_pct': canary_notional / (base_notional + canary_notional) * 100 if (base_notional + canary_notional) > 0 else 0
        }
        
        logger.info(f"Pulled {total_fills} fills for {target_date}: {len(base_fills)} base, {len(canary_fills)} canary")
        logger.info(f"Canary allocation: {result['canary_allocation_pct']:.1f}% of total notional")
        
        fill_store.close()
        return result
        
    except Exception as e:
        logger.error(f"Failed to pull fills for {target_date}: {e}")
        return {
            'status': 'failed',
            'target_date': target_date,
            'error': str(e),
            'total_fills': 0,
            'base_fills': 0,
            'canary_fills': 0
        }


@task(retries=2, retry_delay_seconds=30)
def calc_canary_pnl(fills_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate P&L and NAV changes for base vs canary allocations"""
    logger = get_run_logger()
    
    target_date = fills_summary['target_date']
    
    if fills_summary['status'] != 'success':
        logger.warning(f"Skipping P&L calc due to failed fills pull: {fills_summary.get('error')}")
        return {
            'status': 'skipped',
            'target_date': target_date,
            'reason': 'fills_pull_failed'
        }
    
    try:
        # Compute NAV by tag for current and previous day
        current_nav = compute_tag_based_nav(target_date)
        previous_nav = compute_tag_based_nav(target_date - timedelta(days=1))
        
        # Calculate daily P&L
        daily_pnl = compute_daily_pnl(target_date)
        
        # Calculate percentage changes
        base_pnl_pct = (daily_pnl['base'] / previous_nav['base'] * 100) if previous_nav['base'] > 0 else 0
        canary_pnl_pct = (daily_pnl['ml_canary'] / previous_nav['ml_canary'] * 100) if previous_nav['ml_canary'] > 0 else 0
        
        # Calculate alpha (canary outperformance)
        alpha_bps = (canary_pnl_pct - base_pnl_pct) * 100  # Convert to basis points
        
        result = {
            'status': 'success',
            'target_date': target_date,
            'current_nav': current_nav,
            'previous_nav': previous_nav,
            'daily_pnl': daily_pnl,
            'base_pnl_pct': base_pnl_pct,
            'canary_pnl_pct': canary_pnl_pct,
            'alpha_bps': alpha_bps,
            'fills_count': fills_summary['total_fills']
        }
        
        logger.info(f"P&L calculated for {target_date}:")
        logger.info(f"  Base: ${daily_pnl['base']:.2f} ({base_pnl_pct:+.2f}%), NAV: ${current_nav['base']:,.0f}")
        logger.info(f"  Canary: ${daily_pnl['ml_canary']:.2f} ({canary_pnl_pct:+.2f}%), NAV: ${current_nav['ml_canary']:,.0f}")
        logger.info(f"  Alpha: {alpha_bps:+.1f} bps")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to calculate P&L for {target_date}: {e}")
        return {
            'status': 'failed',
            'target_date': target_date,
            'error': str(e)
        }


@task(retries=2, retry_delay_seconds=30)  
def update_canary_performance(pnl_result: Dict[str, Any]) -> Dict[str, Any]:
    """UPSERT daily performance into canary_performance table"""
    logger = get_run_logger()
    
    target_date = pnl_result['target_date']
    
    if pnl_result['status'] != 'success':
        logger.warning(f"Skipping performance update due to failed P&L calc: {pnl_result.get('error')}")
        return {
            'status': 'skipped',
            'target_date': target_date,
            'reason': 'pnl_calc_failed'
        }
    
    try:
        # Use the comprehensive store_daily_performance function
        success = store_daily_performance(target_date)
        
        if success:
            result = {
                'status': 'success',
                'target_date': target_date,
                'operation': 'upsert',
                'base_nav': pnl_result['current_nav']['base'],
                'canary_nav': pnl_result['current_nav']['ml_canary'],
                'base_pnl': pnl_result['daily_pnl']['base'],
                'canary_pnl': pnl_result['daily_pnl']['ml_canary'],
                'alpha_bps': pnl_result['alpha_bps']
            }
            
            logger.info(f"Updated canary_performance for {target_date}")
            logger.info(f"  Base NAV: ${result['base_nav']:,.0f}, P&L: ${result['base_pnl']:+.2f}")
            logger.info(f"  Canary NAV: ${result['canary_nav']:,.0f}, P&L: ${result['canary_pnl']:+.2f}")
            
            return result
        else:
            return {
                'status': 'failed',
                'target_date': target_date,
                'error': 'store_daily_performance returned False'
            }
            
    except Exception as e:
        logger.error(f"Failed to update canary_performance for {target_date}: {e}")
        return {
            'status': 'failed',
            'target_date': target_date,
            'error': str(e)
        }


@task(retries=2, retry_delay_seconds=30)
def compute_rolling_sharpe_metrics(performance_result: Dict[str, Any], window_days: int = 30) -> Dict[str, Any]:
    """Compute 30-day rolling Sharpe ratios for base and canary"""
    logger = get_run_logger()
    
    target_date = performance_result['target_date']
    
    if performance_result['status'] != 'success':
        logger.warning(f"Skipping Sharpe calc due to failed performance update: {performance_result.get('error')}")
        return {
            'status': 'skipped',
            'target_date': target_date,
            'reason': 'performance_update_failed'
        }
    
    try:
        # Calculate rolling Sharpe for both base and canary
        base_sharpe = compute_rolling_sharpe('base', window_days, target_date)
        canary_sharpe = compute_rolling_sharpe('ml_canary', window_days, target_date)
        
        # Calculate Sharpe difference  
        sharpe_diff = canary_sharpe - base_sharpe
        
        # Check data quality (need minimum observations)
        storage = DataStorage()
        start_date = target_date - timedelta(days=window_days + 10)
        
        count_query = """
            SELECT COUNT(*) 
            FROM canary_performance 
            WHERE date >= ? AND date <= ?
        """
        
        observations = storage.conn.execute(count_query, [str(start_date), str(target_date)]).fetchone()[0]
        storage.close()
        
        # Determine quality status
        if observations >= window_days * 0.8:  # 80% of expected observations
            quality_status = 'good'
        elif observations >= window_days * 0.5:  # 50% of expected observations
            quality_status = 'fair'
        else:
            quality_status = 'poor'
        
        result = {
            'status': 'success',
            'target_date': target_date,
            'window_days': window_days,
            'base_sharpe': base_sharpe,
            'canary_sharpe': canary_sharpe,
            'sharpe_diff': sharpe_diff,
            'observations': observations,
            'data_quality': quality_status,
            'canary_outperforming': sharpe_diff > 0
        }
        
        logger.info(f"Rolling Sharpe ({window_days}d) for {target_date}:")
        logger.info(f"  Base: {base_sharpe:.3f}")
        logger.info(f"  Canary: {canary_sharpe:.3f}")
        logger.info(f"  Difference: {sharpe_diff:+.3f} ({'canary ahead' if sharpe_diff > 0 else 'base ahead'})")
        logger.info(f"  Data quality: {quality_status} ({observations}/{window_days} obs)")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to compute rolling Sharpe for {target_date}: {e}")
        return {
            'status': 'failed',
            'target_date': target_date,
            'error': str(e)
        }


@task(retries=1, retry_delay_seconds=30)
def check_auto_disable_logic(sharpe_result: Dict[str, Any]) -> Dict[str, Any]:
    """Check if canary should be auto-disabled based on performance thresholds"""
    logger = get_run_logger()
    
    target_date = sharpe_result['target_date']
    
    if sharpe_result['status'] != 'success':
        logger.warning(f"Skipping auto-disable check due to failed Sharpe calc: {sharpe_result.get('error')}")
        return {
            'status': 'skipped',
            'target_date': target_date,
            'reason': 'sharpe_calc_failed',
            'action_taken': 'none'
        }
    
    try:
        from mech_exo.execution.allocation import (
            get_allocation_config, update_canary_enabled, is_canary_enabled, 
            check_hysteresis_trigger
        )
        from mech_exo.utils.alerts import AlertManager, Alert, AlertType, AlertLevel
        from mech_exo.datasource.storage import DataStorage
        
        # Get configuration
        allocation_config = get_allocation_config()
        disable_rule = allocation_config['disable_rule']
        
        canary_sharpe = sharpe_result['canary_sharpe']
        data_quality = sharpe_result['data_quality']
        observations = sharpe_result['observations']
        min_observations = disable_rule['min_observations']
        
        logger.info(f"Auto-disable check: Canary Sharpe={canary_sharpe:.3f}, "
                   f"Min Observations={min_observations}, Current Observations={observations}")
        
        # Check hysteresis trigger logic
        hysteresis_result = check_hysteresis_trigger(canary_sharpe)
        
        # Check if we have sufficient data quality and observations
        data_sufficient = (
            data_quality in ['good', 'fair'] and  # Need reasonable data quality
            observations >= min_observations      # Need minimum observations for reliable decision
        )
        
        # Final disable decision combines data quality and hysteresis trigger
        should_disable = data_sufficient and hysteresis_result['should_trigger']
        
        auto_disable_result = {
            'status': 'success',
            'target_date': target_date,
            'canary_sharpe': canary_sharpe,
            'base_sharpe': sharpe_result.get('base_sharpe', 0.0),
            'sharpe_diff': sharpe_result.get('sharpe_diff', 0.0),
            'canary_outperforming': sharpe_result.get('canary_outperforming', False),
            'threshold': hysteresis_result['threshold'],
            'observations': observations,
            'min_observations_required': min_observations,
            'data_quality': data_quality,
            'data_sufficient': data_sufficient,
            'hysteresis_result': hysteresis_result,
            'should_disable': should_disable,
            'action_taken': 'none',
            'alert_sent': False
        }
        
        # If canary should be disabled and is currently enabled
        if should_disable and is_canary_enabled():
            breach_days = hysteresis_result['current_breach_days']
            required_days = hysteresis_result['required_days']
            threshold = hysteresis_result['threshold']
            
            logger.warning(f"🚨 AUTO-DISABLE TRIGGERED: Canary Sharpe {canary_sharpe:.3f} < {threshold:.3f} "
                          f"for {breach_days} consecutive days (>= {required_days} required)")
            
            # Disable canary allocation
            success = update_canary_enabled(False)
            
            if success:
                auto_disable_result['action_taken'] = 'disabled'
                logger.info("✅ Canary allocation disabled successfully")
                
                # Send Telegram alert
                try:
                    alert_manager = AlertManager()
                    
                    # Create alert for auto-disable with hysteresis info
                    alert = Alert(
                        alert_type=AlertType.RISK_VIOLATION,
                        level=AlertLevel.CRITICAL,
                        title="🚨 Canary Auto-Disabled (Hysteresis)",
                        message=f"Canary allocation automatically disabled due to poor performance:\n\n"
                               f"• Canary Sharpe (30d): {canary_sharpe:.3f}\n"
                               f"• Threshold: {threshold:.3f}\n"
                               f"• Consecutive breach days: {breach_days}/{required_days}\n"
                               f"• Total observations: {observations} days\n"
                               f"• Data quality: {data_quality}\n\n"
                               f"Hysteresis protection ensured {required_days} consecutive days below threshold.\n"
                               f"All new orders will use base allocation only.\n"
                               f"Manual review and re-enable required.",
                        timestamp=datetime.now(),
                        data={
                            'canary_sharpe': canary_sharpe,
                            'threshold': threshold,
                            'consecutive_breach_days': breach_days,
                            'required_days': required_days,
                            'observations': observations,
                            'data_quality': data_quality,
                            'auto_disabled': True,
                            'hysteresis_triggered': True
                        }
                    )
                    
                    alert_sent = alert_manager.send_alert(alert, channels=['telegram'])
                    auto_disable_result['alert_sent'] = alert_sent
                    
                    if alert_sent:
                        logger.info("📱 Auto-disable Telegram alert sent successfully")
                    else:
                        logger.warning("⚠️ Failed to send auto-disable Telegram alert")
                        
                except Exception as e:
                    logger.error(f"Failed to send auto-disable alert: {e}")
                    auto_disable_result['alert_error'] = str(e)
            else:
                auto_disable_result['action_taken'] = 'disable_failed'
                logger.error("❌ Failed to disable canary allocation")
        
        elif should_disable and not is_canary_enabled():
            logger.info("ℹ️ Canary would be disabled but is already disabled")
            auto_disable_result['action_taken'] = 'already_disabled'
        
        elif not should_disable:
            breach_days = hysteresis_result['current_breach_days']
            if breach_days > 0:
                logger.info(f"⏳ Canary performance below threshold but hysteresis active: "
                           f"Sharpe {canary_sharpe:.3f}, breach days {breach_days}/{hysteresis_result['required_days']}")
                auto_disable_result['action_taken'] = 'hysteresis_pending'
            else:
                logger.info(f"✅ Canary performance acceptable: Sharpe {canary_sharpe:.3f} >= {hysteresis_result['threshold']:.3f}")
                auto_disable_result['action_taken'] = 'none_needed'
        
        return auto_disable_result
        
    except Exception as e:
        logger.error(f"Failed to check auto-disable logic: {e}")
        return {
            'status': 'failed',
            'target_date': target_date,
            'error': str(e),
            'action_taken': 'error'
        }


@task(retries=1, retry_delay_seconds=30)
def update_health_endpoint_cache(auto_disable_result: Dict[str, Any]) -> Dict[str, Any]:
    """Update cached values for /healthz endpoint"""
    logger = get_run_logger()
    
    target_date = auto_disable_result['target_date']
    
    if auto_disable_result['status'] != 'success':
        logger.warning(f"Skipping health cache update due to failed auto-disable check: {auto_disable_result.get('error')}")
        return {
            'status': 'skipped',
            'target_date': target_date,
            'reason': 'auto_disable_check_failed'
        }
    
    try:
        # Check if canary is currently enabled (may have changed due to auto-disable)
        from mech_exo.execution.allocation import is_canary_enabled
        canary_enabled = is_canary_enabled()
        
        # Create health data structure
        health_data = {
            'canary_sharpe_30d': auto_disable_result['canary_sharpe'],
            'base_sharpe_30d': auto_disable_result.get('base_sharpe', 0.0),  # Should be passed from previous task
            'sharpe_diff': auto_disable_result.get('sharpe_diff', 0.0),
            'canary_enabled': canary_enabled,
            'canary_outperforming': auto_disable_result.get('canary_outperforming', False),
            'data_quality': auto_disable_result['data_quality'],
            'auto_disable_action': auto_disable_result['action_taken'],
            'last_updated': datetime.now().isoformat(),
            'target_date': str(target_date)
        }
        
        # Store in a simple cache file for health endpoint to read
        cache_dir = Path("data")
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / "canary_health_cache.json"
        
        import json
        with open(cache_file, 'w') as f:
            json.dump(health_data, f, indent=2)
        
        result = {
            'status': 'success',
            'target_date': target_date,
            'cache_file': str(cache_file),
            'health_data': health_data,
            'auto_disable_action': auto_disable_result['action_taken']
        }
        
        logger.info(f"Updated health cache for {target_date}:")
        logger.info(f"  Canary enabled: {canary_enabled}")
        logger.info(f"  Canary Sharpe: {auto_disable_result['canary_sharpe']:.3f}")
        logger.info(f"  Auto-disable action: {auto_disable_result['action_taken']}")
        logger.info(f"  Cache file: {cache_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to update health cache for {target_date}: {e}")
        return {
            'status': 'failed',
            'target_date': target_date,
            'error': str(e)
        }


@flow(
    name="Daily Canary Performance Tracker",
    description="Track daily canary vs base performance, compute rolling Sharpe, update health cache",
    task_runner=SequentialTaskRunner(),
    retries=1,
    retry_delay_seconds=300  # 5 minute delay between flow retries
)
def canary_performance_flow(target_date: Optional[str] = None, window_days: int = 30) -> Dict[str, Any]:
    """
    Main canary performance tracking flow
    
    Args:
        target_date: Date to process (YYYY-MM-DD), defaults to today
        window_days: Rolling window for Sharpe calculation (default: 30)
    """
    logger = get_run_logger()
    
    try:
        # Parse target date
        if target_date is None:
            process_date = date.today()
        else:
            process_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        logger.info(f"🎯 Starting canary performance tracking for {process_date}")
        
        # Step 1: Pull today's fills
        logger.info("📊 Step 1: Pulling today's fills")
        fills_summary = pull_fills_today(process_date)
        
        # Step 2: Calculate P&L
        logger.info("💰 Step 2: Calculating canary vs base P&L")
        pnl_result = calc_canary_pnl(fills_summary)
        
        # Step 3: Update performance table
        logger.info("💾 Step 3: Updating canary_performance table")
        performance_result = update_canary_performance(pnl_result)
        
        # Step 4: Compute rolling Sharpe
        logger.info(f"📈 Step 4: Computing {window_days}-day rolling Sharpe")
        sharpe_result = compute_rolling_sharpe_metrics(performance_result, window_days)
        
        # Step 5: Check auto-disable logic and send alerts
        logger.info("🤖 Step 5: Checking auto-disable logic")
        auto_disable_result = check_auto_disable_logic(sharpe_result)
        
        # Step 6: Update health endpoint cache
        logger.info("🏥 Step 6: Updating health endpoint cache")
        health_result = update_health_endpoint_cache(auto_disable_result)
        
        # Create summary
        flow_summary = {
            'status': 'completed',
            'target_date': process_date,
            'timestamp': datetime.now(),
            'steps_completed': {
                'fills_pull': fills_summary['status'],
                'pnl_calc': pnl_result['status'],
                'performance_update': performance_result['status'],
                'sharpe_calc': sharpe_result['status'],
                'auto_disable_check': auto_disable_result['status'],
                'health_cache': health_result['status']
            },
            'performance_metrics': {
                'total_fills': fills_summary.get('total_fills', 0),
                'canary_allocation_pct': fills_summary.get('canary_allocation_pct', 0),
                'alpha_bps': pnl_result.get('alpha_bps', 0) if pnl_result['status'] == 'success' else 0,
                'canary_sharpe': sharpe_result.get('canary_sharpe', 0) if sharpe_result['status'] == 'success' else 0,
                'base_sharpe': sharpe_result.get('base_sharpe', 0) if sharpe_result['status'] == 'success' else 0,
                'sharpe_diff': sharpe_result.get('sharpe_diff', 0) if sharpe_result['status'] == 'success' else 0,
                'auto_disable_action': auto_disable_result.get('action_taken', 'none') if auto_disable_result['status'] == 'success' else 'unknown',
                'alert_sent': auto_disable_result.get('alert_sent', False) if auto_disable_result['status'] == 'success' else False
            }
        }
        
        # Determine overall success
        successful_steps = sum(1 for status in flow_summary['steps_completed'].values() if status == 'success')
        total_steps = len(flow_summary['steps_completed'])
        
        if successful_steps == total_steps:
            flow_summary['overall_status'] = 'success'
            logger.info(f"✅ Canary performance tracking completed successfully for {process_date}")
        elif successful_steps >= total_steps * 0.6:  # 60% success threshold
            flow_summary['overall_status'] = 'partial_success'
            logger.warning(f"⚠️ Canary performance tracking partially successful: {successful_steps}/{total_steps} steps")
        else:
            flow_summary['overall_status'] = 'failed'
            logger.error(f"❌ Canary performance tracking failed: {successful_steps}/{total_steps} steps successful")
        
        # Log key metrics
        if flow_summary['performance_metrics']['total_fills'] > 0:
            logger.info(f"📈 Performance Summary:")
            logger.info(f"  • {flow_summary['performance_metrics']['total_fills']} fills processed")
            logger.info(f"  • {flow_summary['performance_metrics']['canary_allocation_pct']:.1f}% canary allocation")
            logger.info(f"  • {flow_summary['performance_metrics']['alpha_bps']:+.1f} bps alpha")
            logger.info(f"  • Sharpe: Base {flow_summary['performance_metrics']['base_sharpe']:.3f}, Canary {flow_summary['performance_metrics']['canary_sharpe']:.3f}")
        
        return flow_summary
        
    except Exception as e:
        logger.error(f"❌ Canary performance flow failed: {e}")
        return {
            'status': 'failed',
            'target_date': target_date or str(date.today()),
            'timestamp': datetime.now(),
            'error': str(e),
            'overall_status': 'failed'
        }


# Manual execution helper
def run_manual_canary_performance(target_date: Optional[str] = None, window_days: int = 30) -> Dict[str, Any]:
    """Run canary performance tracking manually for testing"""
    print(f"🎯 Running canary performance tracking manually...")
    print(f"Target date: {target_date or 'today'}")
    print(f"Window days: {window_days}")
    
    result = canary_performance_flow(target_date, window_days)
    
    print(f"\n📊 Results:")
    print(f"Status: {result.get('overall_status', 'unknown')}")
    if 'performance_metrics' in result:
        metrics = result['performance_metrics']
        print(f"Fills: {metrics['total_fills']}")
        print(f"Alpha: {metrics['alpha_bps']:+.1f} bps")
        print(f"Sharpe diff: {metrics['sharpe_diff']:+.3f}")
    
    return result


if __name__ == "__main__":
    # Run manual test
    import sys
    
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    window_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    result = run_manual_canary_performance(target_date, window_days)
    print(f"\nFlow completed with status: {result.get('overall_status')}")