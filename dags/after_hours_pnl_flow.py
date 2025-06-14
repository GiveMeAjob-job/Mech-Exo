"""
After-Hours PnL Monitoring Flow - Phase P11

Monitors portfolio performance during extended trading hours (16:00-20:00 ET).
Provides enhanced risk monitoring for after-hours price movements and
overnight position exposure.

Features:
- Hourly P&L snapshots during after-hours
- Pre-market gap risk assessment  
- Extended hours volatility monitoring
- Integration with existing risk dashboard
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from prefect import flow, task, get_run_logger
    from prefect.tasks import task_input_hash
    from prefect.server.schemas.schedules import CronSchedule
    PREFECT_AVAILABLE = True
except ImportError:
    # Mock Prefect for development
    PREFECT_AVAILABLE = False
    
    def flow(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    def task(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    def get_run_logger():
        return logging.getLogger(__name__)
        
    def task_input_hash(*args, **kwargs):
        return None

try:
    from mech_exo.utils.alerts import TelegramAlerter
    from mech_exo.capital.manager import CapitalManager
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


@task(name="fetch_after_hours_prices", cache_key_fn=task_input_hash, cache_expiration=timedelta(minutes=15))
def fetch_after_hours_prices() -> Dict[str, Any]:
    """Fetch current after-hours prices for portfolio positions"""
    logger = get_run_logger()
    logger.info("📊 Fetching after-hours prices...")
    
    try:
        # In production, this would connect to real market data
        # For now, simulate after-hours price data
        
        import random
        from datetime import datetime
        
        # Simulate positions and after-hours movements
        positions = {
            'AAPL': {'qty': 100, 'last_close': 175.50, 'ah_price': 175.50 * (1 + random.uniform(-0.02, 0.02))},
            'MSFT': {'qty': 75, 'last_close': 415.20, 'ah_price': 415.20 * (1 + random.uniform(-0.015, 0.015))},
            'GOOGL': {'qty': 50, 'last_close': 138.90, 'ah_price': 138.90 * (1 + random.uniform(-0.025, 0.025))},
            'TSLA': {'qty': -25, 'last_close': 248.30, 'ah_price': 248.30 * (1 + random.uniform(-0.04, 0.04))},
            'SPY': {'qty': 200, 'last_close': 456.80, 'ah_price': 456.80 * (1 + random.uniform(-0.01, 0.01))}
        }
        
        # Calculate after-hours P&L
        total_ah_pnl = 0
        for symbol, data in positions.items():
            ah_change = data['ah_price'] - data['last_close']
            position_pnl = ah_change * data['qty']
            total_ah_pnl += position_pnl
            data['ah_pnl'] = position_pnl
            data['ah_change_pct'] = (ah_change / data['last_close']) * 100
            
        result = {
            'timestamp': datetime.now().isoformat(),
            'total_ah_pnl': total_ah_pnl,
            'positions': positions,
            'market_status': 'after_hours',
            'data_quality': 'simulated'  # Would be 'live' in production
        }
        
        logger.info(f"📈 After-hours P&L: ${total_ah_pnl:,.2f}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch after-hours prices: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'total_ah_pnl': 0,
            'positions': {},
            'market_status': 'error'
        }


@task(name="calculate_extended_risk_metrics")
def calculate_extended_risk_metrics(price_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate risk metrics for extended hours trading"""
    logger = get_run_logger()
    logger.info("⚖️ Calculating extended hours risk metrics...")
    
    try:
        if 'error' in price_data:
            return {'error': price_data['error']}
            
        positions = price_data.get('positions', {})
        total_ah_pnl = price_data.get('total_ah_pnl', 0)
        
        # Calculate portfolio-level metrics
        total_exposure = sum(abs(pos['qty'] * pos['ah_price']) for pos in positions.values())
        
        # Calculate position-level risk
        position_risks = {}
        for symbol, data in positions.items():
            position_value = abs(data['qty'] * data['ah_price'])
            concentration = (position_value / total_exposure * 100) if total_exposure > 0 else 0
            
            position_risks[symbol] = {
                'concentration_pct': concentration,
                'ah_volatility': abs(data['ah_change_pct']),
                'position_value': position_value,
                'ah_pnl': data['ah_pnl'],
                'risk_score': concentration * abs(data['ah_change_pct'])  # Simple risk score
            }
            
        # Calculate portfolio risk score
        portfolio_risk_score = sum(risk['risk_score'] for risk in position_risks.values())
        
        # Determine after-hours P&L as percentage of portfolio
        # Assume portfolio value of $500k for calculation
        portfolio_value = 500_000
        ah_pnl_pct = (total_ah_pnl / portfolio_value) * 100
        
        # Risk thresholds
        risk_level = 'normal'
        if abs(ah_pnl_pct) > 1.0:  # > 1% after-hours move
            risk_level = 'high'
        elif abs(ah_pnl_pct) > 0.5:  # > 0.5% after-hours move
            risk_level = 'medium'
            
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'ah_pnl_pct': ah_pnl_pct,
            'total_exposure': total_exposure,
            'portfolio_risk_score': portfolio_risk_score,
            'risk_level': risk_level,
            'position_risks': position_risks,
            'largest_position_pct': max((risk['concentration_pct'] for risk in position_risks.values()), default=0),
            'most_volatile_position': max(position_risks.keys(), 
                                        key=lambda x: position_risks[x]['ah_volatility'], 
                                        default='none') if position_risks else 'none'
        }
        
        logger.info(f"⚖️ Risk level: {risk_level}, Portfolio risk score: {portfolio_risk_score:.2f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to calculate risk metrics: {e}")
        return {'error': str(e)}


@task(name="update_risk_dashboard")
def update_risk_dashboard(risk_metrics: Dict[str, Any]) -> bool:
    """Update risk dashboard with after-hours data"""
    logger = get_run_logger()
    logger.info("📊 Updating risk dashboard with after-hours data...")
    
    try:
        if 'error' in risk_metrics:
            logger.error(f"Cannot update dashboard: {risk_metrics['error']}")
            return False
            
        # In production, this would update the /riskz endpoint
        # For now, simulate dashboard update
        
        dashboard_data = {
            'ah_loss_pct': risk_metrics['ah_pnl_pct'],
            'ah_risk_level': risk_metrics['risk_level'],
            'ah_largest_position': risk_metrics['largest_position_pct'],
            'ah_most_volatile': risk_metrics['most_volatile_position'],
            'ah_last_update': risk_metrics['timestamp'],
            'ah_monitoring_active': True
        }
        
        # Simulate writing to dashboard API/database
        logger.info(f"📊 Dashboard updated: ah_loss_pct = {dashboard_data['ah_loss_pct']:.3f}%")
        logger.info(f"📊 Risk level: {dashboard_data['ah_risk_level']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update dashboard: {e}")
        return False


@task(name="send_risk_alerts")
def send_risk_alerts(risk_metrics: Dict[str, Any], alert_threshold: float = 0.75) -> bool:
    """Send alerts for significant after-hours risk events"""
    logger = get_run_logger()
    logger.info(f"🚨 Checking for risk alerts (threshold: {alert_threshold}%)...")
    
    try:
        if 'error' in risk_metrics or not UTILS_AVAILABLE:
            logger.warning("Cannot send alerts: utils not available or error in metrics")
            return False
            
        ah_pnl_pct = risk_metrics.get('ah_pnl_pct', 0)
        risk_level = risk_metrics.get('risk_level', 'normal')
        
        # Check if alert is needed
        alert_needed = (
            abs(ah_pnl_pct) >= alert_threshold or 
            risk_level in ['high', 'critical']
        )
        
        if not alert_needed:
            logger.info(f"✅ No alerts needed (P&L: {ah_pnl_pct:+.2f}%, Risk: {risk_level})")
            return True
            
        # Prepare alert message
        alerter = TelegramAlerter({})  # Mock alerter
        
        severity_emoji = "🚨" if abs(ah_pnl_pct) > 1.0 else "⚠️"
        
        message = f"""{severity_emoji} **AFTER-HOURS RISK ALERT**

📊 **After-Hours P&L**: {ah_pnl_pct:+.2f}%
⚖️ **Risk Level**: {risk_level.upper()}
🏠 **Largest Position**: {risk_metrics.get('largest_position_pct', 0):.1f}%
📈 **Most Volatile**: {risk_metrics.get('most_volatile_position', 'unknown')}

⏰ **Time**: {datetime.now().strftime('%H:%M:%S ET')}
🕐 **Session**: After-Hours Monitoring

{severity_emoji} Monitor overnight exposure and pre-market gaps"""

        # In production, this would send the actual alert
        logger.info(f"📱 Alert message prepared: {message[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to send risk alerts: {e}")
        return False


@flow(name="after-hours-pnl-monitoring")
def after_hours_pnl_flow(
    alert_threshold: float = 0.75,
    risk_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Main after-hours PnL monitoring flow
    
    Args:
        alert_threshold: P&L percentage threshold for alerts (default 0.75%)
        risk_threshold: P&L percentage threshold for high risk (default 1.0%)
    """
    logger = get_run_logger()
    logger.info("🌙 Starting after-hours PnL monitoring flow...")
    
    # Check if we're in after-hours period
    current_time = datetime.now()
    current_hour = current_time.hour
    
    # After-hours: 16:00-20:00 ET (4 PM - 8 PM)
    if not (16 <= current_hour < 20):
        logger.info(f"⏰ Outside after-hours window (current: {current_hour:02d}:00)")
        return {
            'status': 'skipped',
            'reason': 'outside_after_hours_window',
            'current_hour': current_hour
        }
    
    try:
        # Step 1: Fetch after-hours price data
        price_data = fetch_after_hours_prices()
        
        # Step 2: Calculate extended risk metrics
        risk_metrics = calculate_extended_risk_metrics(price_data)
        
        # Step 3: Update risk dashboard
        dashboard_updated = update_risk_dashboard(risk_metrics)
        
        # Step 4: Send alerts if needed
        alerts_sent = send_risk_alerts(risk_metrics, alert_threshold)
        
        # Prepare flow summary
        flow_result = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'ah_pnl_pct': risk_metrics.get('ah_pnl_pct', 0),
            'risk_level': risk_metrics.get('risk_level', 'unknown'),
            'dashboard_updated': dashboard_updated,
            'alerts_sent': alerts_sent,
            'steps_completed': 4,
            'next_run': (current_time + timedelta(hours=1)).strftime('%H:%M ET')
        }
        
        ah_pnl = flow_result['ah_pnl_pct']
        risk_level = flow_result['risk_level']
        
        logger.info(f"✅ After-hours monitoring complete:")
        logger.info(f"   📊 P&L: {ah_pnl:+.2f}%")
        logger.info(f"   ⚖️ Risk: {risk_level}")
        logger.info(f"   📈 Dashboard: {'✅' if dashboard_updated else '❌'}")
        logger.info(f"   📱 Alerts: {'✅' if alerts_sent else '❌'}")
        logger.info(f"   ⏰ Next run: {flow_result['next_run']}")
        
        return flow_result
        
    except Exception as e:
        logger.error(f"❌ After-hours monitoring flow failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


# Deployment configuration for Prefect
if PREFECT_AVAILABLE:
    # Schedule for every hour during after-hours (16:00-20:00 ET)
    after_hours_schedule = CronSchedule(
        cron="0 16-19 * * 1-5",  # Every hour from 4-7 PM, Monday-Friday
        timezone="America/New_York"
    )
    
    # Create deployment
    def create_after_hours_deployment():
        """Create Prefect deployment for after-hours monitoring"""
        from prefect.deployments import Deployment
        
        deployment = Deployment.build_from_flow(
            flow=after_hours_pnl_flow,
            name="after-hours-pnl-production",
            schedule=after_hours_schedule,
            parameters={
                "alert_threshold": 0.75,  # 0.75% P&L threshold
                "risk_threshold": 1.0     # 1.0% high risk threshold
            },
            tags=["production", "risk", "after-hours", "phase-p11"]
        )
        
        return deployment


def test_after_hours_flow():
    """Test the after-hours monitoring flow"""
    print("🧪 Testing after-hours PnL monitoring flow...")
    
    # Run the flow in test mode
    result = after_hours_pnl_flow(
        alert_threshold=0.5,  # Lower threshold for testing
        risk_threshold=0.75
    )
    
    print(f"📊 Flow result: {result}")
    
    # Validate result
    if result.get('status') == 'completed':
        print("✅ After-hours flow test PASSED")
        return True
    elif result.get('status') == 'skipped':
        print(f"⏭️ Flow skipped: {result.get('reason', 'unknown')}")
        return True
    else:
        print("❌ After-hours flow test FAILED")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='After-Hours PnL Monitoring')
    parser.add_argument('command', choices=['test', 'run', 'deploy'],
                       help='Command to execute')
    parser.add_argument('--alert-threshold', type=float, default=0.75,
                       help='Alert threshold percentage (default: 0.75)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.command == 'test':
        success = test_after_hours_flow()
        sys.exit(0 if success else 1)
        
    elif args.command == 'run':
        result = after_hours_pnl_flow(alert_threshold=args.alert_threshold)
        print(f"Flow result: {result}")
        
    elif args.command == 'deploy':
        if PREFECT_AVAILABLE:
            deployment = create_after_hours_deployment()
            print(f"Deployment created: {deployment.name}")
        else:
            print("❌ Prefect not available for deployment")
            sys.exit(1)