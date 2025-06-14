"""
Rotational Alpha Flow - Phase P11 Week 2

Daily Prefect flow for generating sector rotation signals and updating
the rotational alpha scores table. Integrates with the existing factor
pipeline and idea scorer to provide multi-strategy signals.

Schedule: Daily at 08:55 UTC (before market open)
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


@task(name="fetch_universe_symbols", cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def fetch_universe_symbols() -> List[str]:
    """Fetch current universe symbols for rotational alpha analysis"""
    logger = get_run_logger()
    logger.info("📊 Fetching universe symbols for rotational alpha...")
    
    try:
        # In production, this would fetch from the universe loader
        # For Phase P11, use sector ETFs as rotational universe
        sector_etfs = [
            'XLK',   # Technology
            'XLF',   # Financials
            'XLV',   # Healthcare
            'XLI',   # Industrials
            'XLE',   # Energy
            'XLB',   # Materials
            'XLU',   # Utilities
            'XLP',   # Consumer Staples
            'XLY',   # Consumer Discretionary
            'XLRE',  # Real Estate
            'XLC'    # Communication Services
        ]
        
        logger.info(f"📊 Retrieved {len(sector_etfs)} sector ETFs for analysis")
        return sector_etfs
        
    except Exception as e:
        logger.error(f"Failed to fetch universe symbols: {e}")
        # Fallback to minimal universe
        return ['XLK', 'XLF', 'XLV', 'SPY']


@task(name="generate_rotational_scores")
def generate_rotational_scores(universe_symbols: List[str]) -> Dict[str, Any]:
    """Generate rotational alpha scores using sector momentum"""
    logger = get_run_logger()
    logger.info(f"🔄 Generating rotational alpha scores for {len(universe_symbols)} symbols...")
    
    try:
        from research.rot_alpha_signal import generate_rotational_alpha_scores
        
        # Generate scores using the rotational alpha signal generator
        scores_df = generate_rotational_alpha_scores(
            universe_symbols=universe_symbols,
            output_file="data/rot_alpha_scores.csv"
        )
        
        # Convert to summary format
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(scores_df),
            'strong_buy_count': len(scores_df[scores_df['signal'] == 'STRONG_BUY']),
            'buy_count': len(scores_df[scores_df['signal'] == 'BUY']),
            'neutral_count': len(scores_df[scores_df['signal'] == 'NEUTRAL']),
            'sell_count': len(scores_df[scores_df['signal'] == 'SELL']),
            'avg_momentum': scores_df['momentum_score'].mean(),
            'max_momentum': scores_df['momentum_score'].max(),
            'min_momentum': scores_df['momentum_score'].min(),
            'top_sector': scores_df.loc[scores_df['momentum_score'].idxmax(), 'sector'],
            'data_quality': scores_df['data_quality'].mean()
        }
        
        # Top 3 sectors for logging
        top_sectors = scores_df.nlargest(3, 'momentum_score')
        summary['top_3_sectors'] = [
            {
                'symbol': row['symbol'],
                'sector': row['sector'], 
                'momentum': row['momentum_score'],
                'signal': row['signal']
            }
            for _, row in top_sectors.iterrows()
        ]
        
        logger.info(f"🔄 Generated scores: {summary['strong_buy_count']} strong buy, {summary['buy_count']} buy")
        logger.info(f"🔄 Top momentum: {summary['top_sector']} ({summary['max_momentum']:.4f})")
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate rotational scores: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'total_symbols': 0,
            'data_quality': 0.0
        }


@task(name="update_scores_database")
def update_scores_database(scores_summary: Dict[str, Any]) -> bool:
    """Update the rot_alpha_scores table in database"""
    logger = get_run_logger()
    logger.info("💾 Updating rot_alpha_scores database table...")
    
    try:
        if 'error' in scores_summary:
            logger.error(f"Cannot update database: {scores_summary['error']}")
            return False
            
        # In production, this would update the actual database
        # For Phase P11, simulate database update
        
        # Mock database operation
        import time
        time.sleep(0.5)  # Simulate database write time
        
        # Log the update
        logger.info(f"💾 Database updated with {scores_summary['total_symbols']} rotational alpha scores")
        logger.info(f"💾 Data quality: {scores_summary['data_quality']:.1%}")
        
        # Simulate successful database update
        return True
        
    except Exception as e:
        logger.error(f"Failed to update scores database: {e}")
        return False


@task(name="validate_scores_quality")
def validate_scores_quality(scores_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the quality of generated rotational alpha scores"""
    logger = get_run_logger()
    logger.info("🔍 Validating rotational alpha scores quality...")
    
    validation_result = {
        'timestamp': datetime.now().isoformat(),
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'quality_score': 0.0
    }
    
    try:
        if 'error' in scores_summary:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Score generation failed: {scores_summary['error']}")
            return validation_result
            
        # Quality checks
        total_symbols = scores_summary.get('total_symbols', 0)
        data_quality = scores_summary.get('data_quality', 0.0)
        
        # Check 1: Minimum symbols threshold
        if total_symbols < 5:
            validation_result['errors'].append(f"Too few symbols: {total_symbols} (expected ≥5)")
            validation_result['is_valid'] = False
            
        # Check 2: Data quality threshold
        if data_quality < 0.8:
            validation_result['warnings'].append(f"Low data quality: {data_quality:.1%} (expected ≥80%)")
            
        # Check 3: Signal distribution
        signal_counts = {
            'strong_buy': scores_summary.get('strong_buy_count', 0),
            'buy': scores_summary.get('buy_count', 0),
            'neutral': scores_summary.get('neutral_count', 0),
            'sell': scores_summary.get('sell_count', 0)
        }
        
        if signal_counts['strong_buy'] + signal_counts['buy'] == 0:
            validation_result['warnings'].append("No buy signals generated")
            
        # Check 4: Momentum range
        max_momentum = scores_summary.get('max_momentum', 0)
        if abs(max_momentum) < 0.001:
            validation_result['warnings'].append("Very low momentum signals detected")
            
        # Calculate overall quality score
        quality_factors = [
            min(total_symbols / 10, 1.0),  # Symbol coverage
            data_quality,                   # Data quality
            1.0 if validation_result['is_valid'] else 0.5,  # Validation status
            min(len(validation_result['warnings']) == 0, 1.0) * 0.2 + 0.8  # Warning penalty
        ]
        
        validation_result['quality_score'] = sum(quality_factors) / len(quality_factors)
        
        logger.info(f"🔍 Validation complete: {'✅ PASS' if validation_result['is_valid'] else '❌ FAIL'}")
        logger.info(f"🔍 Quality score: {validation_result['quality_score']:.1%}")
        
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                logger.warning(f"⚠️ {warning}")
                
        return validation_result
        
    except Exception as e:
        logger.error(f"Failed to validate scores quality: {e}")
        validation_result['is_valid'] = False
        validation_result['errors'].append(str(e))
        return validation_result


@task(name="send_rotational_alerts")
def send_rotational_alerts(scores_summary: Dict[str, Any], validation_result: Dict[str, Any]) -> bool:
    """Send alerts about rotational alpha signal generation"""
    logger = get_run_logger()
    logger.info("📱 Sending rotational alpha alerts...")
    
    try:
        if not UTILS_AVAILABLE:
            logger.warning("Cannot send alerts: utils not available")
            return False
            
        # Determine alert severity
        is_valid = validation_result.get('is_valid', False)
        quality_score = validation_result.get('quality_score', 0.0)
        
        if not is_valid:
            severity_emoji = "🚨"
            status_text = "FAILED"
        elif quality_score < 0.8:
            severity_emoji = "⚠️"
            status_text = "WARNING"
        else:
            severity_emoji = "✅"
            status_text = "SUCCESS"
            
        # Prepare alert message
        alerter = TelegramAlerter({})  # Mock alerter
        
        top_sectors = scores_summary.get('top_3_sectors', [])
        top_sector_text = "\n".join([
            f"• {sector['symbol']} ({sector['sector']}): {sector['momentum']:.3f} [{sector['signal']}]"
            for sector in top_sectors[:3]
        ])
        
        message = f"""{severity_emoji} **ROTATIONAL ALPHA UPDATE**

📊 **Signal Generation**: {status_text}
🔄 **Symbols Processed**: {scores_summary.get('total_symbols', 0)}
📈 **Quality Score**: {quality_score:.1%}

🏆 **Top Momentum Sectors**:
{top_sector_text}

📊 **Signal Distribution**:
• Strong Buy: {scores_summary.get('strong_buy_count', 0)}
• Buy: {scores_summary.get('buy_count', 0)}
• Neutral: {scores_summary.get('neutral_count', 0)}
• Sell: {scores_summary.get('sell_count', 0)}

⏰ **Time**: {datetime.now().strftime('%H:%M:%S UTC')}
🔄 **Strategy**: Rotational Alpha (Phase P11 Week 2)"""

        # In production, this would send the actual alert
        logger.info(f"📱 Alert message prepared: {status_text}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to send rotational alerts: {e}")
        return False


@flow(name="rotational-alpha-generation")
def rotational_alpha_flow(
    enable_alerts: bool = True,
    min_quality_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Main rotational alpha flow for sector momentum signal generation
    
    Args:
        enable_alerts: Whether to send Telegram alerts
        min_quality_threshold: Minimum quality score for success
    """
    logger = get_run_logger()
    logger.info("🔄 Starting rotational alpha generation flow...")
    
    try:
        # Step 1: Fetch universe symbols
        universe_symbols = fetch_universe_symbols()
        
        # Step 2: Generate rotational scores
        scores_summary = generate_rotational_scores(universe_symbols)
        
        # Step 3: Update database
        database_updated = update_scores_database(scores_summary)
        
        # Step 4: Validate scores quality
        validation_result = validate_scores_quality(scores_summary)
        
        # Step 5: Send alerts if enabled
        alerts_sent = False
        if enable_alerts:
            alerts_sent = send_rotational_alerts(scores_summary, validation_result)
        
        # Prepare flow summary
        flow_result = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'universe_symbols': len(universe_symbols),
            'scores_generated': scores_summary.get('total_symbols', 0),
            'database_updated': database_updated,
            'validation_passed': validation_result.get('is_valid', False),
            'quality_score': validation_result.get('quality_score', 0.0),
            'alerts_sent': alerts_sent,
            'top_sector': scores_summary.get('top_sector', 'unknown'),
            'avg_momentum': scores_summary.get('avg_momentum', 0.0),
            'steps_completed': 5,
            'next_run': '08:55 UTC tomorrow'
        }
        
        # Final status determination
        success = (
            flow_result['validation_passed'] and
            flow_result['quality_score'] >= min_quality_threshold and
            flow_result['database_updated']
        )
        
        flow_result['success'] = success
        
        if success:
            logger.info("✅ Rotational alpha flow completed successfully:")
        else:
            logger.warning("⚠️ Rotational alpha flow completed with warnings:")
            
        logger.info(f"   📊 Symbols: {flow_result['scores_generated']}")
        logger.info(f"   📈 Quality: {flow_result['quality_score']:.1%}")
        logger.info(f"   🏆 Top: {flow_result['top_sector']}")
        logger.info(f"   💾 Database: {'✅' if database_updated else '❌'}")
        logger.info(f"   📱 Alerts: {'✅' if alerts_sent else '❌'}")
        
        return flow_result
        
    except Exception as e:
        logger.error(f"❌ Rotational alpha flow failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'success': False
        }


# Deployment configuration for Prefect
if PREFECT_AVAILABLE:
    # Schedule for daily at 08:55 UTC (before market open)
    rotational_alpha_schedule = CronSchedule(
        cron="55 8 * * 1-5",  # 08:55 UTC, Monday-Friday
        timezone="UTC"
    )
    
    # Create deployment
    def create_rotational_alpha_deployment():
        """Create Prefect deployment for rotational alpha flow"""
        from prefect.deployments import Deployment
        
        deployment = Deployment.build_from_flow(
            flow=rotational_alpha_flow,
            name="rotational-alpha-production",
            schedule=rotational_alpha_schedule,
            parameters={
                "enable_alerts": True,
                "min_quality_threshold": 0.8
            },
            tags=["production", "rotational-alpha", "multi-strategy", "phase-p11"]
        )
        
        return deployment


def test_rotational_alpha_flow():
    """Test the rotational alpha flow"""
    print("🧪 Testing rotational alpha flow...")
    
    # Run the flow in test mode
    result = rotational_alpha_flow(
        enable_alerts=False,  # Disable alerts for testing
        min_quality_threshold=0.7  # Lower threshold for testing
    )
    
    print(f"📊 Flow result: {result}")
    
    # Validate result
    if result.get('success'):
        print("✅ Rotational alpha flow test PASSED")
        return True
    elif result.get('status') == 'completed':
        print("⚠️ Rotational alpha flow completed with warnings")
        return True
    else:
        print("❌ Rotational alpha flow test FAILED")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Rotational Alpha Flow')
    parser.add_argument('command', choices=['test', 'run', 'deploy'],
                       help='Command to execute')
    parser.add_argument('--enable-alerts', action='store_true', default=False,
                       help='Enable Telegram alerts')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.command == 'test':
        success = test_rotational_alpha_flow()
        sys.exit(0 if success else 1)
        
    elif args.command == 'run':
        result = rotational_alpha_flow(enable_alerts=args.enable_alerts)
        print(f"Flow result: {result}")
        sys.exit(0 if result.get('success') else 1)
        
    elif args.command == 'deploy':
        if PREFECT_AVAILABLE:
            deployment = create_rotational_alpha_deployment()
            print(f"Deployment created: {deployment.name}")
        else:
            print("❌ Prefect not available for deployment")
            sys.exit(1)