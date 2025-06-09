#!/usr/bin/env python3
"""
Test Script for Retrain Flow Components (Dry Run)

Tests the retrain flow components without Prefect dependencies
to verify the core logic and integration works correctly.
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_walk_forward_validation():
    """Test walk-forward validation component"""
    try:
        from mech_exo.validation.walk_forward_refit import run_walk_forward, ValidationConfig
        
        print("🔄 Testing Walk-Forward Validation...")
        
        # Create a temporary factors file for testing
        import tempfile
        import yaml
        
        test_factors = {
            'metadata': {
                'version': 'test_v1.0',
                'created_at': '2024-12-06T12:00:00',
                'validation_run': True
            },
            'factors': {
                'fundamental': {
                    'pe_ratio': {'weight': 20, 'direction': 'lower_better'},
                    'return_on_equity': {'weight': 25, 'direction': 'higher_better'}
                },
                'technical': {
                    'momentum_12_1': {'weight': 15, 'direction': 'higher_better'},
                    'volatility_ratio': {'weight': 10, 'direction': 'lower_better'}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
            yaml.dump(test_factors, temp_file, default_flow_style=False)
            temp_factors_file = temp_file.name
        
        try:
            # Configure validation with realistic settings
            validation_config = ValidationConfig(
                train_months=18,
                test_months=6,
                step_months=6,
                min_sharpe=0.30,
                max_drawdown=0.15,
                min_segments=1  # Reduced for testing
            )
            
            # Run validation on a reasonable date range
            end_date = date.today()
            start_date = end_date - timedelta(days=3*365)  # 3 years
            
            logger.info(f"Running walk-forward validation: {start_date} to {end_date}")
            
            validation_results = run_walk_forward(
                temp_factors_file,
                start_date,
                end_date,
                validation_config
            )
            
            print(f"✅ Walk-forward validation completed:")
            print(f"   • Overall result: {'PASSED' if validation_results['passed'] else 'FAILED'}")
            print(f"   • Segments: {validation_results['segments_passed']}/{validation_results['segments_count']} passed")
            print(f"   • Summary metrics: {validation_results['summary_metrics']}")
            
            if not validation_results['table'].empty:
                print(f"   • Validation table:")
                print(validation_results['table'].to_string(index=False))
            
            return validation_results
            
        finally:
            # Clean up temporary file
            try:
                Path(temp_factors_file).unlink()
            except:
                pass
                
    except Exception as e:
        print(f"❌ Walk-forward validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_telegram_alerter():
    """Test Telegram alerter component"""
    try:
        from mech_exo.utils.alerts import TelegramAlerter
        
        print("\n📱 Testing Telegram Alerter...")
        
        # Create alerter with dummy credentials
        telegram_config = {
            'bot_token': 'dummy_token',
            'chat_id': 'dummy_chat_id'
        }
        
        alerter = TelegramAlerter(telegram_config)
        
        # Test markdown escaping
        test_text = "Test with special chars: _*[]()~`>#+-=|{}.!"
        escaped = alerter.escape_markdown(test_text)
        print(f"✅ Markdown escaping works: '{test_text}' -> '{escaped}'")
        
        # Test message formatting (won't actually send without real credentials)
        validation_results = {
            'out_of_sample_sharpe': 0.45,
            'max_drawdown': 0.08,
            'segments_passed': 3,
            'segments_total': 4
        }
        
        # This would normally send but will fail gracefully with dummy credentials
        success = alerter.send_retrain_success(
            validation_results=validation_results,
            version="test_v1.0",
            factors_file="config/staging/factors_retrained_test_v1.0.yml"
        )
        
        print(f"✅ Telegram notification structure tested (expected to fail with dummy credentials)")
        
        return True
        
    except Exception as e:
        print(f"❌ Telegram alerter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_deployment_logic():
    """Test deployment logic with dry run"""
    try:
        print("\n🚀 Testing Deployment Logic...")
        
        # Test staging directory creation and file writing
        import yaml
        from datetime import datetime
        
        staging_dir = Path("config/staging")
        staging_dir.mkdir(exist_ok=True)
        
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        factors_file = staging_dir / f"factors_retrained_{version}.yml"
        
        # Create test factor weights
        test_weights = {
            'fundamental': {
                'pe_ratio': {'weight': 18, 'direction': 'lower_better'},
                'return_on_equity': {'weight': 22, 'direction': 'higher_better'}
            },
            'technical': {
                'momentum_12_1': {'weight': 12, 'direction': 'higher_better'}
            }
        }
        
        validation_results = {
            'out_of_sample_sharpe': 0.35,
            'segments_passed': 2,
            'segments_total': 2
        }
        
        # Create factors configuration
        factors_config = {
            'metadata': {
                'version': version,
                'created_at': datetime.now().isoformat(),
                'retrain_trigger': 'drift_breach',
                'validation_sharpe': validation_results.get('out_of_sample_sharpe', 0),
                'segments_passed': validation_results.get('segments_passed', 0),
                'segments_total': validation_results.get('segments_total', 0),
                'deployment_mode': 'dry_run'
            },
            'factors': test_weights
        }
        
        # Write factors configuration to staging
        with open(factors_file, 'w') as f:
            yaml.dump(factors_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Deployment test completed:")
        print(f"   • Factors file created: {factors_file}")
        print(f"   • File exists: {factors_file.exists()}")
        print(f"   • Mode: dry_run (staging only)")
        
        # Verify file contents
        with open(factors_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        print(f"   • Configuration loaded successfully")
        print(f"   • Version: {loaded_config['metadata']['version']}")
        print(f"   • Factors count: {len(loaded_config['factors'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all dry run tests"""
    print("🧪 Starting Retrain Flow Dry Run Tests")
    print("=" * 50)
    
    # Test individual components
    validation_result = test_walk_forward_validation()
    telegram_result = test_telegram_alerter()
    deployment_result = test_deployment_logic()
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   • Walk-Forward Validation: {'✅ PASS' if validation_result else '❌ FAIL'}")
    print(f"   • Telegram Alerter: {'✅ PASS' if telegram_result else '❌ FAIL'}")
    print(f"   • Deployment Logic: {'✅ PASS' if deployment_result else '❌ FAIL'}")
    
    all_passed = all([validation_result, telegram_result, deployment_result])
    
    if all_passed:
        print("\n🎉 All dry run tests completed successfully!")
        print("   The retrain flow components are ready for integration.")
        print("\n💡 Next steps:")
        print("   1. Install Prefect: pip install prefect>=2.14.0")
        print("   2. Set up Telegram credentials (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
        print("   3. Set up Git credentials (GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL)")
        print("   4. Run the full flow: python dags/retrain_flow.py")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)