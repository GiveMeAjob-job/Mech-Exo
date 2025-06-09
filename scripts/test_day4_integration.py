#!/usr/bin/env python3
"""
Day 4 Integration Test - Telegram Alerts for Rapid Alpha Decay

Tests the complete alpha decay monitoring pipeline with Telegram alerts
without requiring Prefect installation.
"""

import os
import sys
from pathlib import Path
from datetime import date, datetime
import yaml

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_day4_integration():
    """Test complete Day 4 integration"""
    print("🧪 Day 4 Integration Test - Alpha Decay Monitoring with Telegram Alerts")
    print("=" * 70)
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Configuration Loading
    print("\n1️⃣ Testing configuration loading...")
    try:
        config_path = project_root / "config" / "decay.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        alert_threshold = config.get('alert_half_life', 7)
        lookback_days = config.get('lookback_days', 730)
        
        print(f"   ✅ Configuration loaded successfully")
        print(f"   • Alert threshold: {alert_threshold} days")
        print(f"   • Lookback days: {lookback_days}")
        print(f"   • Schedule: {config.get('schedule_cron')}")
        
        success_count += 1
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
    
    # Test 2: Factor Decay Data Generation
    print("\n2️⃣ Testing factor decay data generation...")
    try:
        # Run the factor decay data generation test
        from mech_exo.datasource.storage import DataStorage
        from mech_exo.research.alpha_decay import AlphaDecayEngine
        
        # Create synthetic data for testing
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        dates = pd.date_range(start=date.today().replace(month=1, day=1), 
                             end=date.today(), freq='B')
        
        # Generate factors with predictable correlation patterns for testing
        factors_data = {}
        returns_data = {}
        
        # Fast decaying factor (half-life ~5 days)
        # Create factor values that start correlated and decay
        base_factor = np.random.normal(0, 1, len(dates))
        decay_fast = np.exp(-np.arange(len(dates)) / 5)  # 5-day half-life
        fast_factor = base_factor * decay_fast
        factors_data['fast_momentum'] = pd.Series(fast_factor, index=dates)
        
        # Generate returns correlated with fast factor early, then random
        fast_returns = fast_factor * 0.01 + np.random.normal(0, 0.02, len(dates))
        returns_data['fast_momentum'] = pd.Series(fast_returns, index=dates)
        
        # Medium decaying factor (half-life ~20 days)
        medium_factor = base_factor * np.exp(-np.arange(len(dates)) / 20)
        factors_data['medium_value'] = pd.Series(medium_factor, index=dates)
        
        medium_returns = medium_factor * 0.01 + np.random.normal(0, 0.02, len(dates))
        returns_data['medium_value'] = pd.Series(medium_returns, index=dates)
        
        # Slow decaying factor (half-life ~40 days)
        slow_factor = base_factor * np.exp(-np.arange(len(dates)) / 40)
        factors_data['slow_quality'] = pd.Series(slow_factor, index=dates)
        
        slow_returns = slow_factor * 0.01 + np.random.normal(0, 0.02, len(dates))
        returns_data['slow_quality'] = pd.Series(slow_returns, index=dates)
        
        print(f"   ✅ Generated {len(factors_data)} synthetic factors")
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ Factor data generation test failed: {e}")
    
    # Test 3: Alpha Decay Calculation
    print("\n3️⃣ Testing alpha decay calculation...")
    try:
        decay_engine = AlphaDecayEngine(window=120, min_periods=30)
        decay_results = []
        
        for factor_name in factors_data.keys():
            factor_series = factors_data[factor_name]
            returns_series = returns_data[factor_name]
            
            # Align series
            aligned_df = pd.DataFrame({
                'factor': factor_series,
                'returns': returns_series
            }).dropna()
            
            if len(aligned_df) >= 30:
                decay_metrics = decay_engine.calc_half_life(
                    aligned_df['factor'],
                    aligned_df['returns']
                )
                
                decay_metrics.update({
                    'factor_name': factor_name,
                    'calculation_date': datetime.now(),
                    'data_points': len(aligned_df)
                })
                
                decay_results.append(decay_metrics)
        
        print(f"   ✅ Calculated decay for {len(decay_results)} factors")
        for metrics in decay_results:
            half_life = metrics.get('half_life', 0)
            ic = metrics.get('latest_ic', 0)
            print(f"   • {metrics['factor_name']}: {half_life:.1f}d half-life, IC: {ic:.3f}")
        
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ Alpha decay calculation test failed: {e}")
    
    # Test 4: Database Storage
    print("\n4️⃣ Testing database storage...")
    try:
        storage = DataStorage()
        
        # Create table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS factor_decay (
            date DATE NOT NULL,
            factor VARCHAR NOT NULL,
            half_life DECIMAL(8,2),
            latest_ic DECIMAL(8,4),
            ic_observations INTEGER,
            ic_mean DECIMAL(8,4),
            ic_std DECIMAL(8,4),
            ic_trend DECIMAL(10,6),
            data_points INTEGER,
            calculation_timestamp TIMESTAMP,
            status VARCHAR,
            PRIMARY KEY (date, factor)
        )
        """
        
        storage.conn.execute(create_table_sql)
        
        # Store test data
        calculation_date = date.today()
        
        for metrics in decay_results:
            insert_sql = """
            INSERT OR REPLACE INTO factor_decay 
            (date, factor, half_life, latest_ic, ic_observations, ic_mean, ic_std, 
             ic_trend, data_points, calculation_timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # Handle NaN values by converting to None for database storage
            import math
            
            def safe_float(value):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    return None
                return value
            
            row_data = [
                calculation_date,
                metrics['factor_name'],
                safe_float(metrics.get('half_life')),
                safe_float(metrics.get('latest_ic')),
                metrics.get('ic_observations', 0),
                safe_float(metrics.get('ic_mean')),
                safe_float(metrics.get('ic_std')),
                safe_float(metrics.get('ic_trend')),
                metrics.get('data_points', 0),
                metrics['calculation_date'],
                metrics.get('status', 'calculated')
            ]
            
            storage.conn.execute(insert_sql, row_data)
        
        # Verify storage
        count_sql = "SELECT COUNT(*) as count FROM factor_decay WHERE date = ?"
        result = storage.conn.execute(count_sql, [calculation_date]).fetchdf()
        stored_count = result.iloc[0]['count'] if not result.empty else 0
        
        storage.close()
        
        print(f"   ✅ Stored {stored_count} decay metrics to database")
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ Database storage test failed: {e}")
    
    # Test 5: Telegram Alert Logic
    print("\n5️⃣ Testing Telegram alert logic...")
    try:
        from mech_exo.utils.alerts import TelegramAlerter
        
        # Check for rapid decay factors
        alert_threshold = config.get('alert_half_life', 7)
        rapid_decay_factors = []
        
        for metrics in decay_results:
            half_life = metrics.get('half_life')
            if half_life is not None and half_life < alert_threshold:
                rapid_decay_factors.append({
                    'factor': metrics['factor_name'],
                    'half_life': half_life,
                    'latest_ic': metrics.get('latest_ic', 0)
                })
        
        print(f"   • Found {len(rapid_decay_factors)} factors with half-life < {alert_threshold} days")
        
        if rapid_decay_factors:
            # Test Telegram alerter setup
            telegram_config = {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', 'test_token'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID', 'test_chat')
            }
            
            alerter = TelegramAlerter(telegram_config)
            
            # Format alert message
            alert_message = "⚠️ *Alpha\\\\-decay Alert*\\n\\n"
            
            for factor_info in rapid_decay_factors:
                factor_name = alerter.escape_markdown(factor_info['factor'])
                half_life = factor_info['half_life']
                ic = factor_info['latest_ic']
                
                alert_message += f"📉 *{factor_name}*: half\\\\-life {half_life:.1f}d \\\\(<{alert_threshold}\\\\)\\n"
                alert_message += f"    Latest IC: {ic:.3f}\\n\\n"
            
            alert_message += f"🔍 *Threshold*: {alert_threshold} days\\n"
            today_str = date.today().strftime('%Y-%m-%d').replace('-', '\\\\-')
            alert_message += f"📅 *Date*: {today_str}"
            
            print(f"   • Alert message formatted successfully")
            print(f"   • Would alert on factors: {[f['factor'] for f in rapid_decay_factors]}")
            
            # Check if real credentials are available
            has_real_creds = (telegram_config['bot_token'] != 'test_token' and 
                            telegram_config['chat_id'] != 'test_chat')
            
            if has_real_creds:
                print(f"   • Real Telegram credentials detected - alerts ready to send")
            else:
                print(f"   • Test credentials used - set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID for live alerts")
        
        else:
            print(f"   • No factors meet alert criteria (all have half-life >= {alert_threshold} days)")
        
        print(f"   ✅ Telegram alert logic test completed")
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ Telegram alert test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Day 4 Integration Test Summary:")
    print(f"   • Tests Passed: {success_count}/{total_tests}")
    print(f"   • Success Rate: {success_count/total_tests*100:.0f}%")
    
    if success_count == total_tests:
        print("\n🎉 All Day 4 components are working correctly!")
        print("\n💡 Next steps for production deployment:")
        print("   1. Set TELEGRAM_BOT_TOKEN environment variable")
        print("   2. Set TELEGRAM_CHAT_ID environment variable") 
        print("   3. Install Prefect: pip install prefect>=2.14.0")
        print("   4. Schedule flow: prefect deployment create alpha-decay-flow")
        print("   5. Verify dashboard Factor Health tab displays data")
        
        return True
    else:
        print(f"\n❌ {total_tests - success_count} tests failed. Please review errors above.")
        return False


if __name__ == "__main__":
    success = test_day4_integration()
    sys.exit(0 if success else 1)