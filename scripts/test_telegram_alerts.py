#!/usr/bin/env python3
"""
Test Telegram Alert Functionality

Tests the Telegram alerter for alpha decay notifications.
"""

import os
import sys
from pathlib import Path
from datetime import date

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_telegram_alerts():
    """Test Telegram alert functionality"""
    try:
        from mech_exo.utils.alerts import TelegramAlerter
        
        print("🔧 Testing Telegram Alert Configuration...")
        
        # Check environment variables
        telegram_config = {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
        
        bot_status = "Set" if telegram_config['bot_token'] else "Not Set"
        chat_status = "Set" if telegram_config['chat_id'] else "Not Set"
        
        print(f"   • Bot Token: {bot_status}")
        print(f"   • Chat ID: {chat_status}")
        
        if not telegram_config['bot_token'] or not telegram_config['chat_id']:
            print("   • Using dummy config for format testing...")
            telegram_config = {
                'bot_token': 'dummy_token_for_testing',
                'chat_id': 'dummy_chat_id_for_testing'
            }
        
        # Initialize alerter
        alerter = TelegramAlerter(telegram_config)
        print("✅ TelegramAlerter initialized successfully")
        
        # Test markdown escaping
        test_factor = 'momentum_12_1'
        escaped = alerter.escape_markdown(test_factor)
        print(f'   • Markdown escaping test: "{test_factor}" -> "{escaped}"')
        
        # Test alpha decay alert message formatting
        rapid_decay_factors = [
            {'factor': 'momentum_12_1', 'half_life': 5.2, 'latest_ic': 0.045},
            {'factor': 'pe_ratio', 'half_life': 3.8, 'latest_ic': -0.070},
            {'factor': 'return_on_equity', 'half_life': 2.1, 'latest_ic': 0.142}
        ]
        
        alert_threshold = 7.0
        
        # Create alert message exactly as in the flow
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
        
        print("\n📱 Sample alpha decay alert message:")
        print("=" * 60)
        print(alert_message)
        print("=" * 60)
        
        # Test if we can actually send (only if real credentials)
        if telegram_config['bot_token'] != 'dummy_token_for_testing':
            print("\n🚀 Attempting to send test alert...")
            
            test_message = "🧪 *Alpha Decay Test Alert*\\n\\nThis is a test message from the Mech\\-Exo alpha decay monitoring system\\."
            
            success = alerter.send_message(test_message)
            
            if success:
                print("✅ Test alert sent successfully!")
            else:
                print("❌ Failed to send test alert")
        else:
            print("\n💡 To test actual sending, set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        
        print("\n✅ Telegram alert functionality test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Telegram alert test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_telegram_alerts()
    sys.exit(0 if success else 1)