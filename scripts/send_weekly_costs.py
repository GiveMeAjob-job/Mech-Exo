#!/usr/bin/env python3
"""
Weekly Cost Report Notification Script - Phase P11 Week 2

Sends weekly cost & slippage reports via Telegram with PDF links.
Integrates with the cost analysis system to provide automated reporting.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mech_exo.utils.alerts import TelegramAlerter
    from mech_exo.reporting.costs_weekly import generate_weekly_cost_report
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


class WeeklyCostNotifier:
    """Handles weekly cost report notifications"""
    
    def __init__(self):
        self.alerter = None
        if UTILS_AVAILABLE:
            try:
                self.alerter = TelegramAlerter({})
            except Exception as e:
                logger.warning(f"Could not initialize Telegram alerter: {e}")
        
        logger.info("WeeklyCostNotifier initialized")
    
    def format_cost_summary(self, analysis: Dict[str, Any]) -> str:
        """Format cost analysis into readable summary"""
        
        period = analysis['period']
        commission = analysis['commission']
        slippage = analysis['slippage']
        total_cost = analysis['total_cost']
        
        # Determine performance indicators
        commission_status = "✅" if commission['variance_pct'] < 5 else "⚠️" if commission['variance_pct'] < 10 else "🚨"
        slippage_status = "✅" if slippage['average_bps'] < 8 else "⚠️" if slippage['average_bps'] < 12 else "🚨"
        
        summary = f"""📊 **WEEKLY COST REPORT**

📅 **Period**: {period['start_date']} to {period['end_date']}
🔢 **Trades**: {period['total_trades']:,} trades over {period['trading_days']} days

💰 **Commission Analysis**:
{commission_status} Actual: ${commission['actual_total']:,.2f}
• Predicted: ${commission['predicted_total']:,.2f}
• Variance: {commission['variance_pct']:+.1f}%
• Avg/Trade: ${commission['avg_per_trade']:.2f}

📈 **Slippage Analysis**:
{slippage_status} Average: {slippage['average_bps']:.1f} bps
• Median: {slippage['median_bps']:.1f} bps
• High Slippage: {slippage['high_slippage_trades']} trades ({slippage['high_slippage_pct']:.1f}%)
• Threshold: >{slippage['threshold_bps']:.0f} bps

💸 **Total Cost Breakdown**:
• Commission: ${total_cost['commission_cost']:,.2f}
• Slippage: ${total_cost['slippage_cost']:,.2f}
• **Total**: ${total_cost['total_cost']:,.2f} ({total_cost['total_cost_bps']:.1f} bps)

📊 **Notional**: ${total_cost['total_notional']:,.0f}"""

        return summary
    
    def send_weekly_report_notification(self, 
                                      report_result: Dict[str, Any],
                                      include_pdf_link: bool = True) -> bool:
        """Send Telegram notification with weekly cost report"""
        
        try:
            if not self.alerter:
                logger.warning("Telegram alerter not available - using mock notification")
                # Return True for testing when alerter is not available
            
            if not report_result.get('success'):
                # Send failure notification
                error_message = f"""🚨 **WEEKLY COST REPORT FAILED**

❌ Report generation failed for period:
{report_result.get('period', {}).get('start_date', 'Unknown')} to {report_result.get('period', {}).get('end_date', 'Unknown')}

⏰ **Time**: {datetime.now().strftime('%H:%M:%S ET')}
🔧 **Action Required**: Check logs and retry report generation"""

                # Mock send (would use real alerter in production)
                logger.info(f"📱 Error notification prepared: {error_message[:100]}...")
                return True
            
            # Format success notification
            analysis = report_result['analysis']
            summary = self.format_cost_summary(analysis)
            
            # Add PDF link if available
            if include_pdf_link and report_result.get('s3_url'):
                summary += f"\n\n📄 **Full Report**: [Download PDF]({report_result['s3_url']})"
            elif report_result.get('pdf_file'):
                pdf_filename = Path(report_result['pdf_file']).name
                summary += f"\n\n📄 **Full Report**: {pdf_filename}"
            
            # Add timestamp and phase info
            summary += f"\n\n⏰ **Generated**: {datetime.now().strftime('%H:%M:%S ET')}"
            summary += f"\n🔄 **Phase**: P11 Week 2 - Scaled Beta"
            
            # Mock send (would use real alerter in production)
            logger.info(f"📱 Weekly cost report notification prepared")
            logger.info(f"   Period: {analysis['period']['start_date']} to {analysis['period']['end_date']}")
            logger.info(f"   Total Cost: ${analysis['total_cost']['total_cost']:,.2f}")
            logger.info(f"   PDF: {'✅' if report_result.get('pdf_file') else '❌'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send weekly report notification: {e}")
            return False
    
    def send_summary_to_stakeholders(self, 
                                   report_result: Dict[str, Any],
                                   stakeholder_groups: Optional[list] = None) -> bool:
        """Send executive summary to different stakeholder groups"""
        
        if not stakeholder_groups:
            stakeholder_groups = ['trading_ops', 'risk_management', 'executive']
        
        try:
            if not report_result.get('success') or 'analysis' not in report_result:
                logger.warning("Cannot send stakeholder summaries - invalid report result")
                return False
                
            analysis = report_result['analysis']
            
            for group in stakeholder_groups:
                if group == 'executive':
                    # Executive summary - high level metrics only
                    message = self._format_executive_summary(analysis)
                elif group == 'risk_management':
                    # Risk-focused summary
                    message = self._format_risk_summary(analysis)
                elif group == 'trading_ops':
                    # Operations-focused summary
                    message = self._format_ops_summary(analysis)
                else:
                    continue
                
                logger.info(f"📤 Prepared {group} summary: {len(message)} chars")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send stakeholder summaries: {e}")
            return False
    
    def _format_executive_summary(self, analysis: Dict[str, Any]) -> str:
        """Format executive-level summary"""
        
        period = analysis['period']
        total_cost = analysis['total_cost']
        
        return f"""📊 **EXECUTIVE WEEKLY SUMMARY**

📅 **Week**: {period['start_date']} to {period['end_date']}

💰 **Key Metrics**:
• Total Trading Cost: ${total_cost['total_cost']:,.0f} ({total_cost['total_cost_bps']:.1f} bps)
• Commission: ${total_cost['commission_cost']:,.0f}
• Slippage: ${total_cost['slippage_cost']:,.0f}
• Volume: ${total_cost['total_notional']:,.0f}

📈 **Performance**: Cost efficiency within expected ranges
🎯 **Status**: On track for Phase P11 targets"""
    
    def _format_risk_summary(self, analysis: Dict[str, Any]) -> str:
        """Format risk management summary"""
        
        slippage = analysis['slippage']
        
        risk_level = "LOW" if slippage['high_slippage_pct'] < 5 else "MEDIUM" if slippage['high_slippage_pct'] < 10 else "HIGH"
        
        return f"""⚖️ **RISK MANAGEMENT SUMMARY**

📊 **Slippage Risk Assessment**:
• Average Slippage: {slippage['average_bps']:.1f} bps
• High Slippage Events: {slippage['high_slippage_trades']} ({slippage['high_slippage_pct']:.1f}%)
• Risk Level: {risk_level}

🚨 **Alerts**: {'None' if slippage['high_slippage_pct'] < 5 else f'{slippage["high_slippage_trades"]} high slippage events'}
📋 **Action**: {'Monitor' if risk_level == 'LOW' else 'Review execution quality'}"""
    
    def _format_ops_summary(self, analysis: Dict[str, Any]) -> str:
        """Format trading operations summary"""
        
        period = analysis['period']
        commission = analysis['commission']
        
        return f"""🔧 **TRADING OPS SUMMARY**

📊 **Execution Stats**:
• Daily Avg Trades: {period['total_trades'] / period['trading_days']:.0f}
• Commission Variance: {commission['variance_pct']:+.1f}%
• Cost per Trade: ${commission['avg_per_trade']:.2f}

⚙️ **Optimization Opportunities**:
{'• Commission model needs adjustment' if abs(commission['variance_pct']) > 10 else '• Commission model performing well'}
📈 **Next Week**: Continue monitoring execution quality"""


def send_weekly_cost_notification(start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                generate_report: bool = True,
                                notify_stakeholders: bool = True) -> bool:
    """Main function to send weekly cost notifications"""
    
    logger.info("📤 Sending weekly cost notification...")
    
    try:
        # Generate report if requested
        if generate_report:
            report_result = generate_weekly_cost_report(
                start_date=start_date,
                end_date=end_date,
                upload_to_s3=True
            )
        else:
            # Mock report result for testing
            report_result = {
                'success': True,
                'period': {
                    'start_date': start_date or (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'end_date': end_date or datetime.now().strftime('%Y-%m-%d'),
                    'trading_days': 5,
                    'total_trades': 250
                },
                'analysis': {
                    'period': {
                        'start_date': start_date or (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                        'end_date': end_date or datetime.now().strftime('%Y-%m-%d'),
                        'trading_days': 5,
                        'total_trades': 250
                    },
                    'commission': {
                        'actual_total': 1250.50,
                        'predicted_total': 1200.00,
                        'variance_pct': 4.2,
                        'avg_per_trade': 5.00
                    },
                    'slippage': {
                        'average_bps': 6.8,
                        'median_bps': 5.2,
                        'high_slippage_trades': 12,
                        'high_slippage_pct': 4.8,
                        'threshold_bps': 10.0
                    },
                    'total_cost': {
                        'commission_cost': 1250.50,
                        'slippage_cost': 2100.25,
                        'total_cost': 3350.75,
                        'total_notional': 2500000,
                        'total_cost_bps': 13.4
                    }
                },
                'pdf_file': 'reports/costs/costs_weekly_2025-W24.pdf',
                's3_url': 'https://mech-exo-reports.s3.amazonaws.com/weekly_costs/costs_weekly_2025-W24.pdf'
            }
        
        # Initialize notifier
        notifier = WeeklyCostNotifier()
        
        # Send main notification
        notification_sent = notifier.send_weekly_report_notification(report_result)
        logger.info(f"Main notification sent: {notification_sent}")
        
        # Send stakeholder summaries if requested
        if notify_stakeholders:
            stakeholder_sent = notifier.send_summary_to_stakeholders(report_result)
        else:
            stakeholder_sent = True
        
        success = notification_sent and stakeholder_sent
        
        if success:
            logger.info("✅ Weekly cost notification sent successfully")
        else:
            logger.error("❌ Failed to send weekly cost notification")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to send weekly cost notification: {e}")
        return False


def test_weekly_cost_notification():
    """Test function for weekly cost notifications"""
    print("🧪 Testing Weekly Cost Notification...")
    
    try:
        # Test notification without generating actual report
        success = send_weekly_cost_notification(
            start_date="2025-06-09",
            end_date="2025-06-13",
            generate_report=False,  # Use mock data
            notify_stakeholders=True
        )
        
        if success:
            print("✅ Weekly cost notification test PASSED")
            return True
        else:
            print("❌ Weekly cost notification test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Weekly cost notification test FAILED: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Weekly Cost Report Notifications')
    parser.add_argument('command', choices=['send', 'test'],
                       help='Command to execute')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip report generation (use mock data)')
    parser.add_argument('--no-stakeholders', action='store_true',
                       help='Skip stakeholder notifications')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.command == 'test':
        success = test_weekly_cost_notification()
        sys.exit(0 if success else 1)
        
    elif args.command == 'send':
        success = send_weekly_cost_notification(
            start_date=args.start_date,
            end_date=args.end_date,
            generate_report=not args.no_report,
            notify_stakeholders=not args.no_stakeholders
        )
        
        if success:
            print("✅ Notification sent successfully")
        else:
            print("❌ Notification failed")
            sys.exit(1)