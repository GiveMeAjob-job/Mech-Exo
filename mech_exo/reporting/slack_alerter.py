"""
Slack alerter for daily trading reports and notifications
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from ..utils.config import ConfigManager
from .daily import DailyReport
from .html_renderer import HTMLReportRenderer

logger = logging.getLogger(__name__)


class SlackAlerter:
    """
    Send trading alerts and reports to Slack channels
    
    Features:
    - Daily digest messages
    - Critical alert notifications
    - Performance summary cards
    - Risk violation alerts
    - Customizable message formatting
    """

    def __init__(self, webhook_url: str = None):
        """
        Initialize Slack alerter with webhook configuration
        
        Args:
            webhook_url: Slack webhook URL (if not provided, loads from config)
        """
        self.config_manager = ConfigManager()
        
        if webhook_url:
            self.webhook_url = webhook_url
        else:
            # Load from config/alerts.yml
            alerts_config = self.config_manager.get_alerts_config()
            self.webhook_url = alerts_config.get('slack', {}).get('webhook_url')
            
        if not self.webhook_url:
            raise ValueError("Slack webhook URL not provided and not found in configuration")
            
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Mech-Exo-Trading-Bot/1.0'
        })
        
        logger.info("SlackAlerter initialized successfully")

    def send_daily_digest(self, report: DailyReport) -> bool:
        """
        Send daily trading digest to Slack
        
        Args:
            report: DailyReport instance with trading data
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            summary = report.summary()
            breakdown = report.detailed_breakdown()
            
            # Create Slack message payload
            payload = self._create_daily_digest_payload(summary, breakdown)
            
            # Send to Slack
            response = self.session.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Daily digest sent to Slack for {summary['date']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send daily digest to Slack: {e}")
            return False

    def send_alert(self, title: str, message: str, severity: str = "info", 
                   fields: List[Dict[str, Any]] = None) -> bool:
        """
        Send custom alert to Slack
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            fields: Optional list of field dicts with 'title' and 'value'
            
        Returns:
            bool: True if alert sent successfully
        """
        try:
            payload = self._create_alert_payload(title, message, severity, fields)
            
            response = self.session.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Alert '{title}' sent to Slack with severity {severity}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert to Slack: {e}")
            return False

    def send_risk_violation(self, violation_details: Dict[str, Any]) -> bool:
        """
        Send risk violation alert to Slack
        
        Args:
            violation_details: Dictionary with violation information
            
        Returns:
            bool: True if alert sent successfully
        """
        try:
            title = "🚨 Risk Violation Detected"
            message = f"Risk limit violation detected: {violation_details.get('type', 'Unknown')}"
            
            fields = [
                {
                    "title": "Violation Type",
                    "value": violation_details.get('type', 'Unknown'),
                    "short": True
                },
                {
                    "title": "Current Value", 
                    "value": str(violation_details.get('current_value', 'N/A')),
                    "short": True
                },
                {
                    "title": "Limit",
                    "value": str(violation_details.get('limit', 'N/A')),
                    "short": True
                },
                {
                    "title": "Portfolio NAV",
                    "value": f"${violation_details.get('nav', 0):,.2f}",
                    "short": True
                }
            ]
            
            if violation_details.get('symbol'):
                fields.append({
                    "title": "Symbol",
                    "value": violation_details['symbol'],
                    "short": True
                })
                
            return self.send_alert(title, message, "critical", fields)
            
        except Exception as e:
            logger.error(f"Failed to send risk violation alert: {e}")
            return False

    def send_execution_alert(self, order_id: str, status: str, details: Dict[str, Any]) -> bool:
        """
        Send execution-related alert to Slack
        
        Args:
            order_id: Order ID
            status: Order status (filled, rejected, cancelled)
            details: Order/fill details
            
        Returns:
            bool: True if alert sent successfully
        """
        try:
            emoji_map = {
                'filled': '✅',
                'rejected': '❌', 
                'cancelled': '⚠️',
                'partially_filled': '🔄'
            }
            
            emoji = emoji_map.get(status.lower(), '🔔')
            title = f"{emoji} Order {status.title()}"
            message = f"Order {order_id} has been {status}"
            
            fields = [
                {
                    "title": "Symbol",
                    "value": details.get('symbol', 'Unknown'),
                    "short": True
                },
                {
                    "title": "Quantity", 
                    "value": str(details.get('quantity', 0)),
                    "short": True
                }
            ]
            
            if details.get('price'):
                fields.append({
                    "title": "Price",
                    "value": f"${details['price']:.2f}",
                    "short": True
                })
                
            if details.get('strategy'):
                fields.append({
                    "title": "Strategy",
                    "value": details['strategy'],
                    "short": True
                })
            
            severity = "error" if status == "rejected" else "info"
            return self.send_alert(title, message, severity, fields)
            
        except Exception as e:
            logger.error(f"Failed to send execution alert: {e}")
            return False

    def _create_daily_digest_payload(self, summary: Dict[str, Any], 
                                   breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Create Slack payload for daily digest"""
        
        # Determine overall performance color
        pnl_color = "good" if summary['daily_pnl'] >= 0 else "danger"
        activity_emoji = "📈" if summary['trade_count'] > 0 else "📴"
        
        # Create main attachment
        attachment = {
            "color": pnl_color,
            "title": f"{activity_emoji} Daily Trading Report - {summary['date']}",
            "title_link": f"file://daily_report_{summary['date']}.html",
            "fields": [
                {
                    "title": "Daily P&L",
                    "value": f"${summary['daily_pnl']:,.2f}",
                    "short": True
                },
                {
                    "title": "Trade Count",
                    "value": str(summary['trade_count']),
                    "short": True
                },
                {
                    "title": "Volume",
                    "value": f"${summary['volume']:,.0f}",
                    "short": True
                },
                {
                    "title": "Fees",
                    "value": f"${summary['fees']:,.2f}",
                    "short": True
                }
            ],
            "footer": "Mech-Exo Trading System",
            "ts": int(datetime.now(timezone.utc).timestamp())
        }
        
        # Add execution quality metrics if we have activity
        if summary['trade_count'] > 0:
            attachment["fields"].extend([
                {
                    "title": "Avg Slippage",
                    "value": f"{summary['avg_slippage_bps']:.1f} bps",
                    "short": True
                },
                {
                    "title": "Max Drawdown",
                    "value": f"${summary['max_dd']:,.2f}",
                    "short": True
                }
            ])
            
            # Add top performers
            if breakdown.get('by_symbol'):
                top_symbol = max(breakdown['by_symbol'].items(), 
                               key=lambda x: abs(x[1]['pnl']))
                attachment["fields"].append({
                    "title": "Top Performer",
                    "value": f"{top_symbol[0]}: ${top_symbol[1]['pnl']:,.2f}",
                    "short": False
                })
        
        payload = {
            "username": "Mech-Exo Bot",
            "icon_emoji": ":chart_with_upwards_trend:",
            "attachments": [attachment]
        }
        
        # Add no-activity message if needed
        if summary['trade_count'] == 0:
            payload["text"] = "🔇 No trading activity today"
        else:
            payload["text"] = f"Daily summary for {summary['date']}"
            
        return payload

    def _create_alert_payload(self, title: str, message: str, severity: str,
                            fields: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create Slack payload for alerts"""
        
        color_map = {
            "info": "good",
            "warning": "warning", 
            "error": "danger",
            "critical": "danger"
        }
        
        emoji_map = {
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
            "critical": ":rotating_light:"
        }
        
        attachment = {
            "color": color_map.get(severity, "good"),
            "title": title,
            "text": message,
            "footer": "Mech-Exo Trading System",
            "ts": int(datetime.now(timezone.utc).timestamp())
        }
        
        if fields:
            attachment["fields"] = fields
            
        payload = {
            "username": "Mech-Exo Bot",
            "icon_emoji": emoji_map.get(severity, ":robot_face:"),
            "attachments": [attachment]
        }
        
        return payload

    def test_connection(self) -> bool:
        """
        Test Slack webhook connection
        
        Returns:
            bool: True if connection successful
        """
        try:
            test_payload = {
                "username": "Mech-Exo Bot",
                "icon_emoji": ":wave:",
                "text": "🧪 Test message from Mech-Exo Trading System",
                "attachments": [{
                    "color": "good",
                    "text": "If you see this message, the Slack integration is working correctly!",
                    "footer": "Mech-Exo Trading System Test",
                    "ts": int(datetime.now(timezone.utc).timestamp())
                }]
            }
            
            response = self.session.post(self.webhook_url, json=test_payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Slack connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Slack connection test failed: {e}")
            return False

    def close(self):
        """Close HTTP session"""
        if self.session:
            self.session.close()
            logger.debug("SlackAlerter session closed")


def send_daily_digest_to_slack(date: str = "today", webhook_url: str = None) -> bool:
    """
    Convenience function to send daily digest to Slack
    
    Args:
        date: Date string (YYYY-MM-DD) or "today"
        webhook_url: Optional Slack webhook URL
        
    Returns:
        bool: True if message sent successfully
    """
    try:
        # Generate daily report
        report = DailyReport(date=date)
        
        # Create Slack alerter
        alerter = SlackAlerter(webhook_url=webhook_url)
        
        # Send digest
        success = alerter.send_daily_digest(report)
        
        # Clean up
        alerter.close()
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to send daily digest: {e}")
        return False


def send_risk_alert_to_slack(violation_details: Dict[str, Any], 
                           webhook_url: str = None) -> bool:
    """
    Convenience function to send risk violation alert to Slack
    
    Args:
        violation_details: Risk violation information
        webhook_url: Optional Slack webhook URL
        
    Returns:
        bool: True if alert sent successfully
    """
    try:
        alerter = SlackAlerter(webhook_url=webhook_url)
        success = alerter.send_risk_violation(violation_details)
        alerter.close()
        return success
        
    except Exception as e:
        logger.error(f"Failed to send risk alert: {e}")
        return False


def send_execution_alert_to_slack(order_id: str, status: str, 
                                details: Dict[str, Any],
                                webhook_url: str = None) -> bool:
    """
    Convenience function to send execution alert to Slack
    
    Args:
        order_id: Order ID
        status: Order status
        details: Order/fill details
        webhook_url: Optional Slack webhook URL
        
    Returns:
        bool: True if alert sent successfully  
    """
    try:
        alerter = SlackAlerter(webhook_url=webhook_url)
        success = alerter.send_execution_alert(order_id, status, details)
        alerter.close()
        return success
        
    except Exception as e:
        logger.error(f"Failed to send execution alert: {e}")
        return False