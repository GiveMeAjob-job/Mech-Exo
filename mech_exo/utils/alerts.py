"""
Alert system for fills, rejects, and risk violations
Supports Slack and email notifications
"""

import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

from .config import ConfigManager

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    FILL = "fill"
    ORDER_REJECT = "order_reject"
    RISK_VIOLATION = "risk_violation"
    SYSTEM_ERROR = "system_error"
    DAILY_SUMMARY = "daily_summary"


@dataclass
class Alert:
    """Alert message structure"""
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_type': self.alert_type.value,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data or {}
        }


class SlackAlerter:
    """Slack alert handler"""
    
    def __init__(self, config: Dict[str, Any]):
        if not SLACK_AVAILABLE:
            raise ImportError("slack_sdk not installed. Run: pip install slack_sdk")
        
        self.bot_token = config.get('bot_token')
        self.webhook_url = config.get('webhook_url')
        self.default_channel = config.get('default_channel', '#trading-alerts')
        self.username = config.get('username', 'Mech-Exo Bot')
        
        if not self.bot_token and not self.webhook_url:
            raise ValueError("Either bot_token or webhook_url must be provided for Slack")
        
        self.client = WebClient(token=self.bot_token) if self.bot_token else None
        
    def send_alert(self, alert: Alert, channel: Optional[str] = None) -> bool:
        """Send alert to Slack"""
        try:
            target_channel = channel or self.default_channel
            
            # Format message for Slack
            blocks = self._format_slack_message(alert)
            
            if self.client:
                # Use bot API
                response = self.client.chat_postMessage(
                    channel=target_channel,
                    blocks=blocks,
                    username=self.username,
                    icon_emoji=self._get_emoji_for_level(alert.level)
                )
                
                if response['ok']:
                    logger.info(f"Slack alert sent successfully: {alert.title}")
                    return True
                else:
                    logger.error(f"Slack API error: {response.get('error')}")
                    return False
            
            elif self.webhook_url:
                # Use webhook (simplified)
                import requests
                
                payload = {
                    'text': f"{alert.title}\n{alert.message}",
                    'username': self.username,
                    'channel': target_channel,
                    'icon_emoji': self._get_emoji_for_level(alert.level)
                }
                
                response = requests.post(self.webhook_url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"Slack webhook alert sent successfully: {alert.title}")
                    return True
                else:
                    logger.error(f"Slack webhook error: {response.status_code}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _format_slack_message(self, alert: Alert) -> List[Dict[str, Any]]:
        """Format alert as Slack blocks"""
        color = self._get_color_for_level(alert.level)
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{self._get_emoji_for_level(alert.level)} {alert.title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Type:* {alert.alert_type.value} | *Level:* {alert.level.value} | *Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            }
        ]
        
        # Add data fields if present
        if alert.data:
            fields = []
            for key, value in alert.data.items():
                if isinstance(value, (str, int, float)):
                    fields.append({
                        "type": "mrkdwn",
                        "text": f"*{key.replace('_', ' ').title()}:* {value}"
                    })
            
            if fields:
                blocks.append({
                    "type": "section",
                    "fields": fields
                })
        
        return blocks
    
    def _get_emoji_for_level(self, level: AlertLevel) -> str:
        """Get emoji for alert level"""
        emoji_map = {
            AlertLevel.INFO: ":information_source:",
            AlertLevel.WARNING: ":warning:",
            AlertLevel.ERROR: ":x:",
            AlertLevel.CRITICAL: ":rotating_light:"
        }
        return emoji_map.get(level, ":question:")
    
    def _get_color_for_level(self, level: AlertLevel) -> str:
        """Get color for alert level"""
        color_map = {
            AlertLevel.INFO: "#36a64f",      # Green
            AlertLevel.WARNING: "#ff9500",   # Orange
            AlertLevel.ERROR: "#ff0000",     # Red
            AlertLevel.CRITICAL: "#8b0000"   # Dark red
        }
        return color_map.get(level, "#808080")


class EmailAlerter:
    """Email alert handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.sender_email = config.get('sender_email')
        self.sender_password = config.get('sender_password')
        self.recipient_emails = config.get('recipient_emails', [])
        
        if not self.sender_email or not self.sender_password:
            raise ValueError("sender_email and sender_password must be provided for email alerts")
        
        if not self.recipient_emails:
            raise ValueError("recipient_emails must be provided for email alerts")
    
    def send_alert(self, alert: Alert, recipients: Optional[List[str]] = None) -> bool:
        """Send alert via email"""
        try:
            target_recipients = recipients or self.recipient_emails
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"[Mech-Exo] {alert.title}"
            message["From"] = self.sender_email
            message["To"] = ", ".join(target_recipients)
            
            # Create HTML and text content
            html_content = self._format_html_message(alert)
            text_content = self._format_text_message(alert)
            
            # Attach content
            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")
            
            message.attach(text_part)
            message.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, target_recipients, message.as_string())
            
            logger.info(f"Email alert sent successfully: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _format_html_message(self, alert: Alert) -> str:
        """Format alert as HTML email"""
        color = self._get_color_for_level(alert.level)
        
        html = f"""
        <html>
          <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
              <div style="background-color: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0;">
                <h2 style="margin: 0;">{alert.title}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">{alert.alert_type.value.replace('_', ' ').title()} Alert</p>
              </div>
              
              <div style="border: 1px solid {color}; border-top: none; padding: 20px; border-radius: 0 0 5px 5px;">
                <div style="margin-bottom: 20px;">
                  <h3 style="color: #333; margin-bottom: 10px;">Message:</h3>
                  <p style="background-color: #f9f9f9; padding: 15px; border-radius: 3px; margin: 0;">
                    {alert.message.replace(chr(10), '<br>')}
                  </p>
                </div>
                
                <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                  <div>
                    <strong>Level:</strong> {alert.level.value.upper()}
                  </div>
                  <div>
                    <strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                  </div>
                </div>
        """
        
        # Add data section if present
        if alert.data:
            html += """
                <div>
                  <h3 style="color: #333; margin-bottom: 10px;">Details:</h3>
                  <table style="width: 100%; border-collapse: collapse;">
            """
            
            for key, value in alert.data.items():
                if isinstance(value, (str, int, float)):
                    html += f"""
                    <tr>
                      <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">
                        {key.replace('_', ' ').title()}:
                      </td>
                      <td style="padding: 8px; border-bottom: 1px solid #eee;">
                        {value}
                      </td>
                    </tr>
                    """
            
            html += """
                  </table>
                </div>
            """
        
        html += """
              </div>
              
              <div style="text-align: center; margin-top: 20px; color: #666; font-size: 12px;">
                Generated by Mech-Exo Trading System
              </div>
            </div>
          </body>
        </html>
        """
        
        return html
    
    def _format_text_message(self, alert: Alert) -> str:
        """Format alert as plain text"""
        text = f"""
MECH-EXO ALERT: {alert.title}

Type: {alert.alert_type.value.replace('_', ' ').title()}
Level: {alert.level.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert.message}
"""
        
        if alert.data:
            text += "\nDetails:\n"
            for key, value in alert.data.items():
                if isinstance(value, (str, int, float)):
                    text += f"  {key.replace('_', ' ').title()}: {value}\n"
        
        text += "\n---\nGenerated by Mech-Exo Trading System"
        
        return text
    
    def _get_color_for_level(self, level: AlertLevel) -> str:
        """Get color for alert level"""
        color_map = {
            AlertLevel.INFO: "#2196F3",      # Blue
            AlertLevel.WARNING: "#FF9800",   # Orange
            AlertLevel.ERROR: "#F44336",     # Red
            AlertLevel.CRITICAL: "#8B0000"   # Dark red
        }
        return color_map.get(level, "#757575")


class TelegramAlerter:
    """
    Telegram alerting functionality for trading notifications
    
    Provides Markdown-formatted notifications with proper character escaping
    for strategy retraining and other critical alerts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Telegram alerter
        
        Args:
            config: Configuration dictionary with bot_token and chat_id
        """
        self.bot_token = config.get('bot_token')
        self.chat_id = config.get('chat_id')
        self.username = config.get('username', 'Mech-Exo Bot')
        
        if not self.bot_token:
            raise ValueError("bot_token must be provided for Telegram alerts")
        if not self.chat_id:
            raise ValueError("chat_id must be provided for Telegram alerts")
        
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def escape_markdown(self, text: str) -> str:
        """
        Escape Markdown special characters for Telegram
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text safe for Telegram Markdown
        """
        # Characters that need escaping in Telegram Markdown
        escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        
        escaped_text = text
        for char in escape_chars:
            escaped_text = escaped_text.replace(char, f'\\{char}')
        
        return escaped_text
    
    def send_message(self, message: str, parse_mode: str = "MarkdownV2") -> bool:
        """
        Send message to Telegram
        
        Args:
            message: Message to send
            parse_mode: Parse mode (MarkdownV2, Markdown, HTML, or None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import os
            
            # Check for dry-run mode
            if os.getenv('TELEGRAM_DRY_RUN', 'false').lower() == 'true':
                logger.info("TELEGRAM_DRY_RUN=true - logging message instead of sending")
                logger.info(f"Dry-run Telegram message:\n{message}")
                return True
            
            import requests
            
            payload = {
                "chat_id": self.chat_id,
                "text": message
            }
            
            if parse_mode:
                payload["parse_mode"] = parse_mode
            
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram message failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via Telegram"""
        try:
            # Convert alert to simple format for Telegram
            emoji = self._get_emoji_for_level(alert.level)
            
            message = f"{emoji} *{self.escape_markdown(alert.title)}*\n\n"
            message += f"{self.escape_markdown(alert.message)}\n\n"
            message += f"*Level:* {alert.level.value.upper()}\n"
            message += f"*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def send_retrain_success(self, validation_results: Dict[str, Any], 
                           version: str, factors_file: str) -> bool:
        """
        Send successful retrain notification
        
        Args:
            validation_results: Validation results from walk-forward analysis
            version: Version timestamp
            factors_file: Path to new factors file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            sharpe = validation_results.get('out_of_sample_sharpe', 0)
            max_dd = validation_results.get('max_drawdown', 0)
            segments_passed = validation_results.get('segments_passed', 0)
            segments_total = validation_results.get('segments_total', 0)
            
            # Create success message with proper Markdown escaping
            message = f"""⚙️ *Retrain Completed Successfully*

📈 **Sharpe Ratio**: {sharpe:.2f}
📉 **Max Drawdown**: {max_dd:.1%}
✅ **Validation**: {segments_passed}/{segments_total} segments passed
🔧 **Version**: `{self.escape_markdown(version)}`
📁 **Weights File**: `{self.escape_markdown(factors_file)}`

🚀 *New factors deployed and ready for trading*"""
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send retrain success notification: {e}")
            return False
    
    def send_retrain_failure(self, failure_reason: str, version: str) -> bool:
        """
        Send failed retrain notification
        
        Args:
            failure_reason: Reason for failure
            version: Version timestamp
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create failure message with proper Markdown escaping
            message = f"""❌ *Retrain Failed*

🚫 **Reason**: {self.escape_markdown(failure_reason)}
🔧 **Version**: `{self.escape_markdown(version)}`
📅 **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ *Manual review required \\- existing factors remain active*"""
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send retrain failure notification: {e}")
            return False
    
    def send_validation_failure(self, validation_results: Dict[str, Any], 
                              deployment_reason: str, version: str) -> bool:
        """
        Send validation failure notification
        
        Args:
            validation_results: Validation results from walk-forward analysis
            deployment_reason: Reason deployment was skipped
            version: Version timestamp
            
        Returns:
            True if successful, False otherwise
        """
        try:
            sharpe = validation_results.get('out_of_sample_sharpe', 0)
            max_dd = validation_results.get('max_drawdown', 0)
            segments_passed = validation_results.get('segments_passed', 0)
            segments_total = validation_results.get('segments_total', 0)
            
            # Create validation failure message
            message = f"""⚠️ *Retrain Validation Failed*

📈 **Sharpe Ratio**: {sharpe:.2f}
📉 **Max Drawdown**: {max_dd:.1%}
❌ **Validation**: {segments_passed}/{segments_total} segments passed
🚫 **Reason**: {self.escape_markdown(deployment_reason)}
🔧 **Version**: `{self.escape_markdown(version)}`

🔄 *Factors not deployed \\- existing configuration remains active*"""
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send validation failure notification: {e}")
            return False
    
    def _get_emoji_for_level(self, level: AlertLevel) -> str:
        """Get emoji for alert level"""
        emoji_map = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }
        return emoji_map.get(level, "❓")


class AlertManager:
    """Main alert manager coordinating multiple channels"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager()
        
        # Load alert configuration
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            # Try to load from standard locations
            try:
                self.config = self.config_manager.load_config('alerts')
            except:
                self.config = {}
        
        # Initialize alerters
        self.alerters = {}
        
        # Setup Slack if configured
        slack_config = self.config.get('slack', {})
        if slack_config.get('enabled', False):
            try:
                self.alerters['slack'] = SlackAlerter(slack_config)
                logger.info("Slack alerter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Slack alerter: {e}")
        
        # Setup Email if configured
        email_config = self.config.get('email', {})
        if email_config.get('enabled', False):
            try:
                self.alerters['email'] = EmailAlerter(email_config)
                logger.info("Email alerter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Email alerter: {e}")
        
        # Setup Telegram if configured
        telegram_config = self.config.get('telegram', {})
        if telegram_config.get('enabled', False):
            try:
                self.alerters['telegram'] = TelegramAlerter(telegram_config)
                logger.info("Telegram alerter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Telegram alerter: {e}")
        
        # Alert filtering
        self.min_level = AlertLevel(self.config.get('min_level', 'info'))
        self.enabled_types = set(self.config.get('enabled_types', [t.value for t in AlertType]))
        
        logger.info(f"AlertManager initialized with {len(self.alerters)} alerters")
    
    def send_alert(self, alert: Alert, channels: Optional[List[str]] = None) -> bool:
        """Send alert through configured channels"""
        if not self._should_send_alert(alert):
            logger.debug(f"Alert filtered out: {alert.title}")
            return True
        
        # Determine which channels to use
        target_channels = channels or list(self.alerters.keys())
        
        success_count = 0
        total_count = len([ch for ch in target_channels if ch in self.alerters])
        
        for channel in target_channels:
            if channel in self.alerters:
                try:
                    if self.alerters[channel].send_alert(alert):
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")
        
        if success_count > 0:
            logger.info(f"Alert sent successfully via {success_count}/{total_count} channels: {alert.title}")
            return True
        else:
            logger.error(f"Failed to send alert via any channel: {alert.title}")
            return False
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on filters"""
        # Check level filter
        level_order = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.ERROR: 2,
            AlertLevel.CRITICAL: 3
        }
        
        if level_order[alert.level] < level_order[self.min_level]:
            return False
        
        # Check type filter
        if alert.alert_type.value not in self.enabled_types:
            return False
        
        return True
    
    # Convenience methods for common alerts
    
    def send_fill_alert(self, symbol: str, quantity: int, price: float, fill_id: str) -> bool:
        """Send fill notification"""
        side = "BUY" if quantity > 0 else "SELL"
        alert = Alert(
            alert_type=AlertType.FILL,
            level=AlertLevel.INFO,
            title=f"Order Filled: {symbol}",
            message=f"Order filled: {side} {abs(quantity)} shares of {symbol} @ ${price:.2f}",
            timestamp=datetime.now(),
            data={
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'fill_id': fill_id,
                'side': side,
                'notional_value': abs(quantity) * price
            }
        )
        return self.send_alert(alert)
    
    def send_order_reject_alert(self, symbol: str, quantity: int, reason: str, order_id: str) -> bool:
        """Send order rejection notification"""
        alert = Alert(
            alert_type=AlertType.ORDER_REJECT,
            level=AlertLevel.WARNING,
            title=f"Order Rejected: {symbol}",
            message=f"Order rejected: {symbol} {quantity} shares" + "\n" + f"Reason: {reason}",
            timestamp=datetime.now(),
            data={
                'symbol': symbol,
                'quantity': quantity,
                'rejection_reason': reason,
                'order_id': order_id
            }
        )
        return self.send_alert(alert)
    
    def send_risk_violation_alert(self, violations: List[str], severity: str = 'warning') -> bool:
        """Send risk violation notification"""
        level = AlertLevel.ERROR if severity == 'critical' else AlertLevel.WARNING
        
        alert = Alert(
            alert_type=AlertType.RISK_VIOLATION,
            level=level,
            title="Risk Violation Detected",
            message=f"Risk management violations detected:" + "\n" + "\n".join(f"• {v}" for v in violations),
            timestamp=datetime.now(),
            data={
                'violation_count': len(violations),
                'violations': violations,
                'severity': severity
            }
        )
        return self.send_alert(alert)
    
    def send_system_error_alert(self, component: str, error_message: str, error_data: Optional[Dict] = None) -> bool:
        """Send system error notification"""
        alert = Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.ERROR,
            title=f"System Error: {component}",
            message=f"Error in {component}:" + "\n" + f"{error_message}",
            timestamp=datetime.now(),
            data={
                'component': component,
                'error_message': error_message,
                **(error_data or {})
            }
        )
        return self.send_alert(alert)
    
    def send_daily_summary_alert(self, summary_data: Dict[str, Any]) -> bool:
        """Send daily summary notification"""
        alert = Alert(
            alert_type=AlertType.DAILY_SUMMARY,
            level=AlertLevel.INFO,
            title="Daily Trading Summary",
            message=self._format_daily_summary_message(summary_data),
            timestamp=datetime.now(),
            data=summary_data
        )
        return self.send_alert(alert)
    
    def _format_daily_summary_message(self, summary: Dict[str, Any]) -> str:
        """Format daily summary message"""
        msg = f"Daily trading summary for {summary.get('date', 'today')}:" + "\n\n"
        
        if 'signal_generation' in summary:
            sg = summary['signal_generation']
            msg += f"📊 Signals: {sg.get('signals_generated', 0)} generated" + "\n"
        
        if 'execution' in summary:
            ex = summary['execution']
            msg += f"📈 Orders: {ex.get('orders_submitted', 0)} submitted, {ex.get('fills_received', 0)} filled" + "\n"
        
        if 'risk_management' in summary:
            rm = summary['risk_management']
            msg += f"🛡️ Risk: {rm.get('positions_approved', 0)} approved, {rm.get('violations_count', 0)} violations" + "\n"
        
        if 'system_health' in summary:
            sh = summary['system_health']
            execution_rate = sh.get('execution_rate', 0)
            msg += f"💚 Health: {execution_rate:.1%} execution rate" + "\n"
        
        return msg.strip()