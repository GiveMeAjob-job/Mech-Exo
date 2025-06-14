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
    SYSTEM_ALERT = "system_alert"  # Added for runbook escalations
    SYSTEM_INFO = "system_info"    # Added for rollback notifications
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
    
    def send_document(self, file_path: str, caption: str = None) -> bool:
        """
        Send document to Telegram
        
        Args:
            file_path: Path to file to send
            caption: Optional caption for the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import os
            from pathlib import Path
            
            # Check for dry-run mode
            if os.getenv('TELEGRAM_DRY_RUN', 'false').lower() == 'true':
                logger.info("TELEGRAM_DRY_RUN=true - logging document send instead of sending")
                logger.info(f"Dry-run Telegram document: {file_path}")
                if caption:
                    logger.info(f"Caption: {caption}")
                return True
            
            import requests
            
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Check file size (Telegram limit is 50MB)
            file_size = file_path.stat().st_size
            if file_size > 50 * 1024 * 1024:  # 50MB
                logger.error(f"File too large for Telegram: {file_size} bytes")
                return False
            
            # Prepare multipart form data
            with open(file_path, 'rb') as file:
                files = {
                    'document': (file_path.name, file, 'application/octet-stream')
                }
                
                data = {
                    'chat_id': self.chat_id
                }
                
                if caption:
                    data['caption'] = caption
                
                response = requests.post(
                    f"{self.api_url}/sendDocument",
                    files=files,
                    data=data,
                    timeout=30  # Longer timeout for file uploads
                )
            
            if response.status_code == 200:
                logger.info(f"Telegram document sent successfully: {file_path.name}")
                return True
            else:
                logger.error(f"Telegram document send failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Telegram document: {e}")
            return False
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via Telegram"""
        try:
            # Special handling for daily summary with ML metrics
            if alert.alert_type == AlertType.DAILY_SUMMARY and alert.data and 'ml_live_metrics' in alert.data:
                return self.send_ml_enhanced_daily_summary_telegram(alert)
            
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
    
    def send_ml_enhanced_daily_summary_telegram(self, alert: Alert) -> bool:
        """Send ML-enhanced daily summary with rich Telegram formatting"""
        try:
            summary = alert.data
            date = summary.get('date', 'today')
            
            # Create rich Telegram message with proper Markdown escaping
            message = f"📊 *Daily Trading Summary*\n"
            message += f"📅 {self.escape_markdown(str(date))}\n\n"
            
            # Signal Generation
            if 'signal_generation' in summary:
                sg = summary['signal_generation']
                signals = sg.get('signals_generated', 0)
                message += f"🔮 *Signals Generated:* {signals}\n"
            
            # Execution
            if 'execution' in summary:
                ex = summary['execution']
                orders = ex.get('orders_submitted', 0)
                fills = ex.get('fills_received', 0)
                message += f"📈 *Execution:* {orders} orders, {fills} fills\n"
            
            # ML Live Validation Metrics (Enhanced Day 5 formatting)
            if 'ml_live_metrics' in summary:
                ml = summary['ml_live_metrics']
                hit_rate = ml.get('hit_rate', 0.0)
                auc = ml.get('auc', 0.0)
                ic = ml.get('ic', 0.0)
                
                # Performance indicator with emoji
                if hit_rate > 0.55:
                    status_emoji = "🟢"
                    status = "STRONG"
                elif hit_rate > 0.50:
                    status_emoji = "🟡" 
                    status = "NEUTRAL"
                else:
                    status_emoji = "🔴"
                    status = "WEAK"
                
                message += f"\n🤖 *ML Signal Validation*\n"
                message += f"└ {status_emoji} *Status:* {status}\n"
                message += f"└ 🎯 *Hit Rate:* {hit_rate:.1%}\n"
                message += f"└ 📊 *AUC:* {auc:.3f}\n"
                message += f"└ 📈 *IC:* {ic:.3f}\n"
            
            # Risk Management
            if 'risk_management' in summary:
                rm = summary['risk_management']
                approved = rm.get('positions_approved', 0)
                violations = rm.get('violations_count', 0)
                violation_emoji = "🛡️" if violations == 0 else "⚠️"
                message += f"\n{violation_emoji} *Risk:* {approved} approved, {violations} violations\n"
            
            # System Health
            if 'system_health' in summary:
                sh = summary['system_health']
                execution_rate = sh.get('execution_rate', 0)
                health_emoji = "💚" if execution_rate > 0.8 else "🟡" if execution_rate > 0.5 else "🔴"
                message += f"{health_emoji} *Health:* {execution_rate:.1%} execution rate\n"
            
            message += f"\n🕐 *Generated:* {alert.timestamp.strftime('%H:%M:%S UTC')}"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send ML-enhanced Telegram summary: {e}")
            # Fallback to standard alert formatting
            return self.send_message(f"📊 Daily Summary (fallback): {alert.message}")
    
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

⚠️ *Manual review required - existing factors remain active*"""
            
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

🔄 *Factors not deployed - existing configuration remains active*"""
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send validation failure notification: {e}")
            return False
    
    def send_weight_change(self, old_w: float, new_w: float, 
                          sharpe_ml: float, sharpe_base: float, 
                          adjustment_rule: str, dry_run: bool = False) -> bool:
        """
        Send ML weight change notification with formatted message.
        
        Args:
            old_w: Previous ML weight
            new_w: New ML weight  
            sharpe_ml: ML strategy Sharpe ratio
            sharpe_base: Baseline strategy Sharpe ratio
            adjustment_rule: Rule that triggered the change
            dry_run: If True, only log the message instead of sending
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import os
            import re
            import subprocess
            from datetime import datetime
            
            # Check for dry-run override
            env_dry_run = os.getenv('TELEGRAM_DRY_RUN', 'false').lower() == 'true'
            is_dry_run = dry_run or env_dry_run
            
            # Calculate delta
            delta_sharpe = sharpe_ml - sharpe_base
            
            # Determine direction emoji
            if new_w > old_w:
                direction_emoji = "↗️"
                direction = "increased"
            elif new_w < old_w:
                direction_emoji = "↘️" 
                direction = "decreased"
            else:
                direction_emoji = "➡️"
                direction = "unchanged"
            
            # Get Git commit hash for traceability
            try:
                git_hash = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], 
                    cwd="/Users/binwspacerace/PycharmProjects/Mech-Exo",
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                git_info = f" \\(commit `{git_hash}`\\)"
            except:
                git_info = ""
            
            # Build Markdown V2 message with proper escaping
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message_parts = [
                "⚖️ *ML Weight Auto-Adjusted*",
                "",
                f"• *Weight*: {old_w:.2f} {direction_emoji} {new_w:.2f}",
                f"• *Δ Sharpe*: {delta_sharpe:+.3f} \\({sharpe_ml:.3f} vs {sharpe_base:.3f}\\)",
                f"• *Rule*: `{self.escape_markdown(adjustment_rule)}`",
                f"• *Time*: {time_str}{git_info}"
            ]
            
            message = "\n".join(message_parts)
            
            if is_dry_run:
                logger.info("TELEGRAM_DRY_RUN=true - logging weight change message")
                logger.info(f"Dry-run weight change notification:\n{message}")
                return True
            
            # Send actual message
            success = self.send_message(message, parse_mode="MarkdownV2")
            
            if success:
                logger.info(f"Weight change notification sent: {old_w:.2f} → {new_w:.2f}")
            else:
                logger.error("Failed to send weight change notification")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to send weight change notification: {e}")
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
    
    def is_quiet_hours(self, current_time: datetime = None) -> bool:
        """
        Check if current time is within quiet hours (22:00-06:00 local)
        
        Args:
            current_time: Time to check (defaults to now)
            
        Returns:
            True if in quiet hours
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Get quiet hours configuration (default: 22:00-06:00)
        quiet_config = self.config.get('quiet_hours', {})
        start_hour = quiet_config.get('start_hour', 22)  # 22:00
        end_hour = quiet_config.get('end_hour', 6)       # 06:00
        
        current_hour = current_time.hour
        
        if start_hour <= end_hour:
            # Same day range (e.g., 22:00-23:59) - unlikely but supported
            return start_hour <= current_hour <= end_hour
        else:
            # Crosses midnight (e.g., 22:00-06:00)
            return current_hour >= start_hour or current_hour <= end_hour
    
    def send_alert_with_escalation(self, alert: Alert, 
                                 channels: Optional[List[str]] = None,
                                 respect_quiet_hours: bool = True,
                                 force_send: bool = False) -> bool:
        """
        Send alert with escalation and quiet hours support
        
        Args:
            alert: Alert to send
            channels: Target channels (None for all)
            respect_quiet_hours: If True, suppress alerts during quiet hours
            force_send: If True, override quiet hours for critical alerts
            
        Returns:
            True if sent successfully
        """
        # Check quiet hours
        if respect_quiet_hours and not force_send:
            if self.is_quiet_hours():
                # Only send critical system alerts during quiet hours
                if alert.level != AlertLevel.CRITICAL or alert.alert_type not in [
                    AlertType.SYSTEM_ERROR, AlertType.SYSTEM_ALERT
                ]:
                    logger.info(f"Alert suppressed due to quiet hours: {alert.title}")
                    return True  # Consider it "successful" - just deferred
        
        # For critical system alerts, force send to telegram even in quiet hours
        if (alert.level == AlertLevel.CRITICAL and 
            alert.alert_type in [AlertType.SYSTEM_ERROR, AlertType.SYSTEM_ALERT]):
            force_channels = channels or ['telegram']
            return self.send_alert(alert, channels=force_channels)
        
        return self.send_alert(alert, channels=channels)
    
    def get_escalation_channels(self, escalation_level: str) -> List[str]:
        """
        Get appropriate channels for escalation level
        
        Args:
            escalation_level: 'telegram', 'email', or 'phone'
            
        Returns:
            List of channel names to use
        """
        escalation_map = {
            'telegram': ['telegram'],
            'email': ['email'],
            'phone': ['telegram', 'email']  # Phone not implemented, use both
        }
        
        requested_channels = escalation_map.get(escalation_level, ['telegram'])
        
        # Filter to only available channels
        available_channels = []
        for channel in requested_channels:
            if channel in self.alerters:
                available_channels.append(channel)
        
        # Fallback to any available channel if none match
        if not available_channels and self.alerters:
            available_channels = [list(self.alerters.keys())[0]]
        
        return available_channels
    
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
    
    def send_ml_enhanced_daily_summary(self, summary_data: Dict[str, Any], 
                                      ml_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Send ML-enhanced daily summary with live validation metrics.
        
        Args:
            summary_data: Standard daily summary data
            ml_metrics: ML live metrics (hit_rate, auc, ic, sharpe_30d)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Auto-fetch ML metrics if not provided
            if ml_metrics is None:
                from ..reporting.query import get_latest_ml_live_metrics
                ml_metrics = get_latest_ml_live_metrics()
            
            # Add ML metrics to summary data
            enhanced_summary = summary_data.copy()
            enhanced_summary['ml_live_metrics'] = ml_metrics
            
            return self.send_daily_summary_alert(enhanced_summary)
            
        except Exception as e:
            logger.error(f"Failed to send ML-enhanced daily summary: {e}")
            # Fallback to regular summary
            return self.send_daily_summary_alert(summary_data)
    
    def _format_daily_summary_message(self, summary: Dict[str, Any]) -> str:
        """Format daily summary message"""
        msg = f"Daily trading summary for {summary.get('date', 'today')}:" + "\n\n"
        
        if 'signal_generation' in summary:
            sg = summary['signal_generation']
            msg += f"📊 Signals: {sg.get('signals_generated', 0)} generated" + "\n"
        
        if 'execution' in summary:
            ex = summary['execution']
            msg += f"📈 Orders: {ex.get('orders_submitted', 0)} submitted, {ex.get('fills_received', 0)} filled" + "\n"
        
        # ML Live Validation Metrics (Day 5 enhancement)
        if 'ml_live_metrics' in summary:
            ml = summary['ml_live_metrics']
            hit_rate = ml.get('hit_rate', 0.0)
            auc = ml.get('auc', 0.0)
            ic = ml.get('ic', 0.0)
            
            # Determine emoji based on hit rate performance
            if hit_rate > 0.55:
                ml_emoji = "🟢"
                performance = "STRONG"
            elif hit_rate > 0.50:
                ml_emoji = "🟡"
                performance = "NEUTRAL"
            else:
                ml_emoji = "🔴"
                performance = "WEAK"
            
            msg += f"🤖 ML Signal: {ml_emoji} {performance} | Hit Rate: {hit_rate:.1%} | AUC: {auc:.3f} | IC: {ic:.3f}" + "\n"
        
        if 'risk_management' in summary:
            rm = summary['risk_management']
            msg += f"🛡️ Risk: {rm.get('positions_approved', 0)} approved, {rm.get('violations_count', 0)} violations" + "\n"
        
        if 'system_health' in summary:
            sh = summary['system_health']
            execution_rate = sh.get('execution_rate', 0)
            msg += f"💚 Health: {execution_rate:.1%} execution rate" + "\n"
        
        return msg.strip()


def send_monthly_loss_alert(mtd_pct: float, threshold_pct: float = -3.0) -> bool:
    """
    Send monthly loss alert when monthly threshold is breached
    
    Args:
        mtd_pct: Month-to-date PnL percentage
        threshold_pct: Monthly threshold percentage (default: -3.0%)
        
    Returns:
        True if alert sent successfully
    """
    try:
        import subprocess
        
        # Get current timestamp and commit hash for traceability
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get Git commit hash for traceability
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            commit_info = f" \\(commit `{git_hash}`\\)"
        except:
            commit_info = ""
        
        # Calculate difference to threshold
        diff_to_threshold = mtd_pct - threshold_pct
        
        # Determine alert level
        level = AlertLevel.CRITICAL
        title = f"🛑 MONTHLY STOP-LOSS TRIGGERED: {mtd_pct:+.2f}%"
        
        # Build detailed message with proper MarkdownV2 escaping
        message = f"""🛑 **MONTHLY DRAWDOWN ALERT**

📅 **Month-to-Date**: {mtd_pct:+.3f}%
🚨 **Threshold**: {threshold_pct}%
📉 **Breach Amount**: {diff_to_threshold:+.3f}%

⚡ **KILL-SWITCH ACTIVATED**
🔴 **Trading has been automatically disabled**

🕐 **Time**: {timestamp.replace('-', '-')}{commit_info}

⚠️ *Manual review required to re-enable trading*
📋 *See run-book for escalation procedures*"""
        
        # Verify message length (Telegram limit is 4096 chars)
        if len(message) > 4096:
            logger.warning(f"Monthly alert message too long: {len(message)} chars, truncating")
            message = message[:4093] + "..."
        
        # Create alert
        alert = Alert(
            alert_type=AlertType.SYSTEM_ALERT,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            data={
                'mtd_pct': mtd_pct,
                'threshold_pct': threshold_pct,
                'diff_to_threshold': diff_to_threshold,
                'alert_type': 'monthly_stop_loss',
                'commit_hash': git_hash if 'git_hash' in locals() else None
            }
        )
        
        # Send through alert manager
        try:
            alert_manager = AlertManager()
            
            # Force send critical monthly alerts even during quiet hours
            success = alert_manager.send_alert_with_escalation(
                alert,
                channels=['telegram'],
                respect_quiet_hours=True,
                force_send=True  # Always force send for monthly stop-loss
            )
            
            if success:
                logger.error(f"✅ Monthly loss alert sent: {mtd_pct:+.3f}%")
            else:
                logger.error(f"❌ Failed to send monthly loss alert: {mtd_pct:+.3f}%")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize alert manager for monthly alert: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send monthly loss alert: {e}")
        return False


def send_drill_report(report_path: str, passed: bool) -> bool:
    """
    Send rollback drill report via Telegram
    
    Args:
        report_path: Path to drill report Markdown file
        passed: Whether drill passed (True) or failed (False)
        
    Returns:
        True if alert sent successfully
    """
    try:
        import subprocess
        from pathlib import Path
        
        report_file = Path(report_path)
        if not report_file.exists():
            logger.error(f"Drill report file not found: {report_path}")
            return False
        
        # Check file size (Telegram limit is 50MB, but we want ≤ 300KB)
        file_size = report_file.stat().st_size
        if file_size > 300 * 1024:  # 300KB
            logger.warning(f"Drill report file too large: {file_size} bytes (limit: 300KB)")
            # Could truncate or compress, but for now just warn
        
        # Get current timestamp and commit hash for traceability
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get Git commit hash for traceability
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            commit_info = f" \\\\(commit `{git_hash}`\\\\)"
        except:
            commit_info = ""
        
        # Determine alert level and emoji
        if passed:
            level = AlertLevel.INFO
            result_emoji = "✅"
            status_text = "PASSED"
            title = "🔄 Rollback Drill Completed Successfully"
        else:
            level = AlertLevel.WARNING
            result_emoji = "⚠️"
            status_text = "FAILED"
            title = "🚨 Rollback Drill Failed"
        
        # Extract key info from report for message preview
        try:
            with open(report_file, 'r') as f:
                report_content = f.read()
            
            # Parse duration from report
            duration_match = None
            for line in report_content.split('\\n'):
                if 'Duration' in line:
                    duration_match = line.split('**Duration**:')[1].split('**')[0].strip() if '**Duration**:' in line else None
                    break
            
            duration_str = duration_match or "Unknown"
            
        except Exception as e:
            logger.warning(f"Could not parse report content: {e}")
            duration_str = "Unknown"
        
        # Build detailed message with proper MarkdownV2 escaping
        dash_escaped = '-'  # No escaping needed for regular dash
        message = f"""{result_emoji} **ROLLBACK DRILL COMPLETE**

📋 **Result**: {status_text}
⏱️ **Duration**: {duration_str.replace('-', dash_escaped).replace('.', '.')}
📅 **Time**: {timestamp.replace('-', dash_escaped)}
📄 **Report**: `{report_file.name}`

{result_emoji} **Kill-Switch Test Cycle**:
└ 💾 Backup configuration
└ 🛑 Disable trading  
└ ⏳ Wait period
└ ✅ Restore trading

🕐 **Generated**: {timestamp.replace('-', dash_escaped)}{commit_info}

📊 *Full report attached below*"""
        
        # Verify message length (Telegram limit is 4096 chars)
        if len(message) > 4096:
            logger.warning(f"Drill alert message too long: {len(message)} chars, truncating")
            message = message[:4093] + "..."
        
        # Create alert
        alert = Alert(
            alert_type=AlertType.SYSTEM_INFO,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            data={
                'drill_passed': passed,
                'report_path': str(report_path),
                'report_size_bytes': file_size,
                'drill_status': status_text,
                'report_file': report_file.name,
                'commit_hash': git_hash if 'git_hash' in locals() else None
            }
        )
        
        # Send through alert manager
        try:
            alert_manager = AlertManager()
            
            # Send text alert first
            text_success = alert_manager.send_alert_with_escalation(
                alert,
                channels=['telegram'],
                respect_quiet_hours=False,  # Drill reports can be sent any time
                force_send=False
            )
            
            # Send document if Telegram is available
            document_success = False
            if 'telegram' in alert_manager.alerters:
                telegram_alerter = alert_manager.alerters['telegram']
                
                # Send the report file as document
                caption = f"{result_emoji} Rollback Drill Report - {status_text}"
                document_success = telegram_alerter.send_document(
                    str(report_file),
                    caption=caption
                )
            
            overall_success = text_success and document_success
            
            if overall_success:
                logger.info(f"✅ Drill report sent successfully: {status_text}")
            else:
                logger.error(f"❌ Partial failure sending drill report: text={text_success}, doc={document_success}")
                
            return overall_success
            
        except Exception as e:
            logger.error(f"Failed to initialize alert manager for drill report: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send drill report: {e}")
        return False


def send_intraday_loss_alert(pnl_pct: float, nav_data: Dict[str, Any], threshold_result: Dict[str, Any]) -> bool:
    """
    Send intraday loss alert when PnL threshold is breached
    
    Args:
        pnl_pct: Current PnL percentage
        nav_data: Complete NAV data dictionary
        threshold_result: Threshold check results
        
    Returns:
        True if alert sent successfully
    """
    try:
        # Determine alert level based on severity
        if threshold_result.get('threshold_breached', False):
            level = AlertLevel.CRITICAL
            title = f"🚨 CRITICAL DAY LOSS: {pnl_pct:+.2f}%"
        elif threshold_result.get('warning_level', False):
            level = AlertLevel.WARNING
            title = f"⚠️ WARNING DAY LOSS: {pnl_pct:+.2f}%"
        else:
            level = AlertLevel.INFO
            title = f"📊 Intraday PnL Update: {pnl_pct:+.2f}%"
        
        # Build detailed message
        killswitch_status = "ENABLED" if threshold_result.get('killswitch_triggered', False) else "NOT TRIGGERED"
        action_taken = threshold_result.get('action_taken', 'none')
        
        message = f"""🔴 **INTRADAY PnL ALERT**

📊 **Current PnL**: {pnl_pct:+.3f}%
💰 **Live NAV**: ${nav_data['live_nav']:,.2f}
📈 **Day Start NAV**: ${nav_data['day_start_nav']:,.2f}
💸 **PnL Amount**: ${nav_data['pnl_amount']:+,.2f}

🏢 **Portfolio Status**:
• Positions: {nav_data['position_count']}
• Gross Exposure: ${nav_data.get('gross_exposure', 0):,.0f}
• Net Exposure: ${nav_data.get('net_exposure', 0):,.0f}

🚨 **Kill-Switch**: {killswitch_status}
⚡ **Action**: {action_taken.upper().replace('_', ' ')}

🕐 **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        # Add top positions if available
        if nav_data.get('top_positions'):
            message += "\n\n📈 **Top Positions**:"
            for i, pos in enumerate(nav_data['top_positions'][:3], 1):
                pnl_str = f"{pos['unrealized_pnl']:+,.0f}" if pos.get('unrealized_pnl') else "N/A"
                message += f"\n{i}. {pos['symbol']}: {pos['quantity']:+.0f} @ ${pos.get('current_price', 0):.2f} (${pnl_str})"
        
        # Create alert
        alert = Alert(
            alert_type=AlertType.SYSTEM_ALERT,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            data={
                'pnl_pct': pnl_pct,
                'live_nav': nav_data['live_nav'],
                'day_start_nav': nav_data['day_start_nav'],
                'pnl_amount': nav_data['pnl_amount'],
                'position_count': nav_data['position_count'],
                'killswitch_triggered': threshold_result.get('killswitch_triggered', False),
                'threshold_breached': threshold_result.get('threshold_breached', False),
                'warning_level': threshold_result.get('warning_level', False),
                'action_taken': action_taken
            }
        )
        
        # Send through alert manager
        try:
            alert_manager = AlertManager()
            
            # Force send critical alerts even during quiet hours
            force_send = level == AlertLevel.CRITICAL
            
            success = alert_manager.send_alert_with_escalation(
                alert,
                channels=['telegram'],
                respect_quiet_hours=True,
                force_send=force_send
            )
            
            if success:
                logger.info(f"✅ Intraday loss alert sent: {pnl_pct:+.3f}%")
            else:
                logger.error(f"❌ Failed to send intraday loss alert: {pnl_pct:+.3f}%")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize alert manager: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send intraday loss alert: {e}")
        return False