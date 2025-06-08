"""
HTML renderer for daily trading reports using Jinja2 templates
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .daily import DailyReport

logger = logging.getLogger(__name__)


class HTMLReportRenderer:
    """
    Generate HTML reports from daily trading data using Jinja2 templates
    
    Features:
    - Professional trading report layout
    - Performance metrics visualization
    - Responsive design for email/web viewing
    - Customizable styling and branding
    """

    def __init__(self, template_dir: str = None):
        """
        Initialize HTML renderer with template configuration
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        if template_dir is None:
            # Default to templates directory in reporting module
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters['currency'] = self._format_currency
        self.env.filters['percentage'] = self._format_percentage
        self.env.filters['bps'] = self._format_bps
        self.env.filters['number'] = self._format_number
        
        logger.info(f"HTMLReportRenderer initialized with templates at {self.template_dir}")

    def _format_currency(self, value: float) -> str:
        """Format currency values"""
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:.1f}K"
        else:
            return f"${value:,.2f}"

    def _format_percentage(self, value: float) -> str:
        """Format percentage values"""
        return f"{value:.2%}"

    def _format_bps(self, value: float) -> str:
        """Format basis points"""
        return f"{value:.1f} bps"

    def _format_number(self, value: float) -> str:
        """Format large numbers with K/M suffixes"""
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"{value/1_000:.1f}K"
        else:
            return f"{value:,.0f}"

    def render_daily_snapshot(self, report: DailyReport, output_path: Path = None) -> str:
        """
        Render complete daily snapshot HTML report
        
        Args:
            report: DailyReport instance with trading data
            output_path: Optional file path to save HTML output
            
        Returns:
            HTML string of the rendered report
        """
        try:
            # Prepare template data
            template_data = self._prepare_template_data(report)
            
            # Load and render template
            template = self.env.get_template('daily_snapshot.html')
            html_content = template.render(**template_data)
            
            # Save to file if path provided
            if output_path:
                output_path.write_text(html_content, encoding='utf-8')
                logger.info(f"Daily snapshot saved to {output_path}")
            
            return html_content
            
        except Exception as e:
            logger.error(f"Failed to render daily snapshot: {e}")
            raise

    def render_email_digest(self, report: DailyReport) -> str:
        """
        Render condensed email-friendly version of daily report
        
        Args:
            report: DailyReport instance
            
        Returns:
            HTML string optimized for email clients
        """
        try:
            # Prepare template data
            template_data = self._prepare_template_data(report)
            
            # Load and render email template
            template = self.env.get_template('email_digest.html')
            html_content = template.render(**template_data)
            
            return html_content
            
        except Exception as e:
            logger.error(f"Failed to render email digest: {e}")
            raise

    def _prepare_template_data(self, report: DailyReport) -> Dict[str, Any]:
        """
        Prepare data dictionary for template rendering
        
        Args:
            report: DailyReport instance
            
        Returns:
            Dictionary with template variables
        """
        summary = report.summary()
        breakdown = report.detailed_breakdown()
        
        # Calculate additional metrics
        pnl_color = "success" if summary['daily_pnl'] >= 0 else "danger"
        volume_millions = summary['volume'] / 1_000_000 if summary['volume'] > 0 else 0
        
        # Performance indicators
        performance_indicators = []
        if summary['daily_pnl'] > 0:
            performance_indicators.append(("Profitable Day", "success"))
        if summary['trade_count'] > 100:
            performance_indicators.append(("High Activity", "info"))
        if summary['avg_slippage_bps'] < 5:
            performance_indicators.append(("Good Execution", "success"))
        if summary['max_dd'] < -1000:
            performance_indicators.append(("Drawdown Alert", "warning"))
        
        # Top performers by symbol
        top_symbols = []
        if breakdown['by_symbol']:
            sorted_symbols = sorted(
                breakdown['by_symbol'].items(),
                key=lambda x: abs(x[1]['pnl']),
                reverse=True
            )[:5]
            for symbol, data in sorted_symbols:
                top_symbols.append({
                    'symbol': symbol,
                    'pnl': data['pnl'],
                    'trade_count': data['trade_count'],
                    'volume': data['volume']
                })
        
        # Hourly activity chart data
        hourly_data = breakdown['hourly_pnl']
        
        return {
            'report_date': summary['date'],
            'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'summary': summary,
            'breakdown': breakdown,
            'pnl_color': pnl_color,
            'volume_millions': volume_millions,
            'performance_indicators': performance_indicators,
            'top_symbols': top_symbols,
            'hourly_data': hourly_data,
            'has_activity': summary['trade_count'] > 0
        }

    def create_default_templates(self):
        """Create default HTML templates if they don't exist"""
        try:
            self._create_daily_snapshot_template()
            self._create_email_digest_template()
            self._create_css_styles()
            logger.info("Default templates created successfully")
        except Exception as e:
            logger.error(f"Failed to create default templates: {e}")
            raise

    def _create_daily_snapshot_template(self):
        """Create the main daily snapshot HTML template"""
        template_path = self.template_dir / "daily_snapshot.html"
        if not template_path.exists():
            template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Trading Snapshot - {{ report_date }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        {%- include 'styles.css' %}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <!-- Header -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card shadow-sm">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h1 class="h3 mb-1">
                                    <i class="fas fa-chart-line text-primary"></i>
                                    Daily Trading Snapshot
                                </h1>
                                <p class="text-muted mb-0">{{ report_date }} • Generated {{ generated_at }}</p>
                            </div>
                            <div class="text-end">
                                {% for indicator, badge_type in performance_indicators %}
                                <span class="badge bg-{{ badge_type }} me-1">{{ indicator }}</span>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Key Metrics -->
        <div class="row mb-4">
            <div class="col-md-3 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-dollar-sign fa-2x text-{{ pnl_color }} mb-2"></i>
                        <h4 class="text-{{ pnl_color }}">{{ summary.daily_pnl | currency }}</h4>
                        <p class="text-muted mb-0">Daily P&L</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-exchange-alt fa-2x text-info mb-2"></i>
                        <h4>{{ summary.trade_count | number }}</h4>
                        <p class="text-muted mb-0">Trades</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-chart-area fa-2x text-warning mb-2"></i>
                        <h4>{{ summary.volume | currency }}</h4>
                        <p class="text-muted mb-0">Volume</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-coins fa-2x text-secondary mb-2"></i>
                        <h4>{{ summary.fees | currency }}</h4>
                        <p class="text-muted mb-0">Fees</p>
                    </div>
                </div>
            </div>
        </div>

        {% if has_activity %}
        <!-- Performance Breakdown -->
        <div class="row mb-4">
            <div class="col-lg-6 mb-3">
                <div class="card h-100">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-chart-pie me-2"></i>Strategy Performance</h5>
                    </div>
                    <div class="card-body">
                        {% if breakdown.by_strategy %}
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Strategy</th>
                                        <th class="text-end">P&L</th>
                                        <th class="text-end">Trades</th>
                                        <th class="text-end">Volume</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for strategy, data in breakdown.by_strategy.items() %}
                                    <tr>
                                        <td>{{ strategy }}</td>
                                        <td class="text-end text-{% if data.pnl >= 0 %}success{% else %}danger{% endif %}">
                                            {{ data.pnl | currency }}
                                        </td>
                                        <td class="text-end">{{ data.trade_count }}</td>
                                        <td class="text-end">{{ data.volume | currency }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                        <p class="text-muted">No strategy data available</p>
                        {% endif %}
                    </div>
                </div>
            </div>
            <div class="col-lg-6 mb-3">
                <div class="card h-100">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-chart-bar me-2"></i>Top Symbols</h5>
                    </div>
                    <div class="card-body">
                        {% if top_symbols %}
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th class="text-end">P&L</th>
                                        <th class="text-end">Trades</th>
                                        <th class="text-end">Volume</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for symbol_data in top_symbols %}
                                    <tr>
                                        <td><strong>{{ symbol_data.symbol }}</strong></td>
                                        <td class="text-end text-{% if symbol_data.pnl >= 0 %}success{% else %}danger{% endif %}">
                                            {{ symbol_data.pnl | currency }}
                                        </td>
                                        <td class="text-end">{{ symbol_data.trade_count }}</td>
                                        <td class="text-end">{{ symbol_data.volume | currency }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                        <p class="text-muted">No symbol data available</p>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>

        <!-- Execution Quality -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-tachometer-alt me-2"></i>Execution Quality</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4 text-center">
                                <h4 class="{% if summary.avg_slippage_bps <= 5 %}text-success{% elif summary.avg_slippage_bps <= 10 %}text-warning{% else %}text-danger{% endif %}">
                                    {{ summary.avg_slippage_bps | bps }}
                                </h4>
                                <p class="text-muted">Average Slippage</p>
                            </div>
                            <div class="col-md-4 text-center">
                                <h4 class="{% if summary.max_dd >= -500 %}text-success{% elif summary.max_dd >= -1000 %}text-warning{% else %}text-danger{% endif %}">
                                    {{ summary.max_dd | currency }}
                                </h4>
                                <p class="text-muted">Max Drawdown</p>
                            </div>
                            <div class="col-md-4 text-center">
                                <h4 class="text-info">{{ summary.symbols | length }}</h4>
                                <p class="text-muted">Symbols Traded</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        {% else %}
        <!-- No Activity Message -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-body text-center py-5">
                        <i class="fas fa-pause-circle fa-3x text-muted mb-3"></i>
                        <h4 class="text-muted">No Trading Activity</h4>
                        <p class="text-muted">No trades were executed on {{ report_date }}</p>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Footer -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="text-center text-muted">
                    <small>Generated by Mech-Exo Trading System • {{ generated_at }}</small>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
            template_path.write_text(template_content, encoding='utf-8')

    def _create_email_digest_template(self):
        """Create the email digest HTML template"""
        template_path = self.template_dir / "email_digest.html"
        if not template_path.exists():
            template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Trading Digest - {{ report_date }}</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f8f9fa; }
        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #007bff, #6610f2); color: white; padding: 30px 20px; text-align: center; }
        .content { padding: 30px 20px; }
        .metric-row { display: flex; justify-content: space-between; margin-bottom: 15px; padding: 15px; background: #f8f9fa; border-radius: 6px; }
        .metric-label { font-weight: bold; color: #495057; }
        .metric-value { color: #007bff; font-weight: bold; }
        .success { color: #28a745 !important; }
        .danger { color: #dc3545 !important; }
        .footer { background: #f8f9fa; padding: 20px; text-align: center; color: #6c757d; font-size: 12px; }
        .no-activity { text-align: center; padding: 40px 20px; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Daily Trading Digest</h1>
            <p>{{ report_date }}</p>
        </div>
        <div class="content">
            {% if has_activity %}
            <div class="metric-row">
                <span class="metric-label">Daily P&L:</span>
                <span class="metric-value {{ 'success' if summary.daily_pnl >= 0 else 'danger' }}">
                    {{ summary.daily_pnl | currency }}
                </span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Trades Executed:</span>
                <span class="metric-value">{{ summary.trade_count }}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Total Volume:</span>
                <span class="metric-value">{{ summary.volume | currency }}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Fees Paid:</span>
                <span class="metric-value">{{ summary.fees | currency }}</span>
            </div>
            {% if summary.avg_slippage_bps > 0 %}
            <div class="metric-row">
                <span class="metric-label">Avg Slippage:</span>
                <span class="metric-value">{{ summary.avg_slippage_bps | bps }}</span>
            </div>
            {% endif %}
            
            {% if top_symbols %}
            <h3>Top Performers</h3>
            {% for symbol_data in top_symbols[:3] %}
            <div class="metric-row">
                <span class="metric-label">{{ symbol_data.symbol }}:</span>
                <span class="metric-value {{ 'success' if symbol_data.pnl >= 0 else 'danger' }}">
                    {{ symbol_data.pnl | currency }} ({{ symbol_data.trade_count }} trades)
                </span>
            </div>
            {% endfor %}
            {% endif %}
            {% else %}
            <div class="no-activity">
                <h3>🔇 No Trading Activity</h3>
                <p>No trades were executed today.</p>
            </div>
            {% endif %}
        </div>
        <div class="footer">
            Generated by Mech-Exo Trading System<br>
            {{ generated_at }}
        </div>
    </div>
</body>
</html>"""
            template_path.write_text(template_content, encoding='utf-8')

    def _create_css_styles(self):
        """Create additional CSS styles"""
        styles_path = self.template_dir / "styles.css"
        if not styles_path.exists():
            css_content = """
.card {
    border: none;
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
}

.card-header {
    background-color: #f8f9fa;
    border-bottom: 1px solid #dee2e6;
}

.text-success { color: #28a745 !important; }
.text-danger { color: #dc3545 !important; }
.text-warning { color: #ffc107 !important; }
.text-info { color: #17a2b8 !important; }
.text-primary { color: #007bff !important; }

.bg-success { background-color: #28a745 !important; }
.bg-danger { background-color: #dc3545 !important; }
.bg-warning { background-color: #ffc107 !important; }
.bg-info { background-color: #17a2b8 !important; }

.table th {
    border-top: none;
    font-weight: 600;
    color: #495057;
}

.badge {
    font-size: 0.75em;
}

@media print {
    .container-fluid { margin: 0; padding: 0; }
    .card { box-shadow: none; border: 1px solid #dee2e6; }
}
"""
            styles_path.write_text(css_content, encoding='utf-8')


def render_daily_snapshot(date: str = "today", output_path: str = None) -> str:
    """
    Convenience function to render daily snapshot HTML
    
    Args:
        date: Date string (YYYY-MM-DD) or "today"
        output_path: Optional file path to save HTML
        
    Returns:
        HTML string of rendered report
    """
    from .daily import DailyReport
    
    # Generate daily report
    report = DailyReport(date=date)
    
    # Create renderer and templates
    renderer = HTMLReportRenderer()
    renderer.create_default_templates()
    
    # Render HTML
    output_file = Path(output_path) if output_path else None
    html_content = renderer.render_daily_snapshot(report, output_file)
    
    return html_content


def render_email_digest(date: str = "today") -> str:
    """
    Convenience function to render email digest HTML
    
    Args:
        date: Date string (YYYY-MM-DD) or "today"
        
    Returns:
        HTML string of email digest
    """
    from .daily import DailyReport
    
    # Generate daily report
    report = DailyReport(date=date)
    
    # Create renderer and templates
    renderer = HTMLReportRenderer()
    renderer.create_default_templates()
    
    # Render email HTML
    html_content = renderer.render_email_digest(report)
    
    return html_content