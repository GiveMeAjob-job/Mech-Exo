"""
Command Line Interface for Mech-Exo Trading System
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Mech-Exo Trading System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Daily execution command
    daily_parser = subparsers.add_parser("daily", help="Run daily trading routine")
    daily_parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                             help="Trading mode (paper or live)")
    daily_parser.add_argument("--config", type=str, default="config/",
                             help="Config directory path")
    
    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Execute specific trade")
    trade_parser.add_argument("--signal", type=str, required=True,
                             help="Trading signal (e.g., 'fxi')")
    trade_parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                             help="Trading mode")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--start", type=str, required=True,
                                help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", type=str, required=True,
                                help="End date (YYYY-MM-DD)")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8050,
                                 help="Dashboard port")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate trading reports")
    report_parser.add_argument("--date", type=str, default="today",
                              help="Report date (YYYY-MM-DD) or 'today'")
    report_parser.add_argument("--output", type=str, default=None,
                              help="Output file path for JSON report")
    report_parser.add_argument("--format", choices=["json", "html", "email"], default="json",
                              help="Output format (json, html, or email)")
    report_parser.add_argument("--html-output", type=str, default=None,
                              help="Output file path for HTML report")
    report_parser.add_argument("--slack", action="store_true",
                              help="Send digest to Slack channel")
    report_parser.add_argument("--slack-webhook", type=str, default=None,
                              help="Slack webhook URL (overrides config)")
    
    # Risk command
    risk_parser = subparsers.add_parser("risk", help="Risk management commands")
    risk_subparsers = risk_parser.add_subparsers(dest="risk_command", help="Risk subcommands")
    
    # Risk status subcommand
    status_parser = risk_subparsers.add_parser("status", help="Check current risk status")
    status_parser.add_argument("--nav", type=float, default=100000,
                              help="Portfolio NAV (default: 100000)")
    
    args = parser.parse_args()
    
    if args.command == "daily":
        print(f"Running daily routine in {args.mode} mode...")
        # TODO: Implement daily routine
        
    elif args.command == "trade":
        print(f"Executing trade signal: {args.signal} in {args.mode} mode...")
        # TODO: Implement trade execution
        
    elif args.command == "backtest":
        print(f"Running backtest from {args.start} to {args.end}...")
        # TODO: Implement backtest
        
    elif args.command == "dashboard":
        print(f"Launching dashboard on port {args.port}...")
        # TODO: Implement dashboard launch
        
    elif args.command == "report":
        _handle_report(args.date, args.output, args.format, args.html_output, args.slack, args.slack_webhook)
        
    elif args.command == "risk":
        if args.risk_command == "status":
            _handle_risk_status(args.nav)
        else:
            risk_parser.print_help()
        
    else:
        parser.print_help()
        sys.exit(1)


def _handle_report(date: str, output_path: str = None, format: str = "json", html_output: str = None):
    """Handle report command"""
    try:
        from mech_exo.reporting.daily import DailyReport
        
        # Generate daily report
        report = DailyReport(date=date)
        
        # Get summary
        summary = report.summary()
        
        # Print summary to console
        print(f"📊 Daily Report for {summary['date']}")
        print(f"{'─' * 40}")
        print(f"Daily P&L:     ${summary['daily_pnl']:>10,.2f}")
        print(f"Fees:          ${summary['fees']:>10,.2f}")
        print(f"Max Drawdown:  ${summary['max_dd']:>10,.2f}")
        print(f"Trade Count:   {summary['trade_count']:>10}")
        print(f"Volume:        ${summary['volume']:>10,.0f}")
        
        if summary['avg_slippage_bps'] > 0:
            print(f"Avg Slippage:  {summary['avg_slippage_bps']:>10.1f} bps")
        if summary['avg_routing_latency_ms'] > 0:
            print(f"Avg Latency:   {summary['avg_routing_latency_ms']:>10.1f} ms")
        
        if summary['strategies']:
            print(f"Strategies:    {', '.join(summary['strategies'])}")
        if summary['symbols']:
            print(f"Symbols:       {', '.join(summary['symbols'][:5])}")
            if len(summary['symbols']) > 5:
                print(f"               (+{len(summary['symbols']) - 5} more)")
        
        # Handle different output formats
        if format == "html":
            from mech_exo.reporting.html_renderer import HTMLReportRenderer
            
            # Create renderer and generate HTML
            renderer = HTMLReportRenderer()
            renderer.create_default_templates()
            
            output_file = Path(html_output) if html_output else Path(f"daily_report_{summary['date']}.html")
            html_content = renderer.render_daily_snapshot(report, output_file)
            print(f"\n🌐 HTML report saved to {output_file}")
            
        elif format == "email":
            from mech_exo.reporting.html_renderer import HTMLReportRenderer
            
            # Create renderer and generate email HTML
            renderer = HTMLReportRenderer()
            renderer.create_default_templates()
            html_content = renderer.render_email_digest(report)
            
            output_file = Path(f"daily_digest_{summary['date']}.html")
            output_file.write_text(html_content, encoding='utf-8')
            print(f"\n📧 Email digest saved to {output_file}")
            
        else:  # json format (default)
            # Export to JSON if output path specified
            if output_path:
                output_file = Path(output_path)
                json_output = report.to_json(output_file)
                print(f"\n✅ JSON report saved to {output_file}")
            else:
                # Print JSON to console
                json_output = report.to_json()
                print(f"\n📄 JSON Output:")
                print(json_output)
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        sys.exit(1)


def _handle_risk_status(nav: float):
    """Handle risk status command"""
    try:
        from mech_exo.risk import RiskChecker, Portfolio, Position
        from datetime import datetime
        
        # Create sample portfolio for testing
        portfolio = Portfolio(nav)
        
        # Add some sample positions for demonstration
        sample_positions = [
            Position("AAPL", 100, 150.0, 155.0, datetime.now(), "Technology"),
            Position("GOOGL", 50, 120.0, 125.0, datetime.now(), "Technology"),
            Position("SPY", 200, 400.0, 405.0, datetime.now(), "ETF")
        ]
        
        for pos in sample_positions:
            portfolio.add_position(pos)
        
        # Create risk checker
        checker = RiskChecker(portfolio)
        
        # Get risk status
        status_summary = checker.get_risk_status_summary()
        print(status_summary)
        
        # Get detailed report
        risk_report = checker.check()
        
        print(f"\n📊 Portfolio Summary:")
        print(f"   NAV: ${portfolio.current_nav:,.2f}")
        print(f"   Positions: {len(portfolio.positions)}")
        print(f"   Gross Exposure: {portfolio.gross_exposure / portfolio.current_nav:.1%}")
        print(f"   Unrealized P&L: ${portfolio.total_unrealized_pnl:,.2f}")
        
        if risk_report.get("warnings"):
            print(f"\n⚠️  Warnings ({len(risk_report['warnings'])}):")
            for warning in risk_report["warnings"]:
                print(f"   • {warning}")
        
        if risk_report.get("violations"):
            print(f"\n❌ Violations ({len(risk_report['violations'])}):")
            for violation in risk_report["violations"]:
                print(f"   • {violation}")
        
        checker.close()
        
    except Exception as e:
        print(f"❌ Risk status check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()