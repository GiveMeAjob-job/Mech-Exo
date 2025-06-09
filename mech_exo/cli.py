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
    
    # Dash command (alias for dashboard)
    dash_parser = subparsers.add_parser("dash", help="Launch Dash dashboard")
    dash_parser.add_argument("--port", type=int, default=8050,
                            help="Dashboard port")
    dash_parser.add_argument("--debug", action="store_true",
                            help="Run in debug mode")
    
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
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    backtest_parser.add_argument("--start", type=str, required=True,
                                help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", type=str, required=True,
                                help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--cash", type=float, default=100000,
                                help="Initial cash (default: 100000)")
    backtest_parser.add_argument("--signals", type=str, default=None,
                                help="CSV file with trading signals")
    backtest_parser.add_argument("--symbols", type=str, nargs="+", default=["SPY", "QQQ"],
                                help="Symbols to test (default: SPY QQQ)")
    backtest_parser.add_argument("--html", type=str, default=None,
                                help="Output HTML tear-sheet file")
    
    # Walk-forward analysis command
    walkforward_parser = subparsers.add_parser("walkforward", help="Run walk-forward analysis")
    walkforward_parser.add_argument("--start", type=str, required=True,
                                   help="Start date (YYYY-MM-DD)")
    walkforward_parser.add_argument("--end", type=str, required=True,
                                   help="End date (YYYY-MM-DD)")
    walkforward_parser.add_argument("--window", type=str, default="36M12M",
                                   help="Window format: traintest (e.g., 36M12M, 24M6M)")
    walkforward_parser.add_argument("--cash", type=float, default=100000,
                                   help="Initial cash (default: 100000)")
    walkforward_parser.add_argument("--signals", type=str, default=None,
                                   help="CSV file with trading signals")
    walkforward_parser.add_argument("--rankings", type=str, default=None,
                                   help="CSV file with idea rankings")
    walkforward_parser.add_argument("--symbols", type=str, nargs="+", default=["SPY", "QQQ"],
                                   help="Symbols to test (default: SPY QQQ)")
    walkforward_parser.add_argument("--html", type=str, default=None,
                                   help="Output HTML walk-forward report")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export trading data to CSV/Parquet")
    export_parser.add_argument("--table", required=True,
                              choices=['fills', 'positions', 'backtest_metrics', 'drift_metrics'],
                              help="Table to export")
    export_parser.add_argument("--range", dest="date_range", required=True,
                              help="Date range: YYYY-MM-DD:YYYY-MM-DD or keywords (last7d, last30d, etc.)")
    export_parser.add_argument("--fmt", dest="format", default="parquet",
                              choices=['csv', 'parquet'],
                              help="Output format (default: parquet)")
    export_parser.add_argument("--gzip", action="store_true", default=False,
                              help="Compress output with gzip")
    export_parser.add_argument("--output-dir", default="exports",
                              help="Output directory (default: exports)")
    export_parser.add_argument("--verbose", "-v", action="store_true", default=False,
                              help="Verbose output")
    
    # QuantStats report command
    qs_parser = subparsers.add_parser("qs-report", help="Generate QuantStats PDF performance report")
    qs_parser.add_argument("--from", dest="start_date", required=True,
                          help="Start date (YYYY-MM-DD)")
    qs_parser.add_argument("--to", dest="end_date", required=True,
                          help="End date (YYYY-MM-DD or 'today')")
    qs_parser.add_argument("--pdf", dest="output_pdf", required=True,
                          help="Output PDF file path")
    qs_parser.add_argument("--title", default=None,
                          help="Report title (optional)")
    qs_parser.add_argument("--benchmark", action="store_true", default=True,
                          help="Include benchmark comparison (default: True)")
    qs_parser.add_argument("--no-benchmark", dest="benchmark", action="store_false",
                          help="Exclude benchmark comparison")
    qs_parser.add_argument("--verbose", "-v", action="store_true", default=False,
                          help="Verbose output")
    
    # Decay command for alpha decay analysis
    decay_parser = subparsers.add_parser("decay", help="Alpha decay analysis and monitoring")
    decay_parser.add_argument("--fixture", type=str, default=None,
                             help="CSV fixture file with factor data")
    decay_parser.add_argument("--export", type=str, default=None,
                             help="Export decay metrics to CSV file")
    decay_parser.add_argument("--lookback", type=int, default=365,
                             help="Days of historical data to analyze (default: 365)")
    decay_parser.add_argument("--threshold", type=float, default=7.0,
                             help="Alert threshold in days (default: 7.0)")
    decay_parser.add_argument("--dry-run", action="store_true", default=False,
                             help="Dry run mode - don't send actual alerts")
    
    # Optuna commands for hyper-parameter optimization
    optuna_parser = subparsers.add_parser("optuna-init", help="Initialize Optuna study database")
    optuna_parser.add_argument("--study-file", type=str, default="studies/factor_opt.db",
                              help="Path to study database file")
    optuna_parser.add_argument("--study-name", type=str, default="factor_weight_optimization",
                              help="Name of the optimization study")
    
    optuna_run_parser = subparsers.add_parser("optuna-run", help="Run Optuna optimization trials")
    optuna_run_parser.add_argument("--n-trials", type=int, default=50,
                                  help="Number of optimization trials to run")
    optuna_run_parser.add_argument("--study-file", type=str, default="studies/factor_opt.db",
                                  help="Path to study database file")
    optuna_run_parser.add_argument("--study-name", type=str, default="factor_weight_optimization",
                                  help="Name of the optimization study")
    optuna_run_parser.add_argument("--n-jobs", type=int, default=1,
                                  help="Number of parallel jobs (default: 1, max: 8)")
    optuna_run_parser.add_argument("--timeout", type=int, default=None,
                                  help="Timeout in seconds for optimization")
    optuna_run_parser.add_argument("--export-best", type=str, default=None,
                                  help="Export best trial to YAML file")
    optuna_run_parser.add_argument("--stage", action="store_true", default=False,
                                  help="Copy YAML to staging/ and create git commit")
    optuna_run_parser.add_argument("--notify-progress", action="store_true", default=False,
                                  help="Send Telegram notifications on progress")
    optuna_run_parser.add_argument("--progress-interval", type=int, default=10,
                                  help="Print progress every N trials")
    
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
        
    elif args.command == "dash":
        _handle_dash(args.port, args.debug)
        
    elif args.command == "report":
        _handle_report(args.date, args.output, args.format, args.html_output, args.slack, args.slack_webhook)
        
    elif args.command == "risk":
        if args.risk_command == "status":
            _handle_risk_status(args.nav)
        else:
            risk_parser.print_help()
    
    elif args.command == "backtest":
        _handle_backtest(args.start, args.end, args.cash, args.signals, args.symbols, args.html)
    
    elif args.command == "walkforward":
        _handle_walkforward(args.start, args.end, args.window, args.cash, 
                           args.signals, args.rankings, args.symbols, args.html)
    
    elif args.command == "export":
        _handle_export(args.table, args.date_range, args.format, args.gzip, 
                      args.output_dir, args.verbose)
    
    elif args.command == "qs-report":
        _handle_qs_report(args.start_date, args.end_date, args.output_pdf, 
                         args.title, args.benchmark, args.verbose)
    
    elif args.command == "decay":
        _handle_decay(args.fixture, args.export, args.lookback, args.threshold, args.dry_run)
    
    elif args.command == "optuna-init":
        _handle_optuna_init(args.study_file, args.study_name)
    
    elif args.command == "optuna-run":
        _handle_optuna_run(args.n_trials, args.study_file, args.study_name, 
                          args.n_jobs, args.timeout, args.export_best,
                          args.stage, args.notify_progress, args.progress_interval)
        
    else:
        parser.print_help()
        sys.exit(1)


def _handle_report(date: str, output_path: str = None, format: str = "json", 
                  html_output: str = None, slack: bool = False, slack_webhook: str = None):
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
        
        # Send to Slack if requested
        if slack:
            try:
                from mech_exo.reporting.slack_alerter import SlackAlerter
                
                alerter = SlackAlerter(webhook_url=slack_webhook)
                success = alerter.send_daily_digest(report)
                alerter.close()
                
                if success:
                    print(f"\n💬 Daily digest sent to Slack")
                else:
                    print(f"\n❌ Failed to send digest to Slack")
                    
            except Exception as e:
                print(f"\n❌ Slack integration error: {e}")
        
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


def _handle_dash(port: int, debug: bool = False):
    """Handle dash command - launch dashboard"""
    try:
        print(f"🚀 Starting Mech-Exo Dashboard on port {port}...")
        
        from mech_exo.reporting.dash_app import create_dash_app
        
        # Create and run the dash app
        app = create_dash_app()
        
        print(f"📊 Dashboard available at: http://localhost:{port}")
        print(f"🏥 Health check endpoint: http://localhost:{port}/healthz")
        print("Press Ctrl+C to stop")
        
        app.run(
            debug=debug,
            host='0.0.0.0',
            port=port,
            dev_tools_hot_reload=debug
        )
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        sys.exit(1)


def _handle_backtest(start: str, end: str, cash: float, signals_file: str = None, 
                    symbols: list = None, html_output: str = None):
    """Handle backtest command"""
    try:
        from mech_exo.backtest import Backtester, create_simple_signals
        import pandas as pd
        
        print(f"🔬 Running backtest from {start} to {end}...")
        print(f"💰 Initial cash: ${cash:,.0f}")
        
        # Initialize backtester
        backtester = Backtester(start=start, end=end, cash=cash)
        
        # Load or create signals
        if signals_file:
            print(f"📂 Loading signals from {signals_file}")
            signals = pd.read_csv(signals_file, index_col=0, parse_dates=True)
            signals = signals.astype(bool)
        else:
            print(f"🎯 Creating simple buy-and-hold signals for {symbols}")
            signals = create_simple_signals(symbols, start, end, frequency='monthly')
        
        # Run backtest
        results = backtester.run(signals)
        
        # Display results
        print(results.summary())
        
        # Additional metrics display
        m = results.metrics
        print(f"\n📈 Additional Insights:")
        print(f"   Cost Impact: {m.get('cost_drag_annual', 0):.3%} annual drag from fees")
        print(f"   Risk-Adj Return: {m.get('calmar_ratio', 0):.2f} (CAGR/Max DD)")
        print(f"   Trading Frequency: {m.get('total_trades', 0) / ((pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25):.1f} trades/year")
        
        # Export HTML if requested
        if html_output:
            try:
                print(f"\n📄 Generating HTML tear-sheet...")
                html_path = results.export_html(html_output, strategy_name="Custom Strategy")
                print(f"    ✅ HTML tear-sheet saved to: {html_path}")
                print(f"    🌐 Open in browser to view interactive charts")
            except Exception as e:
                print(f"    ❌ HTML export failed: {e}")
        
        print(f"\n✅ Backtest completed successfully")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        sys.exit(1)


def _handle_walkforward(start: str, end: str, window: str, cash: float,
                       signals_file: str = None, rankings_file: str = None, 
                       symbols: list = None, html_output: str = None):
    """Handle walk-forward analysis command"""
    try:
        from mech_exo.backtest.walk_forward import WalkForwardAnalyzer, make_walk_windows
        from mech_exo.backtest import create_simple_signals
        import pandas as pd
        
        print(f"🚶 Running walk-forward analysis from {start} to {end}...")
        print(f"💰 Initial cash: ${cash:,.0f}")
        print(f"🪟 Window configuration: {window}")
        
        # Parse window format (e.g., "36M12M" -> train="36M", test="12M")
        if len(window) >= 4 and window[-1] in 'MmDdYy':
            # Find split point - look for pattern like "36M12M"
            import re
            match = re.match(r'(\d+[MmDdYy])(\d+[MmDdYy])', window)
            if match:
                train_period, test_period = match.groups()
            else:
                train_period, test_period = "36M", "12M"
        else:
            train_period, test_period = "36M", "12M"
        
        print(f"   Train period: {train_period}, Test period: {test_period}")
        
        # Preview windows
        windows = make_walk_windows(start, end, train_period, test_period)
        print(f"   Generated {len(windows)} walk-forward windows")
        
        if len(windows) == 0:
            print("❌ No valid windows generated. Check date range and window sizes.")
            return
        
        # Initialize analyzer
        analyzer = WalkForwardAnalyzer(train_period, test_period)
        
        # Load or create signals/rankings
        is_rankings = False
        signal_params = {'n_top': 3, 'rebal_freq': 'monthly'}
        
        if rankings_file:
            print(f"📂 Loading idea rankings from {rankings_file}")
            data = pd.read_csv(rankings_file, index_col=0, parse_dates=True)
            is_rankings = True
        elif signals_file:
            print(f"📂 Loading signals from {signals_file}")
            data = pd.read_csv(signals_file, index_col=0, parse_dates=True)
            data = data.astype(bool)
        else:
            print(f"🎯 Creating simple buy-and-hold signals for {symbols}")
            data = create_simple_signals(symbols, start, end, frequency='monthly')
        
        # Run walk-forward analysis
        results = analyzer.run_walk_forward(
            start=start,
            end=end,
            signals_or_rankings=data,
            initial_cash=cash,
            is_rankings=is_rankings,
            signal_params=signal_params
        )
        
        # Display results table
        print(f"\n{results.summary_table()}")
        
        # Export HTML if requested
        if html_output:
            try:
                print(f"\n📄 Generating HTML walk-forward report...")
                html_path = results.export_html(html_output, strategy_name="Walk-Forward Strategy")
                print(f"    ✅ HTML report saved to: {html_path}")
                print(f"    🌐 Open in browser to view merged equity curve & segment analysis")
            except Exception as e:
                print(f"    ❌ HTML export failed: {e}")
        
        print(f"\n✅ Walk-forward analysis completed successfully")
        
    except Exception as e:
        print(f"❌ Walk-forward analysis failed: {e}")
        sys.exit(1)


def _handle_export(table: str, date_range: str, format: str, gzip_compress: bool,
                  output_dir: str, verbose: bool):
    """Handle data export command"""
    import logging
    
    try:
        from mech_exo.cli.export import DataExporter
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        # Initialize exporter
        exporter = DataExporter()
        
        # Override exports directory if specified
        if output_dir != 'exports':
            from pathlib import Path
            exporter.exports_dir = Path(output_dir)
            exporter.exports_dir.mkdir(exist_ok=True)
        
        # Parse date range
        start_date, end_date = exporter.parse_date_range(date_range)
        
        print(f"🔍 Exporting {table} data from {start_date} to {end_date}")
        print(f"📁 Output directory: {exporter.exports_dir}")
        print(f"📄 Format: {format}" + (" (gzipped)" if gzip_compress else ""))
        
        # Export data
        result = exporter.export_data(table, start_date, end_date, format, gzip_compress)
        
        if result['success']:
            print(f"✅ Export completed successfully!")
            print(f"   • Rows exported: {result['rows']:,}")
            print(f"   • File path: {result['file_path']}")
            print(f"   • File size: {result['file_size']:,} bytes ({result['file_size'] / 1024:.1f} KB)")
            if 'columns' in result:
                print(f"   • Columns: {len(result['columns'])}")
                if verbose:
                    print(f"   • Column names: {', '.join(result['columns'])}")
        else:
            print(f"❌ Export failed: {result['message']}")
            if result['rows'] > 0:
                print(f"   • Found {result['rows']} rows but export failed")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Export command failed: {e}")
        if verbose:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)
    
    finally:
        try:
            exporter.close()
        except:
            pass


def _handle_qs_report(start_date: str, end_date: str, output_pdf: str, 
                     title: str = None, include_benchmark: bool = True, verbose: bool = False):
    """Handle QuantStats PDF report generation command"""
    import logging
    from datetime import date, datetime
    
    try:
        from mech_exo.reporting.quantstats_report import generate_performance_pdf
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        # Parse dates
        if end_date.lower() == 'today':
            end_date_parsed = date.today()
        else:
            end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        print(f"📊 Generating QuantStats performance report...")
        print(f"📅 Period: {start_date_parsed} to {end_date_parsed}")
        print(f"📄 Output: {output_pdf}")
        if title:
            print(f"📋 Title: {title}")
        print(f"📈 Benchmark: {'Included' if include_benchmark else 'Excluded'}")
        
        # Generate the PDF report
        result = generate_performance_pdf(
            start_date=start_date_parsed,
            end_date=end_date_parsed,
            output_path=output_pdf,
            title=title,
            include_benchmark=include_benchmark
        )
        
        if result['success']:
            print(f"✅ QuantStats PDF report generated successfully!")
            print(f"   • File path: {result['file_path']}")
            print(f"   • File size: {result['file_size']:,} bytes ({result['file_size'] / 1024:.1f} KB)")
            print(f"   • Title: {result['title']}")
            print(f"   • Returns analyzed: {result['returns_count']} observations")
            
            # Check file size warning
            max_size_mb = 5 * 1024 * 1024  # 5 MB
            if result['file_size'] > max_size_mb:
                print(f"⚠️  Warning: PDF size ({result['file_size']/1024/1024:.1f} MB) is quite large")
            
            print(f"🌐 Open the PDF file to view comprehensive performance analysis")
            
        else:
            print(f"❌ QuantStats PDF generation failed: {result['message']}")
            
            # Provide helpful dependency installation info
            if 'dependencies' in result['message'].lower():
                print(f"\n📦 To install required dependencies:")
                print(f"   pip install quantstats pdfkit")
                print(f"   # Also install wkhtmltopdf:")
                print(f"   # Ubuntu: apt-get install wkhtmltopdf")
                print(f"   # macOS: brew install wkhtmltopdf")
                print(f"   # Windows: Download from https://wkhtmltopdf.org/downloads.html")
            
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ QuantStats report command failed: {e}")
        if verbose:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)


def _handle_decay(fixture_file: str = None, export_path: str = None, 
                 lookback_days: int = 365, alert_threshold: float = 7.0, 
                 dry_run: bool = False):
    """Handle alpha decay analysis command"""
    import os
    import pandas as pd
    from datetime import datetime, date
    
    try:
        print(f"🔬 Alpha Decay Analysis")
        print(f"   • Lookback period: {lookback_days} days")
        print(f"   • Alert threshold: {alert_threshold} days")
        print(f"   • Mode: {'Dry run' if dry_run else 'Live'}")
        
        if fixture_file:
            print(f"   • Using fixture: {fixture_file}")
            
            # Load fixture data
            fixture_df = pd.read_csv(fixture_file)
            print(f"   • Loaded {len(fixture_df)} rows from fixture")
            
            # Validate fixture structure
            required_cols = ['date', 'factor', 'factor_value', 'forward_return']
            missing_cols = [col for col in required_cols if col not in fixture_df.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns in fixture: {missing_cols}")
                sys.exit(1)
            
            # Convert date column
            fixture_df['date'] = pd.to_datetime(fixture_df['date'])
            
            # Group by factor for analysis
            decay_results = []
            
            from mech_exo.research.alpha_decay import AlphaDecayEngine
            decay_engine = AlphaDecayEngine(window=120, min_periods=20)
            
            for factor_name in fixture_df['factor'].unique():
                factor_data = fixture_df[fixture_df['factor'] == factor_name].copy()
                factor_data = factor_data.set_index('date').sort_index()
                
                if len(factor_data) >= 20:  # Minimum data requirement
                    try:
                        decay_metrics = decay_engine.calc_half_life(
                            factor_data['factor_value'],
                            factor_data['forward_return']
                        )
                        
                        decay_metrics.update({
                            'factor_name': factor_name,
                            'calculation_date': datetime.now(),
                            'data_points': len(factor_data)
                        })
                        
                        decay_results.append(decay_metrics)
                        
                        half_life = decay_metrics.get('half_life', float('nan'))
                        ic = decay_metrics.get('latest_ic', float('nan'))
                        
                        if pd.notna(half_life):
                            print(f"   • {factor_name}: {half_life:.1f}d half-life, IC: {ic:.3f}")
                        else:
                            print(f"   • {factor_name}: Unable to calculate half-life")
                        
                    except Exception as e:
                        print(f"   • {factor_name}: Error - {e}")
                else:
                    print(f"   • {factor_name}: Insufficient data ({len(factor_data)} points)")
            
            if not decay_results:
                print(f"❌ No decay metrics calculated")
                sys.exit(1)
            
            # Check for rapid decay factors
            rapid_decay_factors = []
            for metrics in decay_results:
                half_life = metrics.get('half_life')
                if half_life is not None and not pd.isna(half_life) and half_life < alert_threshold:
                    rapid_decay_factors.append({
                        'factor': metrics['factor_name'],
                        'half_life': half_life,
                        'latest_ic': metrics.get('latest_ic', 0)
                    })
            
            if rapid_decay_factors:
                print(f"\n⚠️  {len(rapid_decay_factors)} factors with rapid decay (< {alert_threshold}d):")
                for factor_info in rapid_decay_factors:
                    print(f"   • {factor_info['factor']}: {factor_info['half_life']:.1f}d")
                
                # Test alert logic if not dry run
                if not dry_run:
                    try:
                        from mech_exo.utils.alerts import TelegramAlerter
                        
                        telegram_config = {
                            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
                        }
                        
                        if telegram_config['bot_token'] and telegram_config['chat_id']:
                            alerter = TelegramAlerter(telegram_config)
                            
                            # Create alert message
                            alert_message = "⚠️ *Alpha\\\\-decay Alert \\\\(CLI Test\\\\)*\\n\\n"
                            
                            for factor_info in rapid_decay_factors:
                                factor_name = alerter.escape_markdown(factor_info['factor'])
                                half_life = factor_info['half_life']
                                ic = factor_info['latest_ic']
                                
                                alert_message += f"📉 *{factor_name}*: half\\\\-life {half_life:.1f}d \\\\(<{alert_threshold}\\\\)\\n"
                                alert_message += f"    Latest IC: {ic:.3f}\\n\\n"
                            
                            alert_message += f"🔍 *Threshold*: {alert_threshold} days\\n"
                            today_str = date.today().strftime('%Y-%m-%d').replace('-', '\\\\-')
                            alert_message += f"📅 *Date*: {today_str}"
                            
                            success = alerter.send_message(alert_message)
                            
                            if success:
                                print(f"📱 Telegram alert sent successfully")
                            else:
                                print(f"❌ Failed to send Telegram alert")
                        else:
                            print(f"💡 Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to send alerts")
                            
                    except Exception as e:
                        print(f"❌ Alert test failed: {e}")
            else:
                print(f"\n✅ No factors meet alert criteria")
            
            # Export results if requested
            if export_path:
                # Convert to DataFrame for export
                export_data = []
                for metrics in decay_results:
                    export_data.append({
                        'factor': metrics['factor_name'],
                        'half_life': metrics.get('half_life'),
                        'latest_ic': metrics.get('latest_ic'),
                        'ic_observations': metrics.get('ic_observations', 0),
                        'ic_mean': metrics.get('ic_mean'),
                        'ic_std': metrics.get('ic_std'),
                        'ic_trend': metrics.get('ic_trend'),
                        'data_points': metrics.get('data_points', 0),
                        'calculation_date': metrics['calculation_date'],
                        'status': metrics.get('status', 'calculated')
                    })
                
                export_df = pd.DataFrame(export_data)
                export_df.to_csv(export_path, index=False)
                
                print(f"\n📄 Exported {len(export_data)} decay metrics to {export_path}")
                
                # Validate export - ensure no NaN in half_life for CI
                nan_count = export_df['half_life'].isna().sum()
                if nan_count > 0:
                    print(f"⚠️  Warning: {nan_count} factors have NaN half-life values")
                else:
                    print(f"✅ All factors have valid half-life calculations")
                
                print(f"   • File size: {os.path.getsize(export_path)} bytes")
        
        else:
            # Run on live data using the flow components
            print(f"   • Using live data from database")
            
            # This would call the actual alpha decay flow components
            # For now, show what would happen
            print(f"   • This would load {lookback_days} days of factor data")
            print(f"   • Calculate decay metrics for all factors")
            print(f"   • Store results in database")
            print(f"   • Send alerts for factors < {alert_threshold}d half-life")
            
            if dry_run:
                print(f"   • Dry run mode: alerts would be logged, not sent")
        
        print(f"\n✅ Alpha decay analysis completed")
        
    except Exception as e:
        print(f"❌ Alpha decay analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _handle_optuna_init(study_file: str, study_name: str):
    """Handle Optuna study initialization command"""
    try:
        print(f"🔬 Initializing Optuna study...")
        print(f"   • Study file: {study_file}")
        print(f"   • Study name: {study_name}")
        
        from optimize.opt_factor_weights import create_study_db, FactorWeightOptimizer
        
        # Create study database
        success = create_study_db(study_file)
        
        if success:
            print(f"✅ Study database created successfully")
            
            # Test data loading
            print(f"📊 Testing data loading...")
            optimizer = FactorWeightOptimizer(study_file)
            data = optimizer.load_historical_data()
            
            factor_count = len(data['factor_scores'])
            returns_count = len(data['returns'])
            
            print(f"   • Factor records: {factor_count:,}")
            print(f"   • Return records: {returns_count:,}")
            
            if factor_count > 0:
                print(f"✅ Study ready for optimization")
                print(f"💡 Next steps:")
                print(f"   1. Run optimization: exo optuna-run --n-trials 10")
                print(f"   2. View results: optuna-dashboard --storage sqlite:///{study_file}")
            else:
                print(f"⚠️  Warning: No factor data available")
                
        else:
            print(f"❌ Failed to create study database")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print(f"💡 Install with: pip install optuna optuna-dashboard")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Optuna init failed: {e}")
        sys.exit(1)


def _handle_optuna_run(n_trials: int, study_file: str, study_name: str,
                      n_jobs: int, timeout: int, export_best: str,
                      stage: bool, notify_progress: bool, progress_interval: int):
    """Handle enhanced Optuna optimization run command with batch capabilities"""
    try:
        import os
        import time
        from datetime import date
        from pathlib import Path
        
        # Cap n_jobs to reasonable limits
        max_jobs = min(n_jobs, os.cpu_count() or 1, 8)
        if max_jobs != n_jobs:
            print(f"⚠️  Capping n_jobs from {n_jobs} to {max_jobs} (CPU/safety limit)")
            n_jobs = max_jobs
        
        print(f"🎯 Starting Enhanced Optuna Optimization...")
        print(f"   • Trials: {n_trials}")
        print(f"   • Study file: {study_file}")
        print(f"   • Study name: {study_name}")
        print(f"   • Parallel jobs: {n_jobs}")
        print(f"   • Progress interval: {progress_interval}")
        print(f"   • Notifications: {'Yes' if notify_progress else 'No'}")
        if timeout:
            print(f"   • Timeout: {timeout} seconds")
        
        from optimize.opt_factor_weights import FactorWeightOptimizer
        from mech_exo.utils.opt_callbacks import create_optuna_callback, interrupt_handler
        
        # Initialize optimizer
        optimizer = FactorWeightOptimizer(study_file)
        
        # Load historical data
        print(f"📊 Loading historical data...")
        data = optimizer.load_historical_data()
        
        if len(data['factor_scores']) == 0:
            print(f"❌ No factor data available for optimization")
            sys.exit(1)
        
        # Create/load study with enhanced sampler and pruner
        study = optimizer.create_enhanced_study(study_name)
        
        # Create progress callback
        callback = create_optuna_callback(progress_interval, notify_progress)
        
        print(f"🚀 Running {n_trials} optimization trials...")
        print(f"   • Sampler: {study.sampler.__class__.__name__}")
        print(f"   • Pruner: {study.pruner.__class__.__name__}")
        
        start_time = time.time()
        
        # Define objective function with data
        def objective(trial):
            return optimizer.objective_function(
                trial, data['factor_scores'], data['returns']
            )
        
        # Setup automatic YAML export path
        auto_export_path = f"factors/factors_opt_{date.today().strftime('%Y-%m-%d')}.yml"
        if not export_best:
            export_best = auto_export_path
            print(f"   • Auto-export to: {export_best}")
        
        try:
            # Run optimization with callbacks
            study.optimize(
                objective, 
                n_trials=n_trials,
                n_jobs=n_jobs,
                timeout=timeout,
                callbacks=[callback],
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print(f"\n🛑 Optimization interrupted by user!")
            interrupt_handler(study, export_best)
            return
        
        elapsed_time = time.time() - start_time
        
        # Print comprehensive results
        print(f"\n✅ Optimization completed in {elapsed_time:.1f} seconds")
        print(f"📈 Final Results:")
        print(f"   • Best Sharpe ratio: {study.best_value:.4f}")
        print(f"   • Best trial: #{study.best_trial.number}")
        print(f"   • Total trials: {len(study.trials)}")
        print(f"   • Rate: {len(study.trials) / elapsed_time:.1f} trials/sec")
        
        if study.best_trial.user_attrs:
            max_dd = study.best_trial.user_attrs.get('max_drawdown', 'N/A')
            violations = study.best_trial.user_attrs.get('constraint_violations', 'N/A')
            print(f"   • Best max drawdown: {max_dd}")
            print(f"   • Constraint violations: {violations}")
        
        # Export best trial to YAML
        print(f"\n📄 Exporting best trial to {export_best}...")
        success = _export_best_trial_enhanced(study, export_best)
        if success:
            print(f"✅ Best trial exported successfully")
            
            # Handle staging if requested
            if stage:
                _handle_staging(export_best)
        else:
            print(f"❌ Failed to export best trial")
        
        print(f"\n🎯 Batch optimization complete!")
        print(f"💡 View dashboard: optuna-dashboard --storage sqlite:///{study_file}")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print(f"💡 Install with: pip install optuna optuna-dashboard")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Optuna run failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _export_best_trial_enhanced(study, output_file: str) -> bool:
    """Export best trial parameters to YAML file with enhanced metadata"""
    try:
        import yaml
        from datetime import datetime
        from pathlib import Path
        
        best_trial = study.best_trial
        
        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract factor weights from best trial
        factor_weights = {}
        other_params = {}
        
        for key, value in best_trial.params.items():
            if key.startswith('weight_'):
                factor_name = key.replace('weight_', '')
                factor_weights[factor_name] = value
            else:
                other_params[key] = value
        
        # Get comprehensive metadata from trial attributes
        trial_attrs = best_trial.user_attrs or {}
        
        # Create enhanced YAML structure
        export_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'optimization_method': 'optuna_tpe',
                'study_name': study.study_name,
                'best_trial_number': best_trial.number,
                'best_sharpe_ratio': study.best_value,
                'max_drawdown': trial_attrs.get('max_drawdown', 0),
                'total_return': trial_attrs.get('total_return', 0),
                'volatility': trial_attrs.get('volatility', 0),
                'constraints_satisfied': trial_attrs.get('constraints_satisfied', False),
                'constraint_violations': trial_attrs.get('constraint_violations', 0),
                'total_trials': len(study.trials),
                'sampler': study.sampler.__class__.__name__,
                'pruner': study.pruner.__class__.__name__
            },
            'factors': {
                'fundamental': {
                    name: {
                        'weight': round(factor_weights.get(name, 0), 4), 
                        'direction': 'higher_better',
                        'category': 'fundamental'
                    }
                    for name in ['pe_ratio', 'return_on_equity', 'revenue_growth', 'earnings_growth']
                },
                'technical': {
                    name: {
                        'weight': round(factor_weights.get(name, 0), 4), 
                        'direction': 'higher_better',
                        'category': 'technical'
                    }
                    for name in ['rsi_14', 'momentum_12_1', 'volatility_ratio']
                },
                'sentiment': {
                    name: {
                        'weight': round(factor_weights.get(name, 0), 4), 
                        'direction': 'higher_better',
                        'category': 'sentiment'
                    }
                    for name in ['news_sentiment', 'analyst_revisions']
                }
            },
            'hyperparameters': {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in other_params.items()
            }
        }
        
        # Write to YAML file with preserved order
        with open(output_path, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False, indent=2, sort_keys=False)
        
        print(f"   • File size: {output_path.stat().st_size} bytes")
        print(f"   • Factor count: {len(factor_weights)}")
        print(f"   • Hyperparameters: {len(other_params)}")
        
        return True
        
    except Exception as e:
        print(f"Enhanced export failed: {e}")
        return False


def _handle_staging(yaml_file: str) -> bool:
    """Handle staging and git operations for optimized factors"""
    try:
        from pathlib import Path
        from datetime import date
        import shutil
        
        print(f"📦 Staging optimized factors...")
        
        # Create staging directory
        staging_dir = Path("config/staging")
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy YAML to staging with timestamped name
        yaml_path = Path(yaml_file)
        staged_name = f"factors_optuna_{date.today().strftime('%Y%m%d_%H%M%S')}.yml"
        staged_path = staging_dir / staged_name
        
        shutil.copy2(yaml_path, staged_path)
        print(f"   • Copied to: {staged_path}")
        
        # Try to create git commit if GitPython is available
        try:
            import git
            
            repo = git.Repo(".")
            
            # Add the staged file
            repo.index.add([str(staged_path)])
            
            # Create commit
            commit_msg = f"Add Optuna factors {date.today().strftime('%Y-%m-%d')}"
            commit = repo.index.commit(commit_msg)
            
            print(f"   • Git commit: {commit.hexsha[:8]}")
            print(f"   • Commit message: {commit_msg}")
            
            # Try to push if remote is configured
            try:
                origin = repo.remote('origin')
                origin.push()
                print(f"   • Pushed to remote successfully")
            except Exception as e:
                print(f"   • Remote push skipped: {e}")
                
        except ImportError:
            print(f"   • GitPython not available - skipping git operations")
        except Exception as e:
            print(f"   • Git operations failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Staging failed: {e}")
        return False


def _export_best_trial(study, output_file: str) -> bool:
    """Export best trial parameters to YAML file (legacy function)"""
    return _export_best_trial_enhanced(study, output_file)


if __name__ == "__main__":
    main()