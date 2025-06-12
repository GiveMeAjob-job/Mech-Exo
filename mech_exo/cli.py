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
    
    # ML Feature Building
    ml_features_parser = subparsers.add_parser("ml-features", help="Build ML feature matrices")
    ml_features_parser.add_argument("--start", type=str, required=True,
                                   help="Start date (YYYY-MM-DD)")
    ml_features_parser.add_argument("--end", type=str, required=True,
                                   help="End date (YYYY-MM-DD)")
    ml_features_parser.add_argument("--symbols", type=str, default=None,
                                   help="Comma-separated list of symbols (optional)")
    ml_features_parser.add_argument("--output-dir", type=str, default="data/features",
                                   help="Output directory (default: data/features)")
    ml_features_parser.add_argument("--verbose", action="store_true", default=False,
                                   help="Enable verbose logging")
    
    # ML Model Training
    ml_train_parser = subparsers.add_parser("ml-train", help="Train ML models for alpha signals")
    ml_train_parser.add_argument("--algo", type=str, default="lightgbm",
                                choices=["lightgbm", "xgboost"],
                                help="Algorithm to use (default: lightgbm)")
    ml_train_parser.add_argument("--lookback", type=str, default="3y",
                                help="Training data lookback period (e.g., 3y, 1y, 180d)")
    ml_train_parser.add_argument("--cv", type=int, default=5,
                                help="Number of cross-validation folds (default: 5)")
    ml_train_parser.add_argument("--n-iter", type=int, default=30,
                                help="Number of hyperparameter search iterations (default: 30)")
    ml_train_parser.add_argument("--seed", type=int, default=42,
                                help="Random seed for reproducibility (default: 42)")
    ml_train_parser.add_argument("--features-dir", type=str, default="data/features",
                                help="Directory containing feature files (default: data/features)")
    ml_train_parser.add_argument("--models-dir", type=str, default="models",
                                help="Directory to save models and metrics (default: models)")
    ml_train_parser.add_argument("--verbose", action="store_true", default=False,
                                help="Enable verbose logging")
    
    # ML SHAP Analysis
    ml_shap_parser = subparsers.add_parser("ml-shap", help="Generate SHAP feature importance reports")
    ml_shap_parser.add_argument("--model", type=str, required=True,
                               help="Path to trained model file (required)")
    ml_shap_parser.add_argument("--date", type=str, default="today",
                               help="Date for feature data (YYYY-MM-DD or 'today')")
    ml_shap_parser.add_argument("--features-dir", type=str, default="data/features",
                               help="Directory containing feature files (default: data/features)")
    ml_shap_parser.add_argument("--output-dir", type=str, default="reports/shap",
                               help="Output directory for reports (default: reports/shap)")
    ml_shap_parser.add_argument("--png", type=str, default=None,
                               help="Custom PNG filename (optional)")
    ml_shap_parser.add_argument("--html", type=str, default=None,
                               help="Custom HTML filename (optional)")
    ml_shap_parser.add_argument("--top", type=int, default=20,
                               help="Number of top features to display (default: 20)")
    ml_shap_parser.add_argument("--algo", type=str, default=None,
                               choices=["lightgbm", "xgboost"],
                               help="Algorithm type (auto-detected if not specified)")
    ml_shap_parser.add_argument("--verbose", action="store_true", default=False,
                               help="Enable verbose logging")
    
    # ML Prediction/Inference
    ml_predict_parser = subparsers.add_parser("ml-predict", help="Generate ML alpha predictions")
    ml_predict_parser.add_argument("--model", type=str, required=True,
                                  help="Path to trained model file (required)")
    ml_predict_parser.add_argument("--date", type=str, default="today",
                                  help="Target date for prediction (YYYY-MM-DD or 'today')")
    ml_predict_parser.add_argument("--symbols", type=str, default=None,
                                  help="Comma-separated list of symbols (optional)")
    ml_predict_parser.add_argument("--features-dir", type=str, default="data/features",
                                  help="Directory containing feature files (default: data/features)")
    ml_predict_parser.add_argument("--outfile", type=str, default="ml_scores.csv",
                                  help="Output CSV file path (default: ml_scores.csv)")
    ml_predict_parser.add_argument("--normalize", action="store_true", default=True,
                                  help="Normalize scores to 0-1 range (default: True)")
    ml_predict_parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                                  help="Disable score normalization")
    ml_predict_parser.add_argument("--verbose", action="store_true", default=False,
                                  help="Enable verbose logging")
    
    # Idea Scoring with ML Integration
    score_parser = subparsers.add_parser("score", help="Score and rank investment ideas")
    score_parser.add_argument("--symbols", type=str, nargs="+", default=None,
                             help="Symbols to score (default: all universe)")
    score_parser.add_argument("--use-ml", action="store_true", default=False,
                             help="Integrate ML scores into ranking")
    score_parser.add_argument("--ml-scores", type=str, default=None,
                             help="Path to ML scores CSV file")
    score_parser.add_argument("--output", type=str, default="idea_scores.csv",
                             help="Output CSV file (default: idea_scores.csv)")
    score_parser.add_argument("--config", type=str, default="config/factors.yml",
                             help="Factors configuration file")
    score_parser.add_argument("--top", type=int, default=None,
                             help="Return only top N results")
    score_parser.add_argument("--verbose", "-v", action="store_true", default=False,
                             help="Enable verbose logging")
    
    # Weight Adjustment Command
    weight_parser = subparsers.add_parser("weight-adjust", help="Test ML weight adjustment")
    weight_parser.add_argument("--baseline", type=float, required=True,
                              help="Baseline strategy Sharpe ratio")
    weight_parser.add_argument("--ml", type=float, required=True,
                              help="ML strategy Sharpe ratio")
    weight_parser.add_argument("--current", type=float, required=True,
                              help="Current ML weight (0.0-0.50)")
    weight_parser.add_argument("--notify", action="store_true", default=False,
                              help="Send test Telegram notification")
    weight_parser.add_argument("--dry-run", action="store_true", default=False,
                              help="Dry run mode - don't update YAML or send real notifications")
    
    # Canary A/B Testing Commands
    canary_parser = subparsers.add_parser("canary", help="Canary A/B testing management")
    canary_subparsers = canary_parser.add_subparsers(dest="canary_command", help="Canary subcommands")
    
    # Canary status command
    canary_status_parser = canary_subparsers.add_parser("status", help="Check canary allocation status")
    canary_status_parser.add_argument("--verbose", "-v", action="store_true", default=False,
                                     help="Show detailed status including breach counter")
    
    # Canary enable command
    canary_enable_parser = canary_subparsers.add_parser("enable", help="Enable canary allocation")
    canary_enable_parser.add_argument("--pct", type=float, default=None,
                                     help="Set allocation percentage (0.01-0.30, default: keep current)")
    canary_enable_parser.add_argument("--reset-breaches", action="store_true", default=False,
                                     help="Reset consecutive breach counter to 0")
    
    # Canary disable command  
    canary_disable_parser = canary_subparsers.add_parser("disable", help="Disable canary allocation")
    canary_disable_parser.add_argument("--reason", type=str, default="manual",
                                      help="Disable reason (default: manual)")
    
    # Canary test command
    canary_test_parser = canary_subparsers.add_parser("test", help="Test canary threshold logic")
    canary_test_parser.add_argument("--sharpe", type=float, required=True,
                                   help="Test Sharpe ratio value")
    canary_test_parser.add_argument("--days", type=int, default=1,
                                   help="Simulate consecutive days (default: 1)")
    canary_test_parser.add_argument("--reset", action="store_true", default=False,
                                   help="Reset breach counter before test")
    
    # Canary performance command
    canary_perf_parser = canary_subparsers.add_parser("performance", help="Show canary performance summary")
    canary_perf_parser.add_argument("--days", type=int, default=30,
                                   help="Days of history to show (default: 30)")
    canary_perf_parser.add_argument("--export", type=str, default=None,
                                   help="Export to CSV file")
    
    # Canary config command
    canary_config_parser = canary_subparsers.add_parser("config", help="Show canary configuration")
    canary_config_parser.add_argument("--edit", action="store_true", default=False,
                                     help="Open config file for editing")
    
    # Trading Cost Analysis Commands
    costs_parser = subparsers.add_parser("costs", help="Trading cost & slippage analysis")
    costs_parser.add_argument("--from", dest="start_date", required=True,
                             help="Start date (YYYY-MM-DD)")
    costs_parser.add_argument("--to", dest="end_date", required=True,
                             help="End date (YYYY-MM-DD)")
    costs_parser.add_argument("--html", type=str, default=None,
                             help="Export HTML report to file (e.g., trade_costs.html)")
    costs_parser.add_argument("--csv", type=str, default=None,
                             help="Export detailed data to CSV file")
    costs_parser.add_argument("--update-table", action="store_true", default=False,
                             help="Update trade_costs summary table in database")
    costs_parser.add_argument("--verbose", "-v", action="store_true", default=False,
                             help="Show detailed output")
    
    # Incident Rollback Commands
    rollback_parser = subparsers.add_parser("rollback", help="Emergency rollback tooling")
    rollback_parser.add_argument("--flow", type=str, choices=['ml_reweight', 'canary_perf', 'daily_flow', 'data_pipeline'],
                                help="Flow to rollback (e.g., ml_reweight, canary_perf)")
    rollback_parser.add_argument("--config", type=str,
                                help="Config file to rollback (e.g., config/factors.yml)")
    rollback_parser.add_argument("--database", action="store_true", default=False,
                                help="Rollback database flags (canary state, etc.)")
    rollback_parser.add_argument("--to", dest="target_timestamp", required=True,
                                help="Target timestamp (YYYY-MM-DDTHH:MM:SS)")
    rollback_parser.add_argument("--dry-run", action="store_true", default=False,
                                help="Simulate rollback without making changes")
    rollback_parser.add_argument("--force", action="store_true", default=False,
                                help="Skip confirmation prompt (dangerous)")
    rollback_parser.add_argument("--history", action="store_true", default=False,
                                help="Show recent rollback history")
    
    # Runbook command
    runbook_parser = subparsers.add_parser("runbook", help="On-call runbook and incident management")
    runbook_subparsers = runbook_parser.add_subparsers(dest="runbook_action", help="Runbook actions")
    
    # Export runbook
    export_parser = runbook_subparsers.add_parser("export", help="Export runbook to markdown")
    export_parser.add_argument("--output", type=str, default="oncall_runbook.md",
                              help="Output markdown file path")
    
    # Lookup incident
    lookup_parser = runbook_subparsers.add_parser("lookup", help="Lookup incident by ID")
    lookup_parser.add_argument("incident_id", type=str,
                              help="Incident ID (e.g., system_down, trading_halted)")
    
    # List incidents
    list_parser = runbook_subparsers.add_parser("list", help="List all available incidents")
    
    # Test escalation
    escalate_parser = runbook_subparsers.add_parser("escalate", help="Test escalation system")
    escalate_parser.add_argument("incident_id", type=str,
                                help="Incident ID to escalate")
    escalate_parser.add_argument("--level", choices=["telegram", "email", "phone"],
                                default="telegram", help="Escalation level")
    escalate_parser.add_argument("--details", type=str, default="Test escalation",
                                help="Additional details for escalation")
    
    # Capital management command
    from mech_exo.cli.capital import create_capital_parser
    create_capital_parser(subparsers)
    
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
    
    elif args.command == "ml-features":
        _handle_ml_features(args.start, args.end, args.symbols, args.output_dir, args.verbose)
    
    elif args.command == "ml-train":
        _handle_ml_train(args.algo, args.lookback, args.cv, args.n_iter, args.seed,
                        args.features_dir, args.models_dir, args.verbose)
    
    elif args.command == "ml-shap":
        _handle_ml_shap(args.model, args.date, args.features_dir, args.output_dir,
                       args.png, args.html, args.top, args.algo, args.verbose)
    
    elif args.command == "ml-predict":
        _handle_ml_predict(args.model, args.date, args.symbols, args.features_dir,
                          args.outfile, args.normalize, args.verbose)
    
    elif args.command == "score":
        _handle_score(args.symbols, args.use_ml, args.ml_scores, args.output,
                     args.config, args.top, args.verbose)
    
    elif args.command == "weight-adjust":
        _handle_weight_adjust(args.baseline, args.ml, args.current, args.notify, args.dry_run)
        
    elif args.command == "canary":
        _handle_canary_command(args)
        
    elif args.command == "costs":
        _handle_costs_command(args)
        
    elif args.command == "rollback":
        _handle_rollback_command(args)
        
    elif args.command == "runbook":
        _handle_runbook_command(args)
        
    elif args.command == "capital":
        from mech_exo.cli.capital import handle_capital_command
        handle_capital_command(args)
        
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


def _handle_ml_features(start_date: str, end_date: str, symbols: str = None, 
                       output_dir: str = "data/features", verbose: bool = False):
    """Handle ML feature building command"""
    import logging
    
    try:
        from mech_exo.ml.features import build_features_cli
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        # Parse symbols
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
            
        print(f"🏗️ Building ML features...")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Output dir: {output_dir}")
        
        if symbol_list:
            print(f"   Symbols: {len(symbol_list)} specified ({', '.join(symbol_list[:5])}{'...' if len(symbol_list) > 5 else ''})")
        else:
            print("   Symbols: All available")
            
        # Build features
        build_features_cli(
            start_date=start_date,
            end_date=end_date,
            symbols=symbol_list,
            output_dir=output_dir
        )
        
        print("✅ ML feature building completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the ML module dependencies are installed")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ ML feature building failed: {e}")
        if verbose:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)


def _handle_ml_train(algorithm: str, lookback: str, cv_folds: int, n_iter: int, seed: int,
                    features_dir: str, models_dir: str, verbose: bool = False):
    """Handle ML training command"""
    import logging
    
    try:
        from mech_exo.ml.train_ml import train_ml_cli
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        print(f"🚀 Starting ML model training...")
        print(f"   Algorithm: {algorithm}")
        print(f"   Lookback: {lookback}")
        print(f"   CV folds: {cv_folds}")
        print(f"   Hyperparameter iterations: {n_iter}")
        print(f"   Features dir: {features_dir}")
        print(f"   Models dir: {models_dir}")
        print(f"   Random seed: {seed}")
        
        # Run training
        results = train_ml_cli(
            algorithm=algorithm,
            lookback=lookback,
            cv_folds=cv_folds,
            n_iter=n_iter,
            seed=seed,
            features_dir=features_dir,
            models_dir=models_dir
        )
        
        print(f"\n✅ ML training completed successfully!")
        print(f"   Best AUC: {results['best_auc']:.4f}")
        print(f"   Training samples: {results['training_samples']:,}")
        print(f"   Features: {results['features']}")
        print(f"   Model saved: {results['model_file']}")
        print(f"   Metrics saved: {results['metrics_file']}")
        
        # Display metrics
        metrics = results['metrics']
        print(f"\n📊 Performance Metrics:")
        print(f"   Mean AUC: {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}")
        print(f"   Mean IC: {metrics['mean_ic']:.4f} ± {metrics['std_ic']:.4f}")
        print(f"   Mean Accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        
        if results['best_auc'] >= 0.60:
            print(f"\n🎯 Model meets AUC threshold (≥0.60)!")
        else:
            print(f"\n⚠️  Model AUC below threshold (<0.60)")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure ML dependencies are installed:")
        print("  pip install lightgbm scikit-learn scipy")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ ML training failed: {e}")
        if verbose:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)


def _handle_ml_shap(model_path: str, date: str, features_dir: str, output_dir: str,
                   png_name: str, html_name: str, top_k: int, algorithm: str, 
                   verbose: bool = False):
    """Handle ML SHAP analysis command"""
    import logging
    
    try:
        from mech_exo.ml.report_ml import shap_report_cli
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        print(f"🔍 Starting SHAP feature importance analysis...")
        print(f"   Model: {model_path}")
        print(f"   Date: {date}")
        print(f"   Features dir: {features_dir}")
        print(f"   Output dir: {output_dir}")
        print(f"   Top features: {top_k}")
        if algorithm:
            print(f"   Algorithm: {algorithm}")
        
        # Generate SHAP report
        metadata = shap_report_cli(
            model_path=model_path,
            date=date,
            features_dir=features_dir,
            output_dir=output_dir,
            png_name=png_name,
            html_name=html_name,
            top_k=top_k,
            algorithm=algorithm
        )
        
        print(f"\n✅ SHAP report generated successfully!")
        print(f"   Samples analyzed: {metadata['samples_analyzed']:,}")
        print(f"   Features: {metadata['features_count']}")
        print(f"   Algorithm: {metadata['algorithm']}")
        
        print(f"\n📊 Report Files:")
        print(f"   PNG Summary: {metadata['png_path']} ({metadata['png_size_mb']:.2f} MB)")
        print(f"   HTML Interactive: {metadata['html_path']} ({metadata['html_size_mb']:.2f} MB)")
        
        print(f"\n🔥 Top 5 Features:")
        for i, (feature, score) in enumerate(zip(metadata['top_features'][:5], 
                                                metadata['feature_importance_scores'][:5]), 1):
            print(f"   {i}. {feature}: {score:.4f}")
        
        # File size warnings
        if metadata['png_size_mb'] > 2.0:
            print(f"\n⚠️  PNG file is large ({metadata['png_size_mb']:.2f} MB)")
        if metadata['html_size_mb'] > 5.0:
            print(f"⚠️  HTML file is large ({metadata['html_size_mb']:.2f} MB)")
        
        print(f"\n🎯 SHAP analysis completed! Open {metadata['html_path']} for interactive exploration.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure SHAP dependencies are installed:")
        print("  pip install shap matplotlib")
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure model file and feature directory exist")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ SHAP analysis failed: {e}")
        if verbose:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)


def _handle_ml_predict(model_path: str, date: str, symbols: str, features_dir: str,
                      output_file: str, normalize: bool, verbose: bool = False):
    """Handle ML prediction command"""
    import logging
    
    try:
        from mech_exo.ml.predict import predict_cli
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        print(f"🔮 Starting ML alpha prediction...")
        print(f"   Model: {model_path}")
        print(f"   Date: {date}")
        print(f"   Features dir: {features_dir}")
        print(f"   Output file: {output_file}")
        print(f"   Normalize: {normalize}")
        
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
            print(f"   Symbols: {len(symbol_list)} specified ({', '.join(symbol_list[:5])}{'...' if len(symbol_list) > 5 else ''})")
        else:
            print(f"   Symbols: All available")
        
        # Generate predictions
        metadata = predict_cli(
            model_path=model_path,
            date=date,
            symbols=symbols,
            features_dir=features_dir,
            output_file=output_file,
            normalize=normalize
        )
        
        if metadata['success']:
            print(f"\n✅ ML prediction completed successfully!")
            print(f"   Predictions: {metadata['predictions']:,}")
            print(f"   Algorithm: {metadata['algorithm']}")
            print(f"   Target date: {metadata['target_date']}")
            print(f"   Output: {metadata['output_file']} ({metadata['file_size']} bytes)")
            print(f"   Score range: [{metadata['score_range'][0]:.4f}, {metadata['score_range'][1]:.4f}]")
            
            print(f"\n🏆 Top 5 Predictions:")
            for i, (symbol, score) in enumerate(zip(metadata['top_symbols'][:5], 
                                                   metadata['top_scores'][:5]), 1):
                print(f"   {i}. {symbol}: {score:.4f}")
            
            print(f"\n🎯 Predictions saved to {metadata['output_file']}!")
            
        else:
            print(f"❌ Prediction failed: {metadata['message']}")
            sys.exit(1)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure ML dependencies are installed:")
        print("  pip install lightgbm scikit-learn")
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure model file and feature directory exist")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ ML prediction failed: {e}")
        if verbose:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)


def _handle_score(symbols: list, use_ml: bool, ml_scores_file: str, output_file: str,
                 config_path: str, top_n: int, verbose: bool = False):
    """Handle idea scoring command"""
    import logging
    
    try:
        from mech_exo.scoring.cli import score_cli
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        print(f"🎯 Starting idea scoring...")
        print(f"   Use ML: {use_ml}")
        print(f"   Config: {config_path}")
        print(f"   Output: {output_file}")
        
        if symbols:
            print(f"   Symbols: {len(symbols)} specified ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
        else:
            print("   Symbols: Full universe")
        
        if use_ml and ml_scores_file:
            print(f"   ML scores file: {ml_scores_file}")
        elif use_ml:
            print("   ML scores: From database (latest)")
        
        if top_n:
            print(f"   Top results: {top_n}")
        
        # Run scoring
        metadata = score_cli(
            symbols=symbols,
            use_ml=use_ml,
            ml_scores_file=ml_scores_file,
            output_file=output_file,
            config_path=config_path,
            top_n=top_n,
            verbose=verbose
        )
        
        if metadata['success']:
            print(f"\n✅ Idea scoring completed successfully!")
            print(f"   Results: {metadata['results']:,}")
            print(f"   ML integration: {metadata['use_ml']}")
            print(f"   Output: {metadata['output_file']} ({metadata['file_size']} bytes)")
            
            print(f"\n🏆 Top 5 Ideas:")
            for i, (symbol, score) in enumerate(zip(metadata['top_symbols'], metadata['top_scores']), 1):
                print(f"   {i}. {symbol}: {score:.2f}")
            
            if use_ml and metadata.get('ml_scores_available'):
                print(f"   ML weight: {metadata.get('ml_weight_used', 'N/A')}")
                print(f"📊 ML scores successfully integrated!")
            elif use_ml:
                print(f"⚠️  ML requested but scores not available - using traditional scoring")
            
            print(f"\n🎯 Results saved to {metadata['output_file']}!")
            
        else:
            print(f"❌ Scoring failed: {metadata['message']}")
            sys.exit(1)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure scoring dependencies are available")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Scoring failed: {e}")
        if verbose:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)


def _handle_weight_adjust(baseline_sharpe: float, ml_sharpe: float, current_weight: float,
                         notify: bool = False, dry_run: bool = False):
    """Handle weight adjustment testing command"""
    try:
        from mech_exo.scoring.weight_utils import compute_new_weight, validate_weight_bounds
        from mech_exo.utils.alerts import TelegramAlerter
        from mech_exo.utils.config import ConfigManager
        import os
        
        print(f"⚖️ ML Weight Adjustment Test")
        print(f"{'─' * 40}")
        print(f"Baseline Sharpe: {baseline_sharpe:.3f}")
        print(f"ML Sharpe:       {ml_sharpe:.3f}")
        print(f"Current Weight:  {current_weight:.3f}")
        print(f"Mode:            {'DRY RUN' if dry_run else 'TEST'}")
        
        # Validate current weight
        is_valid, error_msg = validate_weight_bounds(current_weight)
        if not is_valid:
            print(f"❌ {error_msg}")
            sys.exit(1)
        
        # Compute new weight
        new_weight, rule = compute_new_weight(baseline_sharpe, ml_sharpe, current_weight)
        
        # Display results
        delta_sharpe = ml_sharpe - baseline_sharpe
        weight_changed = abs(new_weight - current_weight) > 0.001
        
        print(f"\n📊 Analysis Results:")
        print(f"Sharpe Delta:    {delta_sharpe:+.3f}")
        print(f"Adjustment Rule: {rule}")
        print(f"New Weight:      {new_weight:.3f}")
        
        if weight_changed:
            direction = "↗️" if new_weight > current_weight else "↘️"
            print(f"Change:          {current_weight:.3f} {direction} {new_weight:.3f}")
        else:
            print(f"Change:          No adjustment needed")
        
        # Send notification if requested
        if notify and weight_changed:
            print(f"\n📱 Testing Telegram Notification...")
            
            try:
                # Load Telegram configuration
                config_manager = ConfigManager()
                try:
                    telegram_config = config_manager.load_config('alerts').get('telegram', {})
                except:
                    telegram_config = {
                        'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                        'chat_id': os.getenv('TELEGRAM_CHAT_ID')
                    }
                
                if not telegram_config.get('bot_token') or not telegram_config.get('chat_id'):
                    print("⚠️ Telegram credentials not configured")
                    print("   Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
                    print("   Or configure in config/alerts.yml")
                else:
                    alerter = TelegramAlerter(telegram_config)
                    
                    success = alerter.send_weight_change(
                        old_w=current_weight,
                        new_w=new_weight,
                        sharpe_ml=ml_sharpe,
                        sharpe_base=baseline_sharpe,
                        adjustment_rule=rule,
                        dry_run=True  # Always use dry-run for CLI testing
                    )
                    
                    if success:
                        print("✅ Test notification sent successfully")
                    else:
                        print("❌ Test notification failed")
                        
            except Exception as e:
                print(f"❌ Notification test failed: {e}")
        
        elif notify and not weight_changed:
            print(f"\n📱 Notification skipped (no weight change)")
        
        # Show example Telegram message format
        if notify or weight_changed:
            print(f"\n📝 Example Telegram Message:")
            print(f"   ⚖️ ML Weight Auto-Adjusted")
            print(f"   • Weight: {current_weight:.2f} {'↗️' if new_weight > current_weight else '↘️' if new_weight < current_weight else '➡️'} {new_weight:.2f}")
            print(f"   • Δ Sharpe: {delta_sharpe:+.3f} ({ml_sharpe:.3f} vs {baseline_sharpe:.3f})")
            print(f"   • Rule: {rule}")
            print(f"   • Time: 2024-01-15 09:30:00 (commit abc123)")
        
        print(f"\n✅ Weight adjustment test completed")
        
        if dry_run:
            print(f"💡 Note: This was a dry-run test. No files were modified.")
        
        return True
        
    except Exception as e:
        print(f"❌ Weight adjustment test failed: {e}")
        sys.exit(1)


def _handle_canary_command(args):
    """Handle canary A/B testing commands"""
    try:
        from mech_exo.execution.allocation import (
            get_allocation_config, update_canary_enabled, is_canary_enabled,
            get_canary_allocation, check_hysteresis_trigger, reset_breach_counter,
            get_consecutive_breach_days
        )
        from mech_exo.reporting.query import get_ab_test_summary
        import subprocess
        import os
        
        if args.canary_command == "status":
            print("🧪 Canary A/B Testing Status")
            print("=" * 40)
            
            # Get basic status
            enabled = is_canary_enabled()
            allocation = get_canary_allocation()
            config = get_allocation_config()
            
            print(f"Enabled: {'✅ YES' if enabled else '❌ NO'}")
            print(f"Allocation: {allocation:.1%}")
            
            if enabled:
                # Get performance summary
                try:
                    summary = get_ab_test_summary(days=30)
                    print(f"Status: {summary['status_badge']}")
                    print(f"Sharpe diff: {summary['sharpe_diff']:+.3f}")
                    print(f"Days analyzed: {summary['days_analyzed']}")
                except Exception as e:
                    print(f"Performance data: ❌ Error ({e})")
                
                # Show hysteresis info if verbose
                if args.verbose:
                    breach_days = get_consecutive_breach_days()
                    disable_rule = config.get('disable_rule', {})
                    
                    print(f"\nHysteresis Status:")
                    print(f"  Consecutive breach days: {breach_days}")
                    print(f"  Required for disable: {disable_rule.get('confirm_days', 2)}")
                    print(f"  Sharpe threshold: {disable_rule.get('sharpe_low', 0.0)}")
                    print(f"  Min observations: {disable_rule.get('min_observations', 21)}")
            
        elif args.canary_command == "enable":
            print("🔄 Enabling canary allocation...")
            
            # Update allocation percentage if provided
            if args.pct is not None:
                if not (0.01 <= args.pct <= 0.30):
                    print("❌ Allocation percentage must be between 1% and 30%")
                    sys.exit(1)
                
                # Update config file
                config = get_allocation_config()
                config['canary_allocation'] = args.pct
                
                # Write updated config
                import yaml
                with open('config/allocation.yml', 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                
                print(f"✅ Updated allocation to {args.pct:.1%}")
            
            # Reset breach counter if requested
            if args.reset_breaches:
                reset_breach_counter()
                print("✅ Reset consecutive breach counter")
            
            # Enable canary
            success = update_canary_enabled(True)
            if success:
                print("✅ Canary allocation enabled")
            else:
                print("❌ Failed to enable canary")
                sys.exit(1)
                
        elif args.canary_command == "disable":
            print(f"⏸️ Disabling canary allocation (reason: {args.reason})...")
            
            success = update_canary_enabled(False)
            if success:
                print("✅ Canary allocation disabled")
                
                # Reset breach counter when manually disabled
                reset_breach_counter()
                print("✅ Reset consecutive breach counter")
            else:
                print("❌ Failed to disable canary")
                sys.exit(1)
                
        elif args.canary_command == "test":
            print(f"🧪 Testing hysteresis logic with Sharpe {args.sharpe:.3f}")
            
            if args.reset:
                reset_breach_counter()
                print("✅ Reset breach counter before test")
            
            # Simulate consecutive days
            for day in range(args.days):
                result = check_hysteresis_trigger(args.sharpe)
                
                print(f"Day {day + 1}:")
                print(f"  Breach: {'Yes' if result['is_breach'] else 'No'}")
                print(f"  Consecutive days: {result['current_breach_days']}")
                print(f"  Should trigger: {'Yes' if result['should_trigger'] else 'No'}")
                
                if result['should_trigger']:
                    print("  🚨 AUTO-DISABLE WOULD TRIGGER!")
                    break
            
            # Reset after test
            reset_breach_counter()
            print("✅ Reset breach counter after test")
            
        elif args.canary_command == "performance":
            print(f"📊 Canary Performance Summary ({args.days} days)")
            print("=" * 50)
            
            try:
                summary = get_ab_test_summary(days=args.days)
                
                print(f"Status: {summary['status_badge']}")
                print(f"Days analyzed: {summary['days_analyzed']}")
                print(f"Canary NAV: ${summary.get('canary_nav', 0):,.0f}")
                print(f"Base NAV: ${summary.get('base_nav', 0):,.0f}")
                print(f"Sharpe difference: {summary['sharpe_diff']:+.3f}")
                print(f"Canary outperforming: {'✅' if summary['sharpe_diff'] > 0 else '❌'}")
                
                if args.export:
                    # Export detailed data to CSV
                    from mech_exo.reporting.query import get_canary_equity, get_base_equity
                    import pandas as pd
                    
                    canary_data = get_canary_equity(days=args.days)
                    base_data = get_base_equity(days=args.days)
                    
                    # Merge data
                    merged = pd.merge(canary_data, base_data, on='date', how='outer', suffixes=('_canary', '_base'))
                    merged.to_csv(args.export, index=False)
                    
                    print(f"✅ Exported {len(merged)} records to {args.export}")
                    
            except Exception as e:
                print(f"❌ Failed to get performance data: {e}")
                sys.exit(1)
                
        elif args.canary_command == "config":
            print("📋 Canary Configuration")
            print("=" * 30)
            
            config = get_allocation_config()
            
            print(f"Enabled: {config.get('canary_enabled', True)}")
            print(f"Allocation: {config.get('canary_allocation', 0.1):.1%}")
            
            disable_rule = config.get('disable_rule', {})
            print(f"\nAuto-disable Rules:")
            print(f"  Sharpe threshold: {disable_rule.get('sharpe_low', 0.0)}")
            print(f"  Confirm days: {disable_rule.get('confirm_days', 2)}")
            print(f"  Max DD %: {disable_rule.get('max_dd_pct', 2.0):.1%}")
            print(f"  Min observations: {disable_rule.get('min_observations', 21)}")
            
            # Show breach counter
            breach_days = get_consecutive_breach_days()
            print(f"\nCurrent State:")
            print(f"  Consecutive breach days: {breach_days}")
            
            if args.edit:
                try:
                    # Open config file in default editor
                    editor = os.getenv('EDITOR', 'nano')
                    subprocess.run([editor, 'config/allocation.yml'])
                except Exception as e:
                    print(f"❌ Failed to open editor: {e}")
                    print("   Please edit config/allocation.yml manually")
        else:
            print("❌ Unknown canary command")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Try: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Canary command failed: {e}")
        sys.exit(1)


def _handle_costs_command(args):
    """Handle trading cost analysis commands"""
    try:
        from mech_exo.reporting.costs import TradingCostAnalyzer
        from datetime import datetime
        import pandas as pd
        
        print(f"💰 Trading Cost Analysis: {args.start_date} to {args.end_date}")
        print("=" * 60)
        
        # Validate dates
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        except ValueError as e:
            print(f"❌ Invalid date format: {e}")
            print("   Use YYYY-MM-DD format (e.g., 2025-01-01)")
            sys.exit(1)
        
        if start_date > end_date:
            print("❌ Start date must be before end date")
            sys.exit(1)
        
        # Initialize analyzer
        analyzer = TradingCostAnalyzer()
        
        try:
            # Perform cost analysis
            print("📊 Analyzing trading costs...")
            results = analyzer.analyze_costs(start_date, end_date)
            
            if 'error' in results:
                print(f"❌ Analysis failed: {results['error']}")
                sys.exit(1)
            
            # Display summary
            summary = results.get('summary', {})
            print(f"\n📈 Summary Results:")
            print(f"  Total Trades: {summary.get('total_trades', 0):,}")
            print(f"  Total Notional: ${summary.get('total_notional', 0):,.0f}")
            print(f"  Total Costs: ${summary.get('total_costs', 0):,.2f}")
            print(f"  Average Cost: {summary.get('avg_cost_bps', 0):.1f} bps")
            print(f"  Commission: {summary.get('avg_commission_bps', 0):.1f} bps")
            print(f"  Slippage: {summary.get('avg_slippage_bps', 0):.1f} bps")
            print(f"  Spread: {summary.get('avg_spread_bps', 0):.1f} bps")
            print(f"  Average Trade Size: ${summary.get('avg_trade_size', 0):,.0f}")
            print(f"  Largest Trade: ${summary.get('largest_trade', 0):,.0f}")
            
            # Cost assessment
            avg_cost_bps = summary.get('avg_cost_bps', 0)
            if avg_cost_bps < 5:
                print(f"  Assessment: ✅ Excellent (< 5 bps)")
            elif avg_cost_bps < 10:
                print(f"  Assessment: ✅ Good (5-10 bps)")
            elif avg_cost_bps < 20:
                print(f"  Assessment: ⚠️ Fair (10-20 bps)")
            else:
                print(f"  Assessment: ❌ High (> 20 bps)")
            
            # Show verbose details if requested
            if args.verbose:
                daily_summary = results.get('daily_summary')
                if daily_summary is not None and not daily_summary.empty:
                    print(f"\n📅 Daily Breakdown:")
                    for _, row in daily_summary.iterrows():
                        print(f"  {row['date']}: {row['trades']} trades, "
                              f"${row['total_notional']:,.0f} notional, "
                              f"{row['avg_cost_bps']:.1f} bps avg cost")
                
                symbol_analysis = results.get('symbol_analysis')
                if symbol_analysis is not None and not symbol_analysis.empty:
                    print(f"\n🎯 Top Symbols by Volume:")
                    for _, row in symbol_analysis.head(10).iterrows():
                        print(f"  {row['symbol']}: {row['trades']} trades, "
                              f"${row['total_notional']:,.0f} notional, "
                              f"{row['avg_cost_bps']:.1f} bps avg cost")
            
            # Update trade_costs table if requested
            if args.update_table:
                print(f"\n💾 Updating trade_costs table...")
                success = analyzer.create_trade_costs_table(start_date, end_date)
                if success:
                    print("✅ Trade costs table updated successfully")
                else:
                    print("❌ Failed to update trade costs table")
            
            # Export HTML report if requested
            if args.html:
                print(f"\n📄 Exporting HTML report to {args.html}...")
                success = analyzer.export_html_report(results, args.html)
                if success:
                    print(f"✅ HTML report saved to {args.html}")
                    
                    # Try to open the file
                    try:
                        import webbrowser
                        from pathlib import Path
                        file_path = Path(args.html).resolve()
                        webbrowser.open(f"file://{file_path}")
                        print(f"📖 Opened report in default browser")
                    except Exception:
                        print(f"📖 Open {args.html} in your browser to view the report")
                else:
                    print(f"❌ Failed to export HTML report")
            
            # Export CSV if requested
            if args.csv:
                print(f"\n📊 Exporting detailed data to {args.csv}...")
                try:
                    detailed_costs = results.get('detailed_costs')
                    if detailed_costs:
                        df = pd.DataFrame(detailed_costs)
                        df.to_csv(args.csv, index=False)
                        print(f"✅ CSV data exported to {args.csv}")
                        print(f"   Exported {len(df)} fill records with cost details")
                    else:
                        print(f"❌ No detailed cost data available for CSV export")
                        print(f"   (May be too many records - limit is 1000)")
                except Exception as e:
                    print(f"❌ Failed to export CSV: {e}")
            
            print(f"\n📋 Analysis Period: {summary.get('period_days', 0)} days")
            print(f"📈 Average Trades/Day: {summary.get('avg_trades_per_day', 0):.1f}")
            
        finally:
            analyzer.close()
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Try: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Cost analysis failed: {e}")
        sys.exit(1)


def _handle_rollback_command(args):
    """Handle rollback commands for incident recovery"""
    try:
        from mech_exo.utils.rollback import RollbackManager, confirm_rollback, validate_timestamp
        
        print("🔄 Emergency Rollback Tool")
        print("=" * 40)
        
        # Show history if requested
        if args.history:
            print("📜 Recent Rollback History (7 days):")
            rollback_mgr = RollbackManager(dry_run=True)
            history = rollback_mgr.get_rollback_history()
            
            if history:
                for entry in history:
                    print(f"  {entry['timestamp'][:19]}: {entry['message']}")
                    print(f"    Commit: {entry['commit'][:8]}")
                    print()
            else:
                print("  No recent rollbacks found")
            return
        
        # Validate timestamp
        if not validate_timestamp(args.target_timestamp):
            print("❌ Invalid timestamp format")
            print("   Use: YYYY-MM-DDTHH:MM:SS (e.g., 2025-06-15T09:55:00)")
            sys.exit(1)
        
        # Determine rollback type
        if args.flow:
            rollback_type = f"flow:{args.flow}"
            operation_desc = f"Rollback flow '{args.flow}' to {args.target_timestamp}"
        elif args.config:
            rollback_type = f"config:{args.config}"
            operation_desc = f"Rollback config '{args.config}' to {args.target_timestamp}"
        elif args.database:
            rollback_type = "database:flags"
            operation_desc = f"Rollback database flags to {args.target_timestamp}"
        else:
            print("❌ Must specify --flow, --config, or --database")
            sys.exit(1)
        
        # Initialize rollback manager
        rollback_mgr = RollbackManager(dry_run=args.dry_run)
        
        # Confirmation check (unless forced or dry-run)
        if not args.dry_run and not args.force:
            if not confirm_rollback(operation_desc):
                print("❌ Rollback cancelled by user")
                sys.exit(1)
        
        print(f"\n🎯 Target: {args.target_timestamp}")
        print(f"🔧 Mode: {'DRY RUN' if args.dry_run else 'LIVE EXECUTION'}")
        print(f"📦 Type: {rollback_type}")
        print()
        
        # Execute rollback based on type
        if args.flow:
            print(f"🔄 Rolling back flow: {args.flow}")
            result = rollback_mgr.rollback_flow_deployment(args.flow, args.target_timestamp)
            
        elif args.config:
            print(f"🔄 Rolling back config: {args.config}")
            result = rollback_mgr.rollback_config_file(args.config, args.target_timestamp)
            
        elif args.database:
            print(f"🔄 Rolling back database flags")
            result = rollback_mgr.rollback_database_state(args.target_timestamp, flags_only=True)
        
        # Display results
        if result['success']:
            if args.dry_run:
                print("✅ DRY RUN SUCCESSFUL")
                print(f"📋 Would change {result.get('files_to_change', result.get('files_changed', 0))} files")
                
                if 'rollback_plan' in result:
                    plan = result['rollback_plan']
                    print(f"\n📝 Rollback Plan:")
                    for file_info in plan.get('files', []):
                        print(f"  {file_info['path']}: {file_info['action']}")
                
                if 'changes_preview' in result:
                    preview = result['changes_preview']
                    if preview['has_changes']:
                        print(f"\n📄 Changes Preview ({preview['diff_lines']} lines):")
                        print(preview['preview'][:500])  # First 500 chars
                        if len(preview['preview']) > 500:
                            print("... (truncated)")
                
            else:
                print("✅ ROLLBACK COMPLETED SUCCESSFULLY")
                print(f"📁 Files changed: {result.get('files_changed', 0)}")
                print(f"🎯 Target commit: {result.get('target_commit', 'N/A')[:8]}")
                
                if result.get('notification_sent'):
                    print("📱 Telegram notification sent")
                
                if result.get('backup_path'):
                    print(f"💾 Backup created: {result['backup_path']}")
            
            # Show next steps
            print(f"\n📋 Next Steps:")
            if not args.dry_run:
                print("  1. Verify system functionality")
                print("  2. Check logs for any errors")
                print("  3. Monitor alerts for issues")
                print("  4. Update team on rollback status")
            else:
                print("  1. Review the rollback plan above")
                print("  2. Run without --dry-run if plan looks correct")
                print("  3. Use --force to skip confirmation prompt")
        
        else:
            print("❌ ROLLBACK FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
            if 'available_flows' in result:
                print(f"\nAvailable flows: {', '.join(result['available_flows'])}")
            
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Try: pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Rollback interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Rollback tool failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _handle_runbook_command(args):
    """Handle runbook commands for on-call operations"""
    try:
        from mech_exo.utils.runbook import get_runbook, cli_runbook_export, cli_incident_lookup
        from mech_exo.utils.runbook import EscalationLevel
        from datetime import datetime
        
        if not args.runbook_action:
            print("❌ Runbook action required")
            print("Available actions: export, lookup, list, escalate")
            sys.exit(1)
        
        runbook = get_runbook()
        
        if args.runbook_action == "export":
            print("📖 Exporting On-Call Runbook")
            print("=" * 40)
            
            success = cli_runbook_export(args.output)
            if success:
                print(f"✅ Runbook exported to {args.output}")
                print(f"📋 Contains {len(runbook.runbook_entries)} incident procedures")
                print(f"🚨 Escalation rules: Telegram → Email → Phone")
                print(f"🌙 Quiet hours: 22:00-06:00 local")
            else:
                print("❌ Failed to export runbook")
                sys.exit(1)
        
        elif args.runbook_action == "lookup":
            print(f"🔍 Looking up incident: {args.incident_id}")
            print("=" * 40)
            
            result = cli_incident_lookup(args.incident_id)
            if result:
                print(result)
            else:
                print(f"❌ Incident '{args.incident_id}' not found")
                sys.exit(1)
        
        elif args.runbook_action == "list":
            print("📋 Available Incident Procedures")
            print("=" * 40)
            
            incidents = runbook.list_incidents()
            for i, entry in enumerate(incidents, 1):
                severity_icon = {
                    "critical": "🚨",
                    "high": "⚠️", 
                    "medium": "📋",
                    "low": "ℹ️"
                }.get(entry.severity.value, "❓")
                
                print(f"{i:2}. {severity_icon} {entry.incident_id}")
                print(f"     {entry.title}")
                print(f"     Escalation: {entry.escalation_threshold_minutes}min")
                print()
        
        elif args.runbook_action == "escalate":
            print(f"🚨 Testing Escalation System")
            print("=" * 40)
            
            # Validate incident ID
            entry = runbook.get_runbook_entry(args.incident_id)
            if not entry:
                print(f"❌ Unknown incident ID: {args.incident_id}")
                available_ids = list(runbook.runbook_entries.keys())
                print(f"Available: {', '.join(available_ids)}")
                sys.exit(1)
            
            # Convert level string to enum
            level_map = {
                "telegram": EscalationLevel.TELEGRAM,
                "email": EscalationLevel.EMAIL,
                "phone": EscalationLevel.PHONE
            }
            escalation_level = level_map[args.level]
            
            print(f"📱 Incident: {entry.title}")
            print(f"🔔 Level: {args.level.upper()}")
            print(f"📝 Details: {args.details}")
            print()
            
            # Simulate incident start time (now)
            incident_start = datetime.now()
            
            # Test escalation
            success = runbook.trigger_escalation(
                args.incident_id, 
                incident_start, 
                escalation_level, 
                args.details
            )
            
            if success:
                print("✅ Escalation sent successfully")
                
                # Show what would happen next
                future_escalations = runbook.should_escalate(incident_start, entry.severity)
                if future_escalations:
                    print(f"\n⏰ Future escalations would trigger:")
                    for level in future_escalations:
                        print(f"   • {level.value.upper()}")
            else:
                print("❌ Escalation failed")
                sys.exit(1)
        
        else:
            print(f"❌ Unknown runbook action: {args.runbook_action}")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Try: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Runbook command failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()