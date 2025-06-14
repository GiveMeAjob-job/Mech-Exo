-- Database Migrations for Reconciliation System
-- Run these migrations to set up reconciliation tables

-- Migration 001: Create daily_recon table
CREATE TABLE IF NOT EXISTS daily_recon (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recon_date DATE NOT NULL UNIQUE,
    internal_trades INTEGER NOT NULL DEFAULT 0,
    broker_trades INTEGER NOT NULL DEFAULT 0,
    matched_trades INTEGER NOT NULL DEFAULT 0,
    unmatched_internal INTEGER NOT NULL DEFAULT 0,
    unmatched_broker INTEGER NOT NULL DEFAULT 0,
    total_diff_bps REAL NOT NULL DEFAULT 0.0,
    commission_diff_usd REAL NOT NULL DEFAULT 0.0,
    net_cash_diff_usd REAL NOT NULL DEFAULT 0.0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    pdf_path VARCHAR(500),
    s3_url VARCHAR(500),
    alerts_sent BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary_json TEXT,
    
    -- Indexes for performance
    INDEX idx_daily_recon_date (recon_date),
    INDEX idx_daily_recon_status (status),
    INDEX idx_daily_recon_created (created_at)
);

-- Migration 002: Add commission_source column to fills table
-- This tracks whether commission came from broker statement or estimation
ALTER TABLE fills ADD COLUMN commission_source VARCHAR(20) DEFAULT 'estimate';
ALTER TABLE fills ADD COLUMN original_commission_usd REAL;
ALTER TABLE fills ADD COLUMN last_reconciled_at TIMESTAMP;

-- Create index on commission source for analytics
CREATE INDEX IF NOT EXISTS idx_fills_commission_source ON fills(commission_source);

-- Migration 003: Create reconciliation_audit table for detailed matching
CREATE TABLE IF NOT EXISTS reconciliation_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recon_date DATE NOT NULL,
    fill_id VARCHAR(100),
    broker_trade_id VARCHAR(100),
    match_type VARCHAR(20) NOT NULL, -- 'exact_id', 'fuzzy_match', 'no_match'
    match_score REAL DEFAULT 0.0,
    symbol VARCHAR(10),
    quantity REAL,
    price_internal REAL,
    price_broker REAL,
    price_diff REAL,
    commission_internal REAL,
    commission_broker REAL,
    commission_diff REAL,
    net_cash_internal REAL,
    net_cash_broker REAL,
    net_cash_diff REAL,
    differences_json TEXT, -- JSON blob of all differences
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraints
    FOREIGN KEY (recon_date) REFERENCES daily_recon(recon_date),
    
    -- Indexes
    INDEX idx_reconciliation_audit_date (recon_date),
    INDEX idx_reconciliation_audit_fill (fill_id),
    INDEX idx_reconciliation_audit_broker (broker_trade_id),
    INDEX idx_reconciliation_audit_match_type (match_type)
);

-- Migration 004: Create reconciliation_config table
CREATE TABLE IF NOT EXISTS reconciliation_config (
    id INTEGER PRIMARY KEY,
    price_tolerance REAL NOT NULL DEFAULT 0.01,
    commission_tolerance REAL NOT NULL DEFAULT 0.01,
    net_cash_tolerance REAL NOT NULL DEFAULT 0.05,
    pass_threshold_bps REAL NOT NULL DEFAULT 5.0,
    warning_threshold_bps REAL NOT NULL DEFAULT 2.0,
    alerts_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    alert_channels TEXT NOT NULL DEFAULT 'telegram',
    s3_bucket VARCHAR(100) DEFAULT 'mechexo-audit',
    s3_prefix VARCHAR(100) DEFAULT 'reconciliation',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100) DEFAULT 'system'
);

-- Insert default configuration
INSERT OR IGNORE INTO reconciliation_config (id) VALUES (1);

-- Migration 005: Create view for reconciliation dashboard
CREATE VIEW IF NOT EXISTS reconciliation_summary AS
SELECT 
    recon_date,
    status,
    total_diff_bps,
    matched_trades,
    unmatched_internal + unmatched_broker as total_unmatched,
    commission_diff_usd,
    net_cash_diff_usd,
    CASE 
        WHEN status = 'pass' THEN '✅'
        WHEN status = 'warning' THEN '⚠️'
        WHEN status = 'fail' THEN '❌'
        ELSE '❓'
    END as status_icon,
    CASE
        WHEN total_diff_bps <= 2.0 THEN 'success'
        WHEN total_diff_bps <= 5.0 THEN 'warning'
        ELSE 'danger'
    END as status_color,
    created_at,
    pdf_path IS NOT NULL as has_pdf,
    s3_url IS NOT NULL as has_s3_backup
FROM daily_recon
ORDER BY recon_date DESC;

-- Migration 006: Monthly Drawdown Guard Database Schema
-- Day 3 Module 7: Tables and views for monthly stop-loss protection

-- Create monthly_metrics table for storing daily NAV snapshots
CREATE TABLE IF NOT EXISTS monthly_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_date DATE NOT NULL UNIQUE,
    nav_value REAL NOT NULL,
    pnl_amount REAL DEFAULT 0.0,
    pnl_pct REAL DEFAULT 0.0,
    position_count INTEGER DEFAULT 0,
    gross_exposure REAL DEFAULT 0.0,
    net_exposure REAL DEFAULT 0.0,
    data_source VARCHAR(20) NOT NULL DEFAULT 'calculated', -- 'live', 'daily_metrics', 'calculated'
    is_month_start BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for performance
    INDEX idx_monthly_metrics_date (metric_date),
    INDEX idx_monthly_metrics_month_start (is_month_start),
    INDEX idx_monthly_metrics_source (data_source)
);

-- Create monthly_guard_log table for tracking guard executions
CREATE TABLE IF NOT EXISTS monthly_guard_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guard_date DATE NOT NULL,
    mtd_pct REAL NOT NULL,
    threshold_pct REAL NOT NULL DEFAULT -3.0,
    threshold_breached BOOLEAN NOT NULL DEFAULT FALSE,
    killswitch_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    alert_sent BOOLEAN NOT NULL DEFAULT FALSE,
    should_run BOOLEAN NOT NULL DEFAULT TRUE,
    calculation_successful BOOLEAN NOT NULL DEFAULT TRUE,
    action_taken VARCHAR(50) DEFAULT 'none', -- 'none', 'killswitch_triggered', 'alert_sent'
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    month_start_nav REAL,
    current_nav REAL,
    mtd_amount REAL,
    execution_time_ms INTEGER,
    error_message TEXT,
    flow_run_id VARCHAR(100),
    prefect_state VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_monthly_guard_date (guard_date),
    INDEX idx_monthly_guard_breach (threshold_breached),
    INDEX idx_monthly_guard_killswitch (killswitch_triggered),
    INDEX idx_monthly_guard_flow (flow_run_id)
);

-- Create monthly_config table for configuration management
CREATE TABLE IF NOT EXISTS monthly_config (
    id INTEGER PRIMARY KEY,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    threshold_pct REAL NOT NULL DEFAULT -3.0,
    min_history_days INTEGER NOT NULL DEFAULT 10,
    alert_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    schedule_cron VARCHAR(50) DEFAULT '10 23 * * *', -- 23:10 UTC daily
    max_retry_attempts INTEGER DEFAULT 3,
    retry_delay_minutes INTEGER DEFAULT 5,
    config_version VARCHAR(20) DEFAULT '1.0',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100) DEFAULT 'system'
);

-- Insert default monthly guard configuration
INSERT OR IGNORE INTO monthly_config (id) VALUES (1);

-- Create view for monthly guard dashboard
CREATE VIEW IF NOT EXISTS monthly_guard_summary AS
SELECT 
    guard_date,
    mtd_pct,
    threshold_pct,
    threshold_breached,
    killswitch_triggered,
    alert_sent,
    action_taken,
    CASE 
        WHEN threshold_breached THEN '🛑'
        WHEN mtd_pct <= -2.0 THEN '⚠️'
        WHEN mtd_pct > 0 THEN '✅'
        ELSE '📊'
    END as status_icon,
    CASE
        WHEN threshold_breached THEN 'danger'
        WHEN mtd_pct <= -2.0 THEN 'warning'
        WHEN mtd_pct > 0 THEN 'success'
        ELSE 'info'
    END as status_color,
    CASE
        WHEN threshold_breached THEN 'STOP-LOSS'
        WHEN mtd_pct <= -2.0 THEN 'WARNING'
        WHEN mtd_pct > 0 THEN 'POSITIVE'
        ELSE 'NORMAL'
    END as status_text,
    mtd_amount,
    month_start_nav,
    current_nav,
    execution_time_ms,
    calculation_successful,
    dry_run,
    created_at
FROM monthly_guard_log
ORDER BY guard_date DESC;

-- Create view for monthly performance tracking
CREATE VIEW IF NOT EXISTS monthly_performance AS
SELECT 
    strftime('%Y-%m', metric_date) as month_year,
    MIN(CASE WHEN is_month_start THEN nav_value END) as month_start_nav,
    MAX(nav_value) as month_max_nav,
    MIN(nav_value) as month_min_nav,
    (SELECT nav_value FROM monthly_metrics m2 
     WHERE strftime('%Y-%m', m2.metric_date) = strftime('%Y-%m', mm.metric_date)
     ORDER BY m2.metric_date DESC LIMIT 1) as month_end_nav,
    COUNT(*) as trading_days,
    AVG(pnl_pct) as avg_daily_pnl_pct,
    SUM(pnl_amount) as cumulative_pnl,
    MAX(ABS(pnl_pct)) as max_daily_move,
    CASE 
        WHEN MIN(CASE WHEN is_month_start THEN nav_value END) IS NOT NULL
        THEN ((SELECT nav_value FROM monthly_metrics m2 
               WHERE strftime('%Y-%m', m2.metric_date) = strftime('%Y-%m', mm.metric_date)
               ORDER BY m2.metric_date DESC LIMIT 1) - 
              MIN(CASE WHEN is_month_start THEN nav_value END)) / 
              MIN(CASE WHEN is_month_start THEN nav_value END) * 100
        ELSE 0
    END as month_to_date_pct
FROM monthly_metrics mm
GROUP BY strftime('%Y-%m', metric_date)
ORDER BY month_year DESC;

-- Create index on daily_metrics for monthly guard performance
CREATE INDEX IF NOT EXISTS idx_daily_metrics_nav_date ON daily_metrics(date, nav);

-- Add monthly guard tracking columns to existing daily_metrics table if not exists
-- These columns help track monthly calculations within the existing daily metrics
ALTER TABLE daily_metrics ADD COLUMN month_start_nav REAL;
ALTER TABLE daily_metrics ADD COLUMN mtd_pnl_pct REAL;
ALTER TABLE daily_metrics ADD COLUMN monthly_guard_checked BOOLEAN DEFAULT FALSE;

-- Migration 007: Rollback Drill Database Schema
-- Day 4 Module 6: Tables for drill execution tracking and reporting

-- Create drill_log table for tracking drill executions
CREATE TABLE IF NOT EXISTS drill_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drill_date DATE NOT NULL,
    drill_timestamp TIMESTAMP NOT NULL,
    duration_seconds REAL NOT NULL,
    passed BOOLEAN NOT NULL,
    file_path TEXT,
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    wait_seconds INTEGER DEFAULT 120,
    return_code INTEGER,
    error_message TEXT,
    stdout_snippet TEXT,
    stderr_snippet TEXT,
    flow_run_id TEXT,
    prefect_state TEXT DEFAULT 'completed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for performance
    INDEX idx_drill_log_date (drill_date),
    INDEX idx_drill_log_passed (passed),
    INDEX idx_drill_log_dry_run (dry_run),
    INDEX idx_drill_log_timestamp (drill_timestamp)
);

-- Create view for drill dashboard
CREATE VIEW IF NOT EXISTS drill_summary AS
SELECT 
    drill_date,
    drill_timestamp,
    passed,
    dry_run,
    duration_seconds,
    CASE 
        WHEN passed AND NOT dry_run THEN '✅'
        WHEN passed AND dry_run THEN '🧪'
        WHEN NOT passed THEN '❌'
        ELSE '❓'
    END as status_icon,
    CASE
        WHEN passed AND NOT dry_run THEN 'success'
        WHEN passed AND dry_run THEN 'info'
        WHEN NOT passed THEN 'danger'
        ELSE 'secondary'
    END as status_color,
    CASE
        WHEN passed AND NOT dry_run THEN 'PASSED'
        WHEN passed AND dry_run THEN 'DRY-RUN'
        WHEN NOT passed THEN 'FAILED'
        ELSE 'UNKNOWN'
    END as status_text,
    error_message,
    file_path,
    created_at,
    julianday('now') - julianday(drill_date) as days_ago
FROM drill_log
ORDER BY drill_timestamp DESC;