#!/usr/bin/env python3
"""
Auto-Reconciliation Script

Pulls T-day fills and T+1 statement, matches trades, and produces diff report.
Sends alerts if total difference exceeds threshold.
"""

import sys
import argparse
import logging
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mech_exo.reconciliation.ib_statement_parser import parse_ib_statement
from mech_exo.reconciliation.reconciler import TradeReconciler, ReconciliationStatus
from mech_exo.datasource.storage import DataStorage
from mech_exo.utils.alerts import AlertManager, Alert, AlertType, AlertLevel

logger = logging.getLogger(__name__)


class ReconciliationScript:
    """Main reconciliation script"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reconciliation script"""
        self.config = config or self._default_config()
        self.reconciler = TradeReconciler(self.config.get('reconciler'))
        self.storage = DataStorage()
        
        # Ensure database tables exist
        self._ensure_tables_exist()
        
        if self.config['alerts']['enabled']:
            self.alert_manager = AlertManager()
        else:
            self.alert_manager = None
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'reconciler': {
                'tolerances': {
                    'price_tolerance': 0.01,
                    'commission_tolerance': 0.01,
                    'net_cash_tolerance': 0.05
                },
                'thresholds': {
                    'pass_threshold_bps': 5,
                    'warning_threshold_bps': 2
                }
            },
            'alerts': {
                'enabled': True,
                'channels': ['telegram'],
                'critical_threshold_bps': 10
            },
            'data_sources': {
                'fills_table': 'fills',
                'statement_dir': 'data/statements'
            }
        }
    
    def run_reconciliation(self, 
                          trade_date: Optional[date] = None,
                          statement_file: Optional[str] = None,
                          ci_mode: bool = False,
                          write_db: bool = False) -> Dict[str, Any]:
        """
        Run trade reconciliation for specified date
        
        Args:
            trade_date: Date to reconcile (default: yesterday)
            statement_file: Path to statement file (auto-detect if None)
            ci_mode: CI mode with stub data
            write_db: Write results to database
            
        Returns:
            Reconciliation results dictionary
        """
        if trade_date is None:
            trade_date = date.today() - timedelta(days=1)
        
        logger.info(f"🔄 Starting reconciliation for {trade_date}")
        
        try:
            # Load internal fills
            internal_fills = self._load_internal_fills(trade_date, ci_mode)
            logger.info(f"Loaded {len(internal_fills)} internal fills")
            
            # Load broker statement
            broker_statement = self._load_broker_statement(trade_date, statement_file, ci_mode)
            logger.info(f"Loaded {len(broker_statement)} broker trades")
            
            # Perform reconciliation
            result = self.reconciler.reconcile(internal_fills, broker_statement, trade_date)
            
            # Generate report
            report = self._generate_report(result, trade_date)
            
            # Write to database if requested
            if write_db:
                self._write_daily_recon(result, trade_date)
                self._backfill_commission_into_fills(broker_statement, result)
            
            # Send alerts if needed
            if self.alert_manager and result.status != ReconciliationStatus.PASS:
                self._send_alerts(result, trade_date)
            
            # Return results
            return {
                'status': result.status.value,
                'trade_date': trade_date.isoformat(),
                'total_diff_bps': result.total_diff_bps,
                'matched_trades': len(result.trade_matches),
                'unmatched_internal': len(result.unmatched_internal),
                'unmatched_broker': len(result.unmatched_broker),
                'commission_diff': result.total_commission_diff,
                'net_cash_diff': result.total_net_cash_diff,
                'alerts': result.alerts,
                'report': report,
                'summary': result.summary
            }
            
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            
            # Send error alert
            if self.alert_manager:
                self._send_error_alert(str(e), trade_date)
            
            return {
                'status': 'error',
                'trade_date': trade_date.isoformat(),
                'error': str(e),
                'total_diff_bps': 0,
                'matched_trades': 0,
                'unmatched_internal': 0,
                'unmatched_broker': 0
            }
    
    def _load_internal_fills(self, trade_date: date, ci_mode: bool = False) -> pd.DataFrame:
        """Load internal fills from database"""
        if ci_mode:
            # Return stub data for CI testing
            return self._generate_stub_fills(trade_date)
        
        try:
            # Query fills table for the specified date
            query = f"""
            SELECT 
                fill_id,
                symbol,
                quantity,
                fill_price,
                commission_usd as commission,
                fill_time,
                order_id,
                strategy,
                (quantity * fill_price + commission_usd) as net_cash
            FROM {self.config['data_sources']['fills_table']}
            WHERE DATE(fill_time) = ?
            ORDER BY fill_time
            """
            
            result = self.storage.conn.execute(query, [trade_date.isoformat()]).fetchall()
            
            if not result:
                logger.warning(f"No internal fills found for {trade_date}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = ['fill_id', 'symbol', 'quantity', 'fill_price', 'commission', 
                      'fill_time', 'order_id', 'strategy', 'net_cash']
            df = pd.DataFrame(result, columns=columns)
            
            # Convert types
            df['fill_time'] = pd.to_datetime(df['fill_time'])
            df['quantity'] = pd.to_numeric(df['quantity'])
            df['fill_price'] = pd.to_numeric(df['fill_price'])
            df['commission'] = pd.to_numeric(df['commission'])
            df['net_cash'] = pd.to_numeric(df['net_cash'])
            
            # Add trade_id for matching
            df['trade_id'] = df['fill_id']
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load internal fills: {e}")
            if ci_mode:
                return self._generate_stub_fills(trade_date)
            raise
    
    def _load_broker_statement(self, 
                              trade_date: date, 
                              statement_file: Optional[str] = None,
                              ci_mode: bool = False) -> pd.DataFrame:
        """Load broker statement"""
        if ci_mode:
            # Return stub data for CI testing
            return self._generate_stub_statement(trade_date)
        
        if statement_file:
            # Use provided statement file
            statement_path = Path(statement_file)
        else:
            # Auto-detect statement file
            statement_path = self._find_statement_file(trade_date)
        
        if not statement_path or not statement_path.exists():
            logger.warning(f"No statement file found for {trade_date}")
            if ci_mode:
                return self._generate_stub_statement(trade_date)
            return pd.DataFrame()
        
        try:
            # Parse statement
            df = parse_ib_statement(str(statement_path))
            
            # Filter by trade date if needed
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[df['trade_date'].dt.date == trade_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse statement {statement_path}: {e}")
            if ci_mode:
                return self._generate_stub_statement(trade_date)
            raise
    
    def _find_statement_file(self, trade_date: date) -> Optional[Path]:
        """Find statement file for given date"""
        statement_dir = Path(self.config['data_sources']['statement_dir'])
        
        if not statement_dir.exists():
            logger.warning(f"Statement directory not found: {statement_dir}")
            return None
        
        # Try various filename patterns
        patterns = [
            f"statement_{trade_date.strftime('%Y%m%d')}.csv",
            f"statement_{trade_date.strftime('%Y-%m-%d')}.csv",
            f"ib_statement_{trade_date.strftime('%Y%m%d')}.csv",
            f"statement_{trade_date.strftime('%Y%m%d')}.ofx",
            f"statement_{trade_date.strftime('%Y-%m-%d')}.ofx"
        ]
        
        for pattern in patterns:
            statement_path = statement_dir / pattern
            if statement_path.exists():
                logger.info(f"Found statement file: {statement_path}")
                return statement_path
        
        # Try finding any statement file modified around the trade date
        for file_path in statement_dir.glob("*statement*"):
            if file_path.is_file():
                # Check modification time
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime).date()
                if abs((mtime - trade_date).days) <= 2:  # Within 2 days
                    logger.info(f"Found statement file by date: {file_path}")
                    return file_path
        
        return None
    
    def _generate_stub_fills(self, trade_date: date) -> pd.DataFrame:
        """Generate stub internal fills for testing"""
        stub_data = [
            {
                'fill_id': 'FILL001',
                'symbol': 'AAPL',
                'quantity': 100,
                'fill_price': 150.25,
                'commission': 1.00,
                'fill_time': pd.Timestamp(trade_date) + pd.Timedelta(hours=9, minutes=30),
                'order_id': 'ORD001',
                'strategy': 'test_strategy',
                'net_cash': -15026.00,
                'trade_id': 'FILL001'
            },
            {
                'fill_id': 'FILL002', 
                'symbol': 'MSFT',
                'quantity': -50,
                'fill_price': 305.80,
                'commission': 1.50,
                'fill_time': pd.Timestamp(trade_date) + pd.Timedelta(hours=10, minutes=15),
                'order_id': 'ORD002',
                'strategy': 'test_strategy',
                'net_cash': 15288.50,
                'trade_id': 'FILL002'
            }
        ]
        
        return pd.DataFrame(stub_data)
    
    def _generate_stub_statement(self, trade_date: date) -> pd.DataFrame:
        """Generate stub broker statement for testing"""
        stub_data = [
            {
                'symbol': 'AAPL',
                'qty': 100,
                'price': 150.25,
                'commission': 1.00,
                'net_cash': -15026.00,
                'trade_date': pd.Timestamp(trade_date) + pd.Timedelta(hours=9, minutes=30),
                'trade_id': 'TXN001',
                'currency': 'USD'
            },
            {
                'symbol': 'MSFT',
                'qty': -50,
                'price': 305.80,
                'commission': 1.50,
                'net_cash': 15288.50,
                'trade_date': pd.Timestamp(trade_date) + pd.Timedelta(hours=10, minutes=15),
                'trade_id': 'TXN002',
                'currency': 'USD'
            }
        ]
        
        return pd.DataFrame(stub_data)
    
    def _generate_report(self, result, trade_date: date) -> Dict[str, Any]:
        """Generate reconciliation report"""
        status_icon = {
            ReconciliationStatus.PASS: "✅",
            ReconciliationStatus.WARNING: "⚠️", 
            ReconciliationStatus.FAIL: "❌"
        }
        
        report = {
            'title': f"Trade Reconciliation Report - {trade_date}",
            'status': f"{status_icon.get(result.status, '❓')} {result.status.value.upper()}",
            'summary': {
                'Total Difference': f"{result.total_diff_bps:.1f} basis points",
                'Commission Difference': f"${result.total_commission_diff:.2f}",
                'Net Cash Difference': f"${result.total_net_cash_diff:.2f}",
                'Matched Trades': len(result.trade_matches),
                'Unmatched Internal': len(result.unmatched_internal),
                'Unmatched Broker': len(result.unmatched_broker)
            },
            'details': {
                'matched_trades': [
                    {
                        'internal_symbol': match.internal_trade.get('symbol') if match.internal_trade else None,
                        'broker_symbol': match.broker_trade.get('symbol') if match.broker_trade else None,
                        'match_type': match.match_type.value,
                        'differences': match.differences
                    }
                    for match in result.trade_matches
                ],
                'unmatched_internal': result.unmatched_internal,
                'unmatched_broker': result.unmatched_broker
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _send_alerts(self, result, trade_date: date):
        """Send reconciliation alerts"""
        if not self.alert_manager:
            return
        
        # Determine alert level
        if result.total_diff_bps > self.config['alerts']['critical_threshold_bps']:
            alert_level = AlertLevel.CRITICAL
        elif result.status == ReconciliationStatus.FAIL:
            alert_level = AlertLevel.WARNING
        else:
            alert_level = AlertLevel.INFO
        
        # Create alert message
        status_emoji = {
            ReconciliationStatus.PASS: "✅",
            ReconciliationStatus.WARNING: "⚠️",
            ReconciliationStatus.FAIL: "❌"
        }
        
        emoji = status_emoji.get(result.status, "❓")
        
        message = f"""💰 **Trade Reconciliation Alert**
        
**Date:** {trade_date}
**Status:** {emoji} {result.status.value.upper()}
**Total Difference:** {result.total_diff_bps:.1f} basis points

**Summary:**
• Matched Trades: {len(result.trade_matches)}
• Unmatched Internal: {len(result.unmatched_internal)}
• Unmatched Broker: {len(result.unmatched_broker)}
• Commission Diff: ${result.total_commission_diff:.2f}
• Net Cash Diff: ${result.total_net_cash_diff:.2f}

**Issues:**"""
        
        for alert in result.alerts:
            message += f"\n• {alert}"
        
        # Send alert
        alert = Alert(
            alert_type=AlertType.SYSTEM_ALERT,
            level=alert_level,
            title=f"Reconciliation {result.status.value.title()} - {trade_date}",
            message=message,
            timestamp=datetime.now(),
            data={
                'trade_date': trade_date.isoformat(),
                'status': result.status.value,
                'total_diff_bps': result.total_diff_bps,
                'summary': result.summary
            }
        )
        
        success = self.alert_manager.send_alert_with_escalation(
            alert,
            channels=self.config['alerts']['channels'],
            respect_quiet_hours=False,
            force_send=True
        )
        
        if success:
            logger.info(f"Reconciliation alert sent for {trade_date}")
        else:
            logger.warning(f"Failed to send reconciliation alert for {trade_date}")
    
    def _send_error_alert(self, error_msg: str, trade_date: date):
        """Send error alert"""
        if not self.alert_manager:
            return
        
        alert = Alert(
            alert_type=AlertType.SYSTEM_ALERT,
            level=AlertLevel.CRITICAL,
            title=f"Reconciliation Error - {trade_date}",
            message=f"💥 **Reconciliation Failed**\n\n"
                   f"**Date:** {trade_date}\n"
                   f"**Error:** {error_msg}\n\n"
                   f"Please check the reconciliation system and retry.",
            timestamp=datetime.now(),
            data={
                'trade_date': trade_date.isoformat(),
                'error': error_msg
            }
        )
        
        self.alert_manager.send_alert_with_escalation(
            alert,
            channels=self.config['alerts']['channels'],
            respect_quiet_hours=False,
            force_send=True
        )
    
    def _write_daily_recon(self, result, trade_date: date):
        """Write reconciliation summary to daily_recon table"""
        try:
            import json
            
            # Prepare summary data
            summary_data = {
                'reconciliation_summary': result.summary,
                'trade_matches': len(result.trade_matches),
                'unmatched_details': {
                    'internal_count': len(result.unmatched_internal),
                    'broker_count': len(result.unmatched_broker),
                    'internal_symbols': [t.get('symbol') for t in result.unmatched_internal],
                    'broker_symbols': [t.get('symbol') for t in result.unmatched_broker]
                },
                'alerts': result.alerts
            }
            
            # UPSERT into daily_recon table
            query = """
            INSERT OR REPLACE INTO daily_recon (
                recon_date, internal_trades, broker_trades, matched_trades,
                unmatched_internal, unmatched_broker, total_diff_bps,
                commission_diff_usd, net_cash_diff_usd, status,
                alerts_sent, summary_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.storage.conn.execute(query, [
                trade_date.isoformat(),
                result.summary.get('internal_trades', 0),
                result.summary.get('broker_trades', 0),
                len(result.trade_matches),
                len(result.unmatched_internal),
                len(result.unmatched_broker),
                result.total_diff_bps,
                result.total_commission_diff,
                result.total_net_cash_diff,
                result.status.value,
                bool(self.alert_manager and result.status != ReconciliationStatus.PASS),
                json.dumps(summary_data),
                datetime.now().isoformat()
            ])
            
            self.storage.conn.commit()
            logger.info(f"✅ Wrote reconciliation summary to daily_recon for {trade_date}")
            
            # Write detailed audit records
            self._write_reconciliation_audit(result, trade_date)
            
        except Exception as e:
            logger.error(f"Failed to write daily_recon for {trade_date}: {e}")
            raise
    
    def _write_reconciliation_audit(self, result, trade_date: date):
        """Write detailed reconciliation audit records"""
        try:
            import json
            
            audit_records = []
            
            # Write matched trades
            for match in result.trade_matches:
                internal = match.internal_trade
                broker = match.broker_trade
                
                record = [
                    trade_date.isoformat(),
                    internal.get('fill_id') if internal else None,
                    broker.get('trade_id') if broker else None,
                    match.match_type.value,
                    match.match_score,
                    (internal or broker).get('symbol'),
                    (internal or broker).get('qty') or (internal or broker).get('quantity'),
                    internal.get('price') if internal else None,
                    broker.get('price') if broker else None,
                    (internal.get('price', 0) - broker.get('price', 0)) if (internal and broker) else None,
                    internal.get('commission') if internal else None,
                    broker.get('commission') if broker else None,
                    (internal.get('commission', 0) - broker.get('commission', 0)) if (internal and broker) else None,
                    internal.get('net_cash') if internal else None,
                    broker.get('net_cash') if broker else None,
                    (internal.get('net_cash', 0) - broker.get('net_cash', 0)) if (internal and broker) else None,
                    json.dumps(match.differences),
                    datetime.now().isoformat()
                ]
                audit_records.append(record)
            
            # Write unmatched internal trades
            for trade in result.unmatched_internal:
                record = [
                    trade_date.isoformat(),
                    trade.get('fill_id'),
                    None,  # No broker trade_id
                    'no_match',
                    0.0,
                    trade.get('symbol'),
                    trade.get('qty') or trade.get('quantity'),
                    trade.get('price'),
                    None,  # No broker price
                    None,  # No price diff
                    trade.get('commission'),
                    None,  # No broker commission
                    None,  # No commission diff
                    trade.get('net_cash'),
                    None,  # No broker net_cash
                    None,  # No net_cash diff
                    json.dumps({'type': 'unmatched_internal'}),
                    datetime.now().isoformat()
                ]
                audit_records.append(record)
            
            # Write unmatched broker trades
            for trade in result.unmatched_broker:
                record = [
                    trade_date.isoformat(),
                    None,  # No fill_id
                    trade.get('trade_id'),
                    'no_match',
                    0.0,
                    trade.get('symbol'),
                    trade.get('qty'),
                    None,  # No internal price
                    trade.get('price'),
                    None,  # No price diff
                    None,  # No internal commission
                    trade.get('commission'),
                    None,  # No commission diff
                    None,  # No internal net_cash
                    trade.get('net_cash'),
                    None,  # No net_cash diff
                    json.dumps({'type': 'unmatched_broker'}),
                    datetime.now().isoformat()
                ]
                audit_records.append(record)
            
            # Batch insert audit records
            if audit_records:
                audit_query = """
                INSERT INTO reconciliation_audit (
                    recon_date, fill_id, broker_trade_id, match_type, match_score,
                    symbol, quantity, price_internal, price_broker, price_diff,
                    commission_internal, commission_broker, commission_diff,
                    net_cash_internal, net_cash_broker, net_cash_diff,
                    differences_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                self.storage.conn.executemany(audit_query, audit_records)
                self.storage.conn.commit()
                
                logger.info(f"✅ Wrote {len(audit_records)} audit records for {trade_date}")
            
        except Exception as e:
            logger.error(f"Failed to write reconciliation audit for {trade_date}: {e}")
            # Don't raise - audit is supplementary
    
    def _backfill_commission_into_fills(self, broker_statement: pd.DataFrame, result):
        """Back-fill real commission data into fills table"""
        try:
            if broker_statement.empty:
                logger.warning("No broker statement data for commission back-fill")
                return
            
            backfilled_count = 0
            
            # Process each matched trade
            for match in result.trade_matches:
                if not match.internal_trade or not match.broker_trade:
                    continue
                
                fill_id = match.internal_trade.get('fill_id')
                broker_commission = match.broker_trade.get('commission', 0)
                
                if not fill_id:
                    continue
                
                # Update fills table with real commission
                update_query = """
                UPDATE fills 
                SET 
                    original_commission_usd = commission_usd,
                    commission_usd = ?,
                    commission_source = 'broker',
                    last_reconciled_at = ?
                WHERE fill_id = ?
                """
                
                self.storage.conn.execute(update_query, [
                    broker_commission,
                    datetime.now().isoformat(),
                    fill_id
                ])
                
                backfilled_count += 1
            
            self.storage.conn.commit()
            
            if backfilled_count > 0:
                logger.info(f"✅ Back-filled commission for {backfilled_count} fills")
            else:
                logger.info("No commission back-fill performed (no matched trades)")
                
        except Exception as e:
            logger.error(f"Failed to back-fill commission data: {e}")
            # Don't raise - back-fill is supplementary
    
    def _ensure_tables_exist(self):
        """Ensure reconciliation tables exist"""
        try:
            # Read migrations file
            migrations_path = Path(__file__).parent.parent / "storage" / "migrations.sql"
            
            if migrations_path.exists():
                with open(migrations_path, 'r') as f:
                    migration_sql = f.read()
                
                # Execute migrations (split by semicolon)
                for statement in migration_sql.split(';'):
                    statement = statement.strip()
                    if statement and not statement.startswith('--'):
                        self.storage.conn.execute(statement)
                
                self.storage.conn.commit()
                logger.info("✅ Database migrations applied")
            else:
                logger.warning(f"Migrations file not found: {migrations_path}")
                
        except Exception as e:
            logger.error(f"Failed to apply migrations: {e}")
            # Don't raise - let reconciliation continue


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Trade Reconciliation Script")
    
    parser.add_argument('--date', type=str, help='Trade date (YYYY-MM-DD), default: yesterday')
    parser.add_argument('--statement', type=str, help='Path to statement file')
    parser.add_argument('--ci', action='store_true', help='CI mode with stub data')
    parser.add_argument('--write-db', action='store_true', help='Write results to database')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse trade date
    if args.date:
        try:
            trade_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        trade_date = date.today() - timedelta(days=1)
    
    # Load config if provided
    config = None
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    try:
        # Run reconciliation
        script = ReconciliationScript(config)
        result = script.run_reconciliation(
            trade_date=trade_date,
            statement_file=args.statement,
            ci_mode=args.ci,
            write_db=args.write_db
        )
        
        # Print results
        print(f"\n🔄 Reconciliation Results - {trade_date}")
        print(f"Status: {result['status'].upper()}")
        print(f"Total Difference: {result['total_diff_bps']:.1f} basis points")
        print(f"Matched Trades: {result['matched_trades']}")
        print(f"Unmatched Internal: {result['unmatched_internal']}")
        print(f"Unmatched Broker: {result['unmatched_broker']}")
        
        if result['alerts']:
            print(f"\nAlerts:")
            for alert in result['alerts']:
                print(f"  • {alert}")
        
        # Exit code based on status
        if result['status'] == 'pass':
            print("\n✅ Reconciliation PASSED")
            sys.exit(0)
        elif result['status'] == 'warning':
            print("\n⚠️ Reconciliation WARNING")
            sys.exit(0)  # Don't fail CI for warnings
        else:
            print("\n❌ Reconciliation FAILED")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Reconciliation script failed: {e}")
        print(f"\n💥 Reconciliation ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()