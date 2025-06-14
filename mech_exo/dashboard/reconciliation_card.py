"""
Reconciliation Status Dashboard Card

Provides real-time reconciliation status for trading dashboard.
Shows recent reconciliation results, alerts, and trend analysis.
"""

import logging
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
import json

from ..datasource.storage import DataStorage

logger = logging.getLogger(__name__)


class ReconciliationStatusCard:
    """Dashboard card for reconciliation status and trends"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize reconciliation status card
        
        Args:
            db_path: Path to database file
        """
        self.storage = DataStorage(db_path)
    
    def get_status_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get reconciliation status summary for recent days
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dict with status summary and metrics
        """
        logger.info(f"Getting reconciliation status for last {days_back} days")
        
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            # Get recent reconciliation data
            query = """
            SELECT 
                recon_date,
                status,
                total_diff_bps,
                matched_trades,
                unmatched_internal,
                unmatched_broker,
                commission_diff_usd,
                net_cash_diff_usd,
                alerts_sent,
                pdf_path IS NOT NULL as has_pdf,
                s3_url IS NOT NULL as has_s3_backup,
                created_at
            FROM daily_recon 
            WHERE recon_date >= ? AND recon_date <= ?
            ORDER BY recon_date DESC
            """
            
            results = self.storage.conn.execute(query, [
                start_date.isoformat(),
                end_date.isoformat()
            ]).fetchall()
            
            if not results:
                return self._empty_status_summary(start_date, end_date)
            
            # Convert to DataFrame for analysis
            columns = [
                'recon_date', 'status', 'total_diff_bps', 'matched_trades',
                'unmatched_internal', 'unmatched_broker', 'commission_diff_usd',
                'net_cash_diff_usd', 'alerts_sent', 'has_pdf', 'has_s3_backup', 'created_at'
            ]
            df = pd.DataFrame(results, columns=columns)
            df['recon_date'] = pd.to_datetime(df['recon_date'])
            
            # Calculate summary metrics
            summary = self._calculate_summary_metrics(df, start_date, end_date)
            
            # Get recent status
            recent_status = self._get_recent_status(df)
            
            # Get trend analysis
            trend_analysis = self._get_trend_analysis(df)
            
            # Get alerts and issues
            alerts = self._get_alerts_and_issues(df)
            
            return {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days_back': days_back
                },
                'summary': summary,
                'recent_status': recent_status,
                'trend_analysis': trend_analysis,
                'alerts': alerts,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get reconciliation status: {e}")
            return {'error': str(e), 'last_updated': datetime.now().isoformat()}
    
    def _empty_status_summary(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Return empty status summary"""
        return {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days_back': (end_date - start_date).days
            },
            'summary': {
                'total_reconciliations': 0,
                'pass_rate': 0.0,
                'avg_diff_bps': 0.0,
                'total_matched_trades': 0,
                'pdf_completion_rate': 0.0
            },
            'recent_status': {
                'latest_date': None,
                'latest_status': 'unknown',
                'latest_diff_bps': 0.0,
                'days_since_last': None
            },
            'trend_analysis': {
                'status_trend': 'stable',
                'diff_trend': 'stable',
                'issue_count': 0
            },
            'alerts': [],
            'last_updated': datetime.now().isoformat()
        }
    
    def _calculate_summary_metrics(self, df: pd.DataFrame, start_date: date, end_date: date) -> Dict[str, Any]:
        """Calculate summary metrics from reconciliation data"""
        total_recons = len(df)
        pass_count = len(df[df['status'] == 'pass'])
        pass_rate = (pass_count / total_recons * 100) if total_recons > 0 else 0.0
        
        avg_diff_bps = df['total_diff_bps'].mean() if total_recons > 0 else 0.0
        total_matched = df['matched_trades'].sum() if total_recons > 0 else 0
        
        pdf_count = df['has_pdf'].sum() if total_recons > 0 else 0
        pdf_completion_rate = (pdf_count / total_recons * 100) if total_recons > 0 else 0.0
        
        return {
            'total_reconciliations': total_recons,
            'pass_rate': round(pass_rate, 1),
            'avg_diff_bps': round(avg_diff_bps, 2),
            'total_matched_trades': int(total_matched),
            'pdf_completion_rate': round(pdf_completion_rate, 1),
            'unmatched_trades_total': int(df['unmatched_internal'].sum() + df['unmatched_broker'].sum()),
            'commission_diff_total': round(df['commission_diff_usd'].sum(), 2),
            'alerts_sent_count': int(df['alerts_sent'].sum())
        }
    
    def _get_recent_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get most recent reconciliation status"""
        if df.empty:
            return {
                'latest_date': None,
                'latest_status': 'unknown',
                'latest_diff_bps': 0.0,
                'days_since_last': None
            }
        
        latest = df.iloc[0]  # Already sorted by date DESC
        latest_date = latest['recon_date'].date()
        days_since = (date.today() - latest_date).days
        
        # Determine status color/icon
        status_info = self._get_status_display_info(latest['status'], latest['total_diff_bps'])
        
        return {
            'latest_date': latest_date.isoformat(),
            'latest_status': latest['status'],
            'latest_diff_bps': round(latest['total_diff_bps'], 2),
            'days_since_last': days_since,
            'matched_trades': int(latest['matched_trades']),
            'unmatched_trades': int(latest['unmatched_internal'] + latest['unmatched_broker']),
            'has_pdf': bool(latest['has_pdf']),
            'has_s3_backup': bool(latest['has_s3_backup']),
            'status_icon': status_info['icon'],
            'status_color': status_info['color'],
            'status_message': status_info['message']
        }
    
    def _get_status_display_info(self, status: str, diff_bps: float) -> Dict[str, str]:
        """Get display information for status"""
        if status == 'pass':
            return {
                'icon': '✅',
                'color': 'success',
                'message': f'Reconciliation passed ({diff_bps:.1f}bp)'
            }
        elif status == 'warning':
            return {
                'icon': '⚠️',
                'color': 'warning',
                'message': f'Reconciliation warning ({diff_bps:.1f}bp)'
            }
        elif status == 'fail':
            return {
                'icon': '❌',
                'color': 'danger',
                'message': f'Reconciliation failed ({diff_bps:.1f}bp)'
            }
        else:
            return {
                'icon': '❓',
                'color': 'secondary',
                'message': 'Status unknown'
            }
    
    def _get_trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in reconciliation data"""
        if len(df) < 2:
            return {
                'status_trend': 'stable',
                'diff_trend': 'stable',
                'issue_count': 0,
                'trend_message': 'Insufficient data for trend analysis'
            }
        
        # Status trend
        recent_statuses = df['status'].head(3).tolist()
        pass_count = recent_statuses.count('pass')
        fail_count = recent_statuses.count('fail')
        
        if pass_count >= 2:
            status_trend = 'improving'
        elif fail_count >= 2:
            status_trend = 'deteriorating'
        else:
            status_trend = 'stable'
        
        # Difference trend
        recent_diffs = df['total_diff_bps'].head(3)
        if len(recent_diffs) >= 2:
            if recent_diffs.iloc[0] < recent_diffs.iloc[1]:
                diff_trend = 'improving'
            elif recent_diffs.iloc[0] > recent_diffs.iloc[1]:
                diff_trend = 'worsening'
            else:
                diff_trend = 'stable'
        else:
            diff_trend = 'stable'
        
        # Issue count
        issue_count = len(df[(df['status'] == 'fail') | (df['unmatched_internal'] > 0) | (df['unmatched_broker'] > 0)])
        
        # Generate trend message
        trend_message = self._generate_trend_message(status_trend, diff_trend, issue_count, len(df))
        
        return {
            'status_trend': status_trend,
            'diff_trend': diff_trend,
            'issue_count': issue_count,
            'trend_message': trend_message,
            'avg_diff_last_3': round(recent_diffs.head(3).mean(), 2) if len(recent_diffs) >= 3 else 0.0
        }
    
    def _generate_trend_message(self, status_trend: str, diff_trend: str, issue_count: int, total_days: int) -> str:
        """Generate human-readable trend message"""
        if status_trend == 'improving' and diff_trend == 'improving':
            return "🟢 Reconciliation quality is improving"
        elif status_trend == 'deteriorating' or diff_trend == 'worsening':
            return "🔴 Reconciliation quality needs attention"
        elif issue_count == 0:
            return "🟢 All reconciliations clean"
        elif issue_count / total_days > 0.5:
            return "🟡 Frequent reconciliation issues detected"
        else:
            return "🟡 Reconciliation performance is stable"
    
    def _get_alerts_and_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get active alerts and issues"""
        alerts = []
        
        # Check for recent failures
        recent_failures = df[(df['status'] == 'fail') & (df['recon_date'] >= datetime.now().date() - timedelta(days=3))]
        for _, failure in recent_failures.iterrows():
            alerts.append({
                'type': 'error',
                'date': failure['recon_date'].date().isoformat(),
                'message': f"Reconciliation failed on {failure['recon_date'].date()} ({failure['total_diff_bps']:.1f}bp)",
                'severity': 'high'
            })
        
        # Check for missing PDFs
        missing_pdfs = df[~df['has_pdf']]
        if len(missing_pdfs) > 0:
            alerts.append({
                'type': 'warning',
                'date': datetime.now().date().isoformat(),
                'message': f"{len(missing_pdfs)} reconciliation(s) missing PDF reports",
                'severity': 'medium'
            })
        
        # Check for high differences
        high_diffs = df[df['total_diff_bps'] > 10]
        if len(high_diffs) > 0:
            alerts.append({
                'type': 'warning',
                'date': datetime.now().date().isoformat(),
                'message': f"{len(high_diffs)} reconciliation(s) with >10bp differences",
                'severity': 'medium'
            })
        
        # Check for stale data
        if not df.empty:
            latest_date = df['recon_date'].max().date()
            days_stale = (date.today() - latest_date).days
            if days_stale > 1:
                alerts.append({
                    'type': 'warning',
                    'date': datetime.now().date().isoformat(),
                    'message': f"Latest reconciliation is {days_stale} days old",
                    'severity': 'medium'
                })
        
        return alerts
    
    def get_daily_trend_data(self, days_back: int = 30) -> Dict[str, Any]:
        """Get daily trend data for charting"""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            query = """
            SELECT 
                recon_date,
                status,
                total_diff_bps,
                matched_trades,
                unmatched_internal + unmatched_broker as unmatched_trades
            FROM daily_recon 
            WHERE recon_date >= ? AND recon_date <= ?
            ORDER BY recon_date ASC
            """
            
            results = self.storage.conn.execute(query, [
                start_date.isoformat(),
                end_date.isoformat()
            ]).fetchall()
            
            if not results:
                return {'dates': [], 'diff_bps': [], 'statuses': [], 'matched_trades': []}
            
            # Prepare data for charting
            dates = []
            diff_bps = []
            statuses = []
            matched_trades = []
            
            for row in results:
                dates.append(row[0])
                diff_bps.append(float(row[2]))
                statuses.append(row[1])
                matched_trades.append(int(row[3]))
            
            return {
                'dates': dates,
                'diff_bps': diff_bps,
                'statuses': statuses,
                'matched_trades': matched_trades,
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get trend data: {e}")
            return {'error': str(e)}
    
    def get_reconciliation_health_score(self) -> Dict[str, Any]:
        """Calculate overall reconciliation health score (0-100)"""
        try:
            # Look at last 7 days
            summary = self.get_status_summary(days_back=7)
            
            if 'error' in summary:
                return {'health_score': 0, 'grade': 'F', 'message': 'Unable to calculate health score'}
            
            score = 100
            grade = 'A'
            issues = []
            
            # Deduct points for failures
            pass_rate = summary['summary']['pass_rate']
            if pass_rate < 100:
                deduction = (100 - pass_rate) * 0.8  # Up to 80 points for pass rate
                score -= deduction
                issues.append(f"Pass rate: {pass_rate}%")
            
            # Deduct points for high differences
            avg_diff = summary['summary']['avg_diff_bps']
            if avg_diff > 5:
                deduction = min(10, (avg_diff - 5) * 2)  # Up to 10 points for high diffs
                score -= deduction
                issues.append(f"Avg difference: {avg_diff}bp")
            
            # Deduct points for missing PDFs
            pdf_rate = summary['summary']['pdf_completion_rate']
            if pdf_rate < 100:
                deduction = (100 - pdf_rate) * 0.1  # Up to 10 points for missing PDFs
                score -= deduction
                issues.append(f"PDF completion: {pdf_rate}%")
            
            # Deduct points for stale data
            days_since = summary['recent_status']['days_since_last']
            if days_since and days_since > 1:
                deduction = min(10, days_since * 2)
                score -= deduction
                issues.append(f"Data staleness: {days_since} days")
            
            # Determine grade
            score = max(0, score)  # Floor at 0
            if score >= 90:
                grade = 'A'
            elif score >= 80:
                grade = 'B'
            elif score >= 70:
                grade = 'C'
            elif score >= 60:
                grade = 'D'
            else:
                grade = 'F'
            
            # Generate message
            if score >= 95:
                message = "Excellent reconciliation health"
            elif score >= 85:
                message = "Good reconciliation health"
            elif score >= 70:
                message = "Acceptable reconciliation health"
            elif score >= 50:
                message = "Poor reconciliation health"
            else:
                message = "Critical reconciliation issues"
            
            return {
                'health_score': round(score, 1),
                'grade': grade,
                'message': message,
                'issues': issues,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return {'health_score': 0, 'grade': 'F', 'message': f'Error: {e}'}
    
    def close(self):
        """Close database connections"""
        if self.storage:
            self.storage.close()


def get_reconciliation_dashboard_data(days_back: int = 7) -> Dict[str, Any]:
    """
    Convenience function to get all reconciliation dashboard data
    
    Args:
        days_back: Number of days to look back for analysis
        
    Returns:
        Complete dashboard data dictionary
    """
    card = ReconciliationStatusCard()
    
    try:
        status_summary = card.get_status_summary(days_back)
        trend_data = card.get_daily_trend_data(days_back * 4)  # More data for trends
        health_score = card.get_reconciliation_health_score()
        
        return {
            'status_summary': status_summary,
            'trend_data': trend_data,
            'health_score': health_score,
            'generated_at': datetime.now().isoformat()
        }
        
    finally:
        card.close()


if __name__ == "__main__":
    # Example usage
    data = get_reconciliation_dashboard_data()
    print(json.dumps(data, indent=2, default=str))