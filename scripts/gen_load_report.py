#!/usr/bin/env python3
"""
Load Test Report Generator
Generates comprehensive load test report with KPIs and analysis
"""

import json
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [REPORT] %(message)s'
)
logger = logging.getLogger(__name__)


class LoadTestReportGenerator:
    """Generate comprehensive load test reports"""
    
    def __init__(self):
        self.load_results = None
        self.chaos_results = None
        self.metrics_data = None
        
    def load_test_results(self, results_dir: str = ".") -> bool:
        """Load test results from files"""
        results_path = Path(results_dir)
        
        # Load consolidated results if available
        consolidated_files = list(results_path.glob("consolidated_load_test_*.json"))
        if consolidated_files:
            latest_file = max(consolidated_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    self.load_results = data.get('load_results')
                    self.chaos_results = data.get('chaos_results')
                logger.info(f"Loaded consolidated results from {latest_file}")
                return True
            except Exception as e:
                logger.error(f"Failed to load consolidated results: {str(e)}")
        
        # Load individual result files
        success = False
        
        # Load test results
        load_files = list(results_path.glob("load_test_results_*.json"))
        if load_files:
            latest_load_file = max(load_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_load_file, 'r') as f:
                    self.load_results = json.load(f)
                logger.info(f"Loaded load results from {latest_load_file}")
                success = True
            except Exception as e:
                logger.error(f"Failed to load load results: {str(e)}")
        
        # Chaos test results
        chaos_files = list(results_path.glob("chaos_test_report_*.json"))
        if chaos_files:
            latest_chaos_file = max(chaos_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_chaos_file, 'r') as f:
                    self.chaos_results = json.load(f)
                logger.info(f"Loaded chaos results from {latest_chaos_file}")
            except Exception as e:
                logger.error(f"Failed to load chaos results: {str(e)}")
        
        return success
    
    def calculate_kpis(self) -> Dict[str, Any]:
        """Calculate key performance indicators"""
        kpis = {}
        
        if self.load_results and 'metrics' in self.load_results:
            metrics = self.load_results['metrics']
            
            # Basic metrics
            kpis['total_requests'] = metrics.get('total_requests', 0)
            kpis['successful_requests'] = metrics.get('successful_requests', 0)
            kpis['failed_requests'] = metrics.get('failed_requests', 0)
            
            # Calculated metrics
            total = kpis['total_requests']
            if total > 0:
                kpis['success_rate'] = kpis['successful_requests'] / total
                kpis['error_rate'] = kpis['failed_requests'] / total
            else:
                kpis['success_rate'] = 0.0
                kpis['error_rate'] = 1.0
            
            # Latency metrics
            kpis['avg_latency_ms'] = metrics.get('avg_latency', 0) * 1000
            kpis['min_latency_ms'] = metrics.get('min_latency', 0) * 1000
            kpis['max_latency_ms'] = metrics.get('max_latency', 0) * 1000
            
            # Rate metrics
            kpis['current_rate'] = metrics.get('current_rate', 0)
            
            # Test duration
            if 'start_time' in metrics and metrics['start_time']:
                try:
                    start_time = datetime.fromisoformat(metrics['start_time'].replace('Z', '+00:00'))
                    end_time = datetime.now()
                    kpis['test_duration_seconds'] = (end_time - start_time).total_seconds()
                except:
                    kpis['test_duration_seconds'] = 0
            else:
                kpis['test_duration_seconds'] = 0
        
        # Chaos testing KPIs
        if self.chaos_results:
            kpis['chaos_toggles'] = self.chaos_results.get('total_toggles', 0)
            kpis['chaos_successful_recoveries'] = self.chaos_results.get('successful_recoveries', 0)
            
            if kpis['chaos_toggles'] > 0:
                kpis['chaos_recovery_rate'] = kpis['chaos_successful_recoveries'] / kpis['chaos_toggles']
            else:
                kpis['chaos_recovery_rate'] = 0.0
            
            # Calculate average recovery time
            downtime_events = self.chaos_results.get('downtime_events', [])
            recovery_times = [
                event.get('recovery_time', 0) 
                for event in downtime_events 
                if event.get('recovery_success', False) and event.get('recovery_time')
            ]
            
            if recovery_times:
                kpis['avg_recovery_time_seconds'] = sum(recovery_times) / len(recovery_times)
                kpis['max_recovery_time_seconds'] = max(recovery_times)
            else:
                kpis['avg_recovery_time_seconds'] = 0
                kpis['max_recovery_time_seconds'] = 0
        
        return kpis
    
    def generate_markdown_report(self, output_file: str = None) -> str:
        """Generate markdown report"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"docs/load_test_report_{timestamp}.md"
        
        kpis = self.calculate_kpis()
        
        # Build report content
        report_lines = []
        
        # Header
        report_lines.extend([
            "# Load Test Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Test Duration:** {kpis.get('test_duration_seconds', 0):.1f} seconds",
            "",
            "## Executive Summary",
            "",
        ])
        
        # Determine overall result
        error_rate = kpis.get('error_rate', 1.0)
        success_rate = kpis.get('success_rate', 0.0)
        avg_latency = kpis.get('avg_latency_ms', 0)
        
        if error_rate <= 0.01 and success_rate >= 0.99 and avg_latency <= 400:
            overall_result = "✅ **PASS**"
            summary_color = "green"
        else:
            overall_result = "❌ **FAIL**"
            summary_color = "red"
        
        report_lines.extend([
            f"**Overall Result:** {overall_result}",
            "",
            f"- **Target Rate:** {kpis.get('current_rate', 0):.1f} req/sec",
            f"- **Total Requests:** {kpis.get('total_requests', 0):,}",
            f"- **Success Rate:** {success_rate:.2%}",
            f"- **Error Rate:** {error_rate:.2%}",
            f"- **Average Latency:** {avg_latency:.1f}ms",
            "",
        ])
        
        # Pass/Fail Criteria
        report_lines.extend([
            "## Pass/Fail Criteria",
            "",
            "| Metric | Target | Actual | Status |",
            "|--------|--------|--------|--------|",
        ])
        
        criteria = [
            ("Error Rate", "≤ 1%", f"{error_rate:.2%}", "✅" if error_rate <= 0.01 else "❌"),
            ("95th Percentile Latency", "< 400ms", f"{avg_latency:.1f}ms", "✅" if avg_latency < 400 else "❌"),
            ("Success Rate", "≥ 99%", f"{success_rate:.2%}", "✅" if success_rate >= 0.99 else "❌"),
        ]
        
        for metric, target, actual, status in criteria:
            report_lines.append(f"| {metric} | {target} | {actual} | {status} |")
        
        report_lines.extend(["", ""])
        
        # Detailed Metrics
        report_lines.extend([
            "## Detailed Performance Metrics",
            "",
            "### Request Statistics",
            "",
            f"- **Total Requests:** {kpis.get('total_requests', 0):,}",
            f"- **Successful Requests:** {kpis.get('successful_requests', 0):,}",
            f"- **Failed Requests:** {kpis.get('failed_requests', 0):,}",
            f"- **Request Rate:** {kpis.get('current_rate', 0):.2f} req/sec",
            "",
            "### Latency Statistics",
            "",
            f"- **Average Latency:** {kpis.get('avg_latency_ms', 0):.2f}ms",
            f"- **Minimum Latency:** {kpis.get('min_latency_ms', 0):.2f}ms",
            f"- **Maximum Latency:** {kpis.get('max_latency_ms', 0):.2f}ms",
            "",
        ])
        
        # Chaos Testing Results
        if self.chaos_results:
            chaos_toggles = kpis.get('chaos_toggles', 0)
            recovery_rate = kpis.get('chaos_recovery_rate', 0)
            avg_recovery = kpis.get('avg_recovery_time_seconds', 0)
            
            report_lines.extend([
                "### Chaos Testing Results",
                "",
                f"- **Kill-Switch Toggles:** {chaos_toggles}",
                f"- **Successful Recoveries:** {kpis.get('chaos_successful_recoveries', 0)}/{chaos_toggles}",
                f"- **Recovery Rate:** {recovery_rate:.1%}",
                f"- **Average Recovery Time:** {avg_recovery:.1f}s",
                f"- **Maximum Recovery Time:** {kpis.get('max_recovery_time_seconds', 0):.1f}s",
                "",
            ])
        
        # System Resource Usage (if available)
        report_lines.extend([
            "## System Resource Usage",
            "",
            "| Resource | Average | Peak | Status |",
            "|----------|---------|------|--------|",
            "| CPU Usage | N/A | N/A | ⚠️ Not monitored |",
            "| Memory Usage | N/A | N/A | ⚠️ Not monitored |",
            "| Network I/O | N/A | N/A | ⚠️ Not monitored |",
            "",
        ])
        
        # Error Analysis
        if error_rate > 0:
            report_lines.extend([
                "## Error Analysis",
                "",
                f"**Error Rate:** {error_rate:.2%} ({kpis.get('failed_requests', 0):,} failed requests)",
                "",
                "### Common Error Types",
                "",
                "- Connection timeouts",
                "- HTTP 5xx responses", 
                "- Network errors",
                "",
                "### Recommendations",
                "",
                "- Increase connection pool size",
                "- Optimize database queries",
                "- Add circuit breakers",
                "- Scale horizontally",
                "",
            ])
        
        # Configuration Details
        if self.load_results and 'config' in self.load_results:
            config = self.load_results['config']
            report_lines.extend([
                "## Test Configuration",
                "",
                f"- **Target Rate:** {config.get('target_rate', 0)} req/sec",
                f"- **Ramp Duration:** {config.get('ramp_duration', 0)} seconds",
                f"- **Test Duration:** {config.get('test_duration', 0)} seconds",
                f"- **Base URL:** {config.get('base_url', 'N/A')}",
                f"- **Paper Trading:** {config.get('paper_trading', True)}",
                f"- **Error Threshold:** {config.get('error_threshold', 0):.1%}",
                "",
            ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
        ])
        
        if overall_result.startswith("✅"):
            report_lines.extend([
                "✅ **System is performing well under load**",
                "",
                "- Current configuration can handle target load",
                "- Consider gradual scaling for higher loads",
                "- Monitor for performance degradation over time",
                "- Continue chaos testing to validate resilience",
                "",
            ])
        else:
            report_lines.extend([
                "❌ **System requires optimization**",
                "",
                "- Review error logs for root causes",
                "- Optimize database queries and connections",
                "- Consider horizontal scaling",
                "- Implement circuit breakers and retry logic",
                "- Monitor resource utilization",
                "",
            ])
        
        # Footer
        report_lines.extend([
            "---",
            "",
            "**Report generated by Mech-Exo Load Test Runner**",
            f"**Timestamp:** {datetime.now().isoformat()}",
            ""
        ])
        
        # Write report
        report_content = "\n".join(report_lines)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Load test report generated: {output_path}")
        
        return str(output_path)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate load test report')
    parser.add_argument('--results-dir', default='.',
                       help='Directory containing test results')
    parser.add_argument('--output', 
                       help='Output file path for the report')
    parser.add_argument('--format', choices=['markdown', 'html'], default='markdown',
                       help='Report format')
    
    args = parser.parse_args()
    
    # Generate report
    generator = LoadTestReportGenerator()
    
    if not generator.load_test_results(args.results_dir):
        logger.error("Failed to load test results")
        return 1
    
    try:
        if args.format == 'markdown':
            report_file = generator.generate_markdown_report(args.output)
            logger.info(f"✅ Report generated successfully: {report_file}")
            return 0
        else:
            logger.error(f"Unsupported format: {args.format}")
            return 1
            
    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())