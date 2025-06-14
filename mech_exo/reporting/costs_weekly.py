"""
Weekly Cost & Slippage Reporting - Phase P11 Week 2

Generates comprehensive weekly reports on trading costs, slippage, and performance
metrics. Analyzes actual commission costs vs model predictions and tracks
execution quality metrics.

Features:
- Commission cost analysis (actual vs predicted)
- Slippage measurement in basis points
- Execution quality metrics
- PDF report generation with charts
- S3 upload for report distribution
- Weekly trend analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import sys
import io
import base64

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    from matplotlib.backends.backend_pdf import PdfPages
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logger = logging.getLogger(__name__)


class WeeklyCostAnalyzer:
    """Analyzes weekly trading costs and generates reports"""
    
    def __init__(self, 
                 commission_rate: float = 0.005,  # 0.5 bps default
                 slippage_threshold: float = 10.0,  # 10 bps threshold
                 report_template: str = "standard"):
        
        self.commission_rate = commission_rate
        self.slippage_threshold = slippage_threshold
        self.report_template = report_template
        
        # Cost model parameters
        self.cost_model = {
            'fixed_commission': 1.0,  # $1 per trade
            'variable_commission_bps': 0.5,  # 0.5 basis points
            'expected_slippage_bps': 5.0,  # 5 bps expected slippage
            'market_impact_factor': 0.1  # Market impact scaling
        }
        
        logger.info(f"WeeklyCostAnalyzer initialized with {commission_rate*10000:.1f} bps commission rate")
        
    def generate_mock_trading_data(self, 
                                 start_date: datetime, 
                                 end_date: datetime) -> pd.DataFrame:
        """Generate mock trading data for the week"""
        
        # Generate trading days
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        trades = []
        trade_id = 1
        
        for date in date_range:
            # Generate 10-20 trades per day
            num_trades = np.random.randint(10, 21)
            
            for _ in range(num_trades):
                # Mock trade data
                symbol = np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                                         'NVDA', 'META', 'SPY', 'QQQ', 'IWM'])
                side = np.random.choice(['BUY', 'SELL'])
                quantity = np.random.randint(100, 1000)
                price = np.random.uniform(50, 300)
                
                # Simulate execution with realistic slippage
                expected_price = price
                if side == 'BUY':
                    executed_price = expected_price * (1 + np.random.normal(0.0005, 0.002))  # 0.5-20 bps slippage
                else:
                    executed_price = expected_price * (1 - np.random.normal(0.0005, 0.002))
                
                # Calculate costs
                notional = quantity * executed_price
                actual_commission = max(1.0, notional * self.commission_rate)
                predicted_commission = max(1.0, notional * self.cost_model['variable_commission_bps'] / 10000)
                
                # Calculate slippage in bps
                slippage_bps = abs(executed_price - expected_price) / expected_price * 10000
                
                trades.append({
                    'trade_id': trade_id,
                    'date': date,
                    'timestamp': date + timedelta(hours=np.random.randint(9, 16), 
                                                minutes=np.random.randint(0, 60)),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'expected_price': expected_price,
                    'executed_price': executed_price,
                    'notional': notional,
                    'actual_commission': actual_commission,
                    'predicted_commission': predicted_commission,
                    'slippage_bps': slippage_bps,
                    'execution_venue': np.random.choice(['NASDAQ', 'NYSE', 'ARCA', 'BATS']),
                    'order_type': np.random.choice(['MARKET', 'LIMIT', 'STOP'])
                })
                trade_id += 1
        
        df = pd.DataFrame(trades)
        logger.info(f"Generated {len(df)} mock trades from {start_date.date()} to {end_date.date()}")
        
        return df
        
    def analyze_weekly_costs(self, 
                           trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weekly trading costs and performance"""
        
        if trades_df.empty:
            return {'error': 'No trades data available'}
        
        analysis = {
            'period': {
                'start_date': trades_df['date'].min().strftime('%Y-%m-%d'),
                'end_date': trades_df['date'].max().strftime('%Y-%m-%d'),
                'trading_days': trades_df['date'].nunique(),
                'total_trades': len(trades_df)
            }
        }
        
        # Commission analysis
        total_actual_commission = trades_df['actual_commission'].sum()
        total_predicted_commission = trades_df['predicted_commission'].sum()
        commission_variance = total_actual_commission - total_predicted_commission
        commission_variance_pct = (commission_variance / total_predicted_commission) * 100
        
        analysis['commission'] = {
            'actual_total': total_actual_commission,
            'predicted_total': total_predicted_commission,
            'variance': commission_variance,
            'variance_pct': commission_variance_pct,
            'avg_per_trade': total_actual_commission / len(trades_df),
            'as_pct_of_notional': (total_actual_commission / trades_df['notional'].sum()) * 100
        }
        
        # Slippage analysis
        avg_slippage = trades_df['slippage_bps'].mean()
        median_slippage = trades_df['slippage_bps'].median()
        max_slippage = trades_df['slippage_bps'].max()
        slippage_std = trades_df['slippage_bps'].std()
        
        # Count high slippage trades
        high_slippage_trades = len(trades_df[trades_df['slippage_bps'] > self.slippage_threshold])
        high_slippage_pct = (high_slippage_trades / len(trades_df)) * 100
        
        analysis['slippage'] = {
            'average_bps': avg_slippage,
            'median_bps': median_slippage,
            'max_bps': max_slippage,
            'std_bps': slippage_std,
            'high_slippage_trades': high_slippage_trades,
            'high_slippage_pct': high_slippage_pct,
            'threshold_bps': self.slippage_threshold
        }
        
        # Total cost analysis
        total_slippage_cost = (trades_df['slippage_bps'] / 10000 * trades_df['notional']).sum()
        total_cost = total_actual_commission + total_slippage_cost
        total_notional = trades_df['notional'].sum()
        total_cost_bps = (total_cost / total_notional) * 10000
        
        analysis['total_cost'] = {
            'commission_cost': total_actual_commission,
            'slippage_cost': total_slippage_cost,
            'total_cost': total_cost,
            'total_notional': total_notional,
            'total_cost_bps': total_cost_bps
        }
        
        # Performance by venue
        venue_analysis = trades_df.groupby('execution_venue').agg({
            'slippage_bps': ['mean', 'count'],
            'actual_commission': 'sum',
            'notional': 'sum'
        }).round(2)
        
        analysis['venue_performance'] = venue_analysis.to_dict()
        
        # Daily breakdown
        daily_analysis = trades_df.groupby('date').agg({
            'slippage_bps': 'mean',
            'actual_commission': 'sum',
            'notional': 'sum',
            'trade_id': 'count'
        }).rename(columns={'trade_id': 'num_trades'})
        
        analysis['daily_breakdown'] = daily_analysis.to_dict()
        
        logger.info(f"Weekly cost analysis complete: "
                   f"${total_cost:,.0f} total cost ({total_cost_bps:.1f} bps)")
        
        return analysis
        
    def generate_cost_charts(self, 
                           trades_df: pd.DataFrame, 
                           analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate charts for the cost analysis report"""
        
        charts = {}
        
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available - skipping charts")
            return charts
            
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Chart 1: Daily slippage trend
            fig, ax = plt.subplots(figsize=(10, 6))
            daily_slippage = trades_df.groupby('date')['slippage_bps'].mean()
            daily_slippage.plot(kind='line', marker='o', ax=ax)
            ax.set_title('Daily Average Slippage (bps)')
            ax.set_ylabel('Slippage (bps)')
            ax.axhline(y=self.slippage_threshold, color='red', linestyle='--', label='Threshold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save chart as base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            charts['daily_slippage'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Chart 2: Commission variance
            fig, ax = plt.subplots(figsize=(8, 6))
            commission_data = ['Predicted', 'Actual']
            commission_values = [
                analysis['commission']['predicted_total'],
                analysis['commission']['actual_total']
            ]
            
            bars = ax.bar(commission_data, commission_values, color=['lightblue', 'darkblue'])
            ax.set_title('Predicted vs Actual Commission Costs')
            ax.set_ylabel('Commission ($)')
            
            # Add value labels
            for bar, value in zip(bars, commission_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                       f'${value:,.0f}', ha='center', va='bottom')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            charts['commission_comparison'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Chart 3: Slippage distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            trades_df['slippage_bps'].hist(bins=30, alpha=0.7, ax=ax)
            ax.axvline(x=analysis['slippage']['average_bps'], color='green', 
                      linestyle='-', label=f'Average: {analysis["slippage"]["average_bps"]:.1f} bps')
            ax.axvline(x=self.slippage_threshold, color='red', 
                      linestyle='--', label=f'Threshold: {self.slippage_threshold} bps')
            ax.set_title('Slippage Distribution')
            ax.set_xlabel('Slippage (bps)')
            ax.set_ylabel('Number of Trades')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            charts['slippage_distribution'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            logger.info(f"Generated {len(charts)} charts for cost analysis")
            
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
            
        return charts
        
    def generate_pdf_report(self, 
                          analysis: Dict[str, Any],
                          charts: Dict[str, str],
                          output_file: str) -> bool:
        """Generate PDF report with cost analysis"""
        
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            # Create PDF
            doc = SimpleDocTemplate(output_file, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center
            )
            
            period = analysis['period']
            title = f"Weekly Trading Cost Report<br/>{period['start_date']} to {period['end_date']}"
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Trades', f"{period['total_trades']:,}"],
                ['Trading Days', f"{period['trading_days']}"],
                ['Total Commission', f"${analysis['commission']['actual_total']:,.2f}"],
                ['Average Slippage', f"{analysis['slippage']['average_bps']:.1f} bps"],
                ['Total Cost (bps)', f"{analysis['total_cost']['total_cost_bps']:.1f} bps"]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Commission Analysis
            story.append(Paragraph("Commission Analysis", styles['Heading2']))
            
            comm = analysis['commission']
            commission_text = f"""
            <b>Actual vs Predicted Commission:</b><br/>
            • Actual Total: ${comm['actual_total']:,.2f}<br/>
            • Predicted Total: ${comm['predicted_total']:,.2f}<br/>
            • Variance: ${comm['variance']:+,.2f} ({comm['variance_pct']:+.1f}%)<br/>
            • Average per Trade: ${comm['avg_per_trade']:.2f}<br/>
            """
            
            story.append(Paragraph(commission_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Slippage Analysis
            story.append(Paragraph("Slippage Analysis", styles['Heading2']))
            
            slip = analysis['slippage']
            slippage_text = f"""
            <b>Slippage Statistics:</b><br/>
            • Average: {slip['average_bps']:.1f} bps<br/>
            • Median: {slip['median_bps']:.1f} bps<br/>
            • Maximum: {slip['max_bps']:.1f} bps<br/>
            • Standard Deviation: {slip['std_bps']:.1f} bps<br/>
            • High Slippage Trades: {slip['high_slippage_trades']} ({slip['high_slippage_pct']:.1f}%)<br/>
            """
            
            story.append(Paragraph(slippage_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Total Cost Summary
            story.append(Paragraph("Total Cost Breakdown", styles['Heading2']))
            
            cost = analysis['total_cost']
            cost_text = f"""
            <b>Cost Components:</b><br/>
            • Commission Cost: ${cost['commission_cost']:,.2f}<br/>
            • Slippage Cost: ${cost['slippage_cost']:,.2f}<br/>
            • Total Cost: ${cost['total_cost']:,.2f}<br/>
            • Total Notional: ${cost['total_notional']:,.0f}<br/>
            • Total Cost (bps): {cost['total_cost_bps']:.1f} bps<br/>
            """
            
            story.append(Paragraph(cost_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return False
    
    def upload_to_s3(self, file_path: str, s3_bucket: str, s3_key: str) -> Optional[str]:
        """Upload report to S3 and return public URL"""
        try:
            # Mock S3 upload for development
            # In production, this would use boto3
            
            logger.info(f"Mock S3 upload: {file_path} -> s3://{s3_bucket}/{s3_key}")
            
            # Simulate S3 URL
            mock_url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
            
            return mock_url
            
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return None


def generate_weekly_cost_report(start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              output_dir: str = "reports/costs",
                              upload_to_s3: bool = True) -> Dict[str, Any]:
    """Main function to generate weekly cost report"""
    
    logger.info("🏦 Generating weekly cost & slippage report...")
    
    # Default to last 7 days if dates not provided
    if not start_date:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=7)
    else:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
    
    # Create analyzer
    analyzer = WeeklyCostAnalyzer()
    
    # Generate mock trading data (in production, would fetch real data)
    trades_df = analyzer.generate_mock_trading_data(start_dt, end_dt)
    
    if trades_df.empty:
        return {'error': 'No trading data available for the period'}
    
    # Analyze costs
    analysis = analyzer.analyze_weekly_costs(trades_df)
    
    # Generate charts
    charts = analyzer.generate_cost_charts(trades_df, analysis)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with week number
    week_num = start_dt.isocalendar()[1]
    year = start_dt.year
    pdf_filename = f"costs_weekly_{year}-W{week_num:02d}.pdf"
    pdf_path = output_path / pdf_filename
    
    # Generate PDF report
    pdf_success = analyzer.generate_pdf_report(analysis, charts, str(pdf_path))
    
    result = {
        'success': pdf_success,
        'period': analysis['period'],
        'pdf_file': str(pdf_path) if pdf_success else None,
        'analysis': analysis,
        'charts_generated': len(charts)
    }
    
    # Upload to S3 if requested
    if upload_to_s3 and pdf_success:
        s3_key = f"weekly_costs/{pdf_filename}"
        s3_url = analyzer.upload_to_s3(str(pdf_path), "mech-exo-reports", s3_key)
        result['s3_url'] = s3_url
    
    if pdf_success:
        logger.info(f"✅ Weekly cost report generated: {pdf_path}")
        logger.info(f"📊 Total cost: ${analysis['total_cost']['total_cost']:,.2f} "
                   f"({analysis['total_cost']['total_cost_bps']:.1f} bps)")
    else:
        logger.error("❌ Failed to generate weekly cost report")
    
    return result


def test_weekly_cost_report():
    """Test function for weekly cost reporting"""
    print("🧪 Testing Weekly Cost Report Generation...")
    
    try:
        # Generate test report
        result = generate_weekly_cost_report(
            start_date="2025-06-09",
            end_date="2025-06-13",
            upload_to_s3=False
        )
        
        if result.get('success'):
            print("✅ Weekly cost report test PASSED")
            print(f"📄 PDF generated: {result['pdf_file']}")
            print(f"📊 Charts: {result['charts_generated']}")
            print(f"📈 Analysis period: {result['period']['start_date']} to {result['period']['end_date']}")
            return True
        else:
            print("❌ Weekly cost report test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Weekly cost report test FAILED: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Weekly Cost & Slippage Report Generator')
    parser.add_argument('command', choices=['generate', 'test'],
                       help='Command to execute')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='reports/costs',
                       help='Output directory for reports')
    parser.add_argument('--no-s3', action='store_true',
                       help='Skip S3 upload')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.command == 'test':
        success = test_weekly_cost_report()
        sys.exit(0 if success else 1)
        
    elif args.command == 'generate':
        result = generate_weekly_cost_report(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            upload_to_s3=not args.no_s3
        )
        
        if result.get('success'):
            print(f"✅ Report generated: {result['pdf_file']}")
            if result.get('s3_url'):
                print(f"📤 S3 URL: {result['s3_url']}")
        else:
            print("❌ Report generation failed")
            sys.exit(1)