"""
Audit PDF Report Generator

Generates PDF audit reports for trade reconciliation with upload to S3.
Creates professional PDF reports with cover page, diff tables, and sign-off page.
"""

import logging
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        REPORTLAB_AVAILABLE = True
        FPDF_AVAILABLE = False
    except ImportError:
        REPORTLAB_AVAILABLE = False
        FPDF_AVAILABLE = False

from ..datasource.storage import DataStorage

logger = logging.getLogger(__name__)


class AuditReportGenerator:
    """Generate PDF audit reports for trade reconciliation"""
    
    def __init__(self, db_path: Optional[str] = None, s3_config: Optional[Dict[str, str]] = None):
        """
        Initialize audit report generator
        
        Args:
            db_path: Path to database file
            s3_config: S3 configuration dict with bucket, prefix, etc.
        """
        self.storage = DataStorage(db_path)
        self.s3_config = s3_config or self._default_s3_config()
        
        # Check PDF library availability
        if not FPDF_AVAILABLE and not REPORTLAB_AVAILABLE:
            logger.warning("No PDF library available (fpdf2 or reportlab). Install one for PDF generation.")
        
        # Initialize S3 client if configured
        self.s3_client = None
        if self.s3_config.get('enabled', False):
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.s3_config.get('access_key'),
                    aws_secret_access_key=self.s3_config.get('secret_key'),
                    region_name=self.s3_config.get('region', 'us-east-1')
                )
                logger.info(f"S3 client initialized for bucket: {self.s3_config['bucket']}")
            except (NoCredentialsError, ClientError) as e:
                logger.warning(f"Failed to initialize S3 client: {e}")
                self.s3_client = None
    
    def _default_s3_config(self) -> Dict[str, Any]:
        """Default S3 configuration"""
        return {
            'enabled': False,
            'bucket': 'mechexo-audit',
            'prefix': 'reconciliation',
            'region': 'us-east-1',
            'access_key': None,
            'secret_key': None
        }
    
    def generate_audit_pdf(self, recon_date: date, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive audit PDF for reconciliation date
        
        Args:
            recon_date: Date to generate audit report for
            output_path: Optional output path (default: auto-generate)
            
        Returns:
            Dict with generation results and paths
        """
        logger.info(f"Generating audit PDF for {recon_date}")
        
        if not FPDF_AVAILABLE and not REPORTLAB_AVAILABLE:
            raise RuntimeError("No PDF library available. Install fpdf2 or reportlab.")
        
        try:
            # Get reconciliation data
            recon_data = self._load_reconciliation_data(recon_date)
            
            if not recon_data:
                logger.warning(f"No reconciliation data found for {recon_date}")
                return {'success': False, 'error': 'No reconciliation data found'}
            
            # Generate output path if not provided
            if not output_path:
                output_path = f"data/audit/recon_{recon_date.strftime('%Y%m%d')}.pdf"
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Generate PDF using available library
            if REPORTLAB_AVAILABLE:
                success = self._generate_pdf_reportlab(recon_data, output_path, recon_date)
            else:
                success = self._generate_pdf_fpdf(recon_data, output_path, recon_date)
            
            if not success:
                return {'success': False, 'error': 'PDF generation failed'}
            
            # Upload to S3 if configured
            s3_url = None
            if self.s3_client and self.s3_config.get('enabled', False):
                s3_url = self._upload_to_s3(output_path, recon_date)
            
            # Update database with PDF path and S3 URL
            self._update_daily_recon_paths(recon_date, output_path, s3_url)
            
            logger.info(f"✅ Audit PDF generated: {output_path}")
            
            return {
                'success': True,
                'pdf_path': output_path,
                's3_url': s3_url,
                'recon_date': recon_date.isoformat(),
                'file_size': Path(output_path).stat().st_size if Path(output_path).exists() else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to generate audit PDF: {e}")
            return {'success': False, 'error': str(e)}
    
    def _load_reconciliation_data(self, recon_date: date) -> Optional[Dict[str, Any]]:
        """Load reconciliation data for the specified date"""
        try:
            # Get daily reconciliation summary
            daily_query = """
            SELECT * FROM daily_recon 
            WHERE recon_date = ?
            """
            daily_result = self.storage.conn.execute(daily_query, [recon_date.isoformat()]).fetchone()
            
            if not daily_result:
                return None
            
            # Convert to dict
            daily_data = dict(daily_result)
            
            # Get detailed audit records
            audit_query = """
            SELECT * FROM reconciliation_audit 
            WHERE recon_date = ?
            ORDER BY match_type, symbol, quantity
            """
            audit_results = self.storage.conn.execute(audit_query, [recon_date.isoformat()]).fetchall()
            
            # Get column names for audit table
            audit_columns = [desc[0] for desc in self.storage.conn.execute(audit_query, [recon_date.isoformat()]).description]
            audit_data = [dict(zip(audit_columns, row)) for row in audit_results]
            
            return {
                'daily_summary': daily_data,
                'audit_records': audit_data,
                'recon_date': recon_date
            }
            
        except Exception as e:
            logger.error(f"Failed to load reconciliation data for {recon_date}: {e}")
            return None
    
    def _generate_pdf_reportlab(self, recon_data: Dict[str, Any], output_path: str, recon_date: date) -> bool:
        """Generate PDF using ReportLab library"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            
            # Cover page
            story.append(Paragraph("Trade Reconciliation Audit Report", title_style))
            story.append(Spacer(1, 0.5*inch))
            
            story.append(Paragraph(f"<b>Reconciliation Date:</b> {recon_date.strftime('%B %d, %Y')}", styles['Normal']))
            story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Paragraph(f"<b>Status:</b> {recon_data['daily_summary'].get('status', 'Unknown').upper()}", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Executive summary
            daily = recon_data['daily_summary']
            summary_data = [
                ['Metric', 'Value'],
                ['Internal Trades', f"{daily.get('internal_trades', 0):,}"],
                ['Broker Trades', f"{daily.get('broker_trades', 0):,}"],
                ['Matched Trades', f"{daily.get('matched_trades', 0):,}"],
                ['Unmatched Internal', f"{daily.get('unmatched_internal', 0):,}"],
                ['Unmatched Broker', f"{daily.get('unmatched_broker', 0):,}"],
                ['Total Difference (bps)', f"{daily.get('total_diff_bps', 0):.1f}"],
                ['Commission Difference', f"${daily.get('commission_diff_usd', 0):.2f}"],
                ['Net Cash Difference', f"${daily.get('net_cash_diff_usd', 0):.2f}"]
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Executive Summary", heading_style))
            story.append(summary_table)
            story.append(PageBreak())
            
            # Detailed differences table
            if recon_data['audit_records']:
                story.append(Paragraph("Detailed Trade Analysis", heading_style))
                
                # Create differences table
                audit_data = [
                    ['Symbol', 'Qty', 'Match Type', 'Price Diff', 'Comm Diff', 'Net Cash Diff']
                ]
                
                for record in recon_data['audit_records'][:50]:  # Limit to first 50 records
                    audit_data.append([
                        record.get('symbol', ''),
                        f"{record.get('quantity', 0):.0f}",
                        record.get('match_type', ''),
                        f"${record.get('price_diff', 0):.2f}" if record.get('price_diff') else '-',
                        f"${record.get('commission_diff', 0):.2f}" if record.get('commission_diff') else '-',
                        f"${record.get('net_cash_diff', 0):.2f}" if record.get('net_cash_diff') else '-'
                    ])
                
                audit_table = Table(audit_data, colWidths=[1*inch, 0.8*inch, 1.2*inch, 1*inch, 1*inch, 1*inch])
                audit_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(audit_table)
                story.append(PageBreak())
            
            # Sign-off page
            story.append(Paragraph("Reconciliation Sign-off", heading_style))
            story.append(Spacer(1, 0.3*inch))
            
            story.append(Paragraph("This reconciliation has been reviewed and approved:", styles['Normal']))
            story.append(Spacer(1, 0.5*inch))
            
            signoff_data = [
                ['Role', 'Name', 'Signature', 'Date'],
                ['Operations Manager', '________________', '________________', '________________'],
                ['Risk Manager', '________________', '________________', '________________'],
                ['Compliance Officer', '________________', '________________', '________________']
            ]
            
            signoff_table = Table(signoff_data, colWidths=[2*inch, 2*inch, 2*inch, 1.5*inch])
            signoff_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            story.append(signoff_table)
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph("<i>Automated reconciliation report generated by Mech-Exo Trading System</i>", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            logger.info(f"✅ PDF generated using ReportLab: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate PDF with ReportLab: {e}")
            return False
    
    def _generate_pdf_fpdf(self, recon_data: Dict[str, Any], output_path: str, recon_date: date) -> bool:
        """Generate PDF using FPDF library (fallback)"""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            
            # Cover page
            pdf.cell(0, 10, 'Trade Reconciliation Audit Report', 0, 1, 'C')
            pdf.ln(10)
            
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f"Reconciliation Date: {recon_date.strftime('%B %d, %Y')}", 0, 1)
            pdf.cell(0, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
            pdf.cell(0, 10, f"Status: {recon_data['daily_summary'].get('status', 'Unknown').upper()}", 0, 1)
            pdf.ln(10)
            
            # Executive summary
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Executive Summary', 0, 1)
            pdf.ln(5)
            
            pdf.set_font('Arial', '', 10)
            daily = recon_data['daily_summary']
            
            summary_items = [
                f"Internal Trades: {daily.get('internal_trades', 0):,}",
                f"Broker Trades: {daily.get('broker_trades', 0):,}",
                f"Matched Trades: {daily.get('matched_trades', 0):,}",
                f"Total Difference: {daily.get('total_diff_bps', 0):.1f} basis points",
                f"Commission Difference: ${daily.get('commission_diff_usd', 0):.2f}",
                f"Net Cash Difference: ${daily.get('net_cash_diff_usd', 0):.2f}"
            ]
            
            for item in summary_items:
                pdf.cell(0, 8, item, 0, 1)
            
            pdf.output(output_path)
            logger.info(f"✅ PDF generated using FPDF: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate PDF with FPDF: {e}")
            return False
    
    def _upload_to_s3(self, file_path: str, recon_date: date) -> Optional[str]:
        """Upload PDF to S3 and return public URL"""
        if not self.s3_client:
            logger.warning("S3 client not configured, skipping upload")
            return None
        
        try:
            bucket = self.s3_config['bucket']
            key = f"{self.s3_config['prefix']}/recon_{recon_date.strftime('%Y%m%d')}.pdf"
            
            # Upload file
            with open(file_path, 'rb') as f:
                self.s3_client.upload_fileobj(
                    f, 
                    bucket, 
                    key,
                    ExtraArgs={'ContentType': 'application/pdf'}
                )
            
            # Generate URL
            s3_url = f"https://{bucket}.s3.amazonaws.com/{key}"
            logger.info(f"✅ PDF uploaded to S3: {s3_url}")
            return s3_url
            
        except Exception as e:
            logger.error(f"Failed to upload PDF to S3: {e}")
            return None
    
    def _update_daily_recon_paths(self, recon_date: date, pdf_path: str, s3_url: Optional[str]):
        """Update daily_recon table with PDF path and S3 URL"""
        try:
            update_query = """
            UPDATE daily_recon 
            SET pdf_path = ?, s3_url = ?, updated_at = ?
            WHERE recon_date = ?
            """
            
            self.storage.conn.execute(update_query, [
                pdf_path,
                s3_url,
                datetime.now().isoformat(),
                recon_date.isoformat()
            ])
            
            self.storage.conn.commit()
            logger.info(f"✅ Updated daily_recon with PDF paths for {recon_date}")
            
        except Exception as e:
            logger.error(f"Failed to update daily_recon paths: {e}")
    
    def generate_all_missing_pdfs(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate PDFs for all dates missing PDF reports"""
        logger.info(f"Generating missing PDFs from {start_date} to {end_date}")
        
        try:
            # Find dates missing PDFs
            query = """
            SELECT recon_date FROM daily_recon 
            WHERE recon_date BETWEEN ? AND ?
            AND (pdf_path IS NULL OR pdf_path = '')
            ORDER BY recon_date
            """
            
            missing_dates = self.storage.conn.execute(query, [
                start_date.isoformat(),
                end_date.isoformat()
            ]).fetchall()
            
            if not missing_dates:
                logger.info("No missing PDFs found")
                return {'success': True, 'generated_count': 0, 'missing_dates': []}
            
            results = []
            success_count = 0
            
            for (date_str,) in missing_dates:
                recon_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                result = self.generate_audit_pdf(recon_date)
                results.append({
                    'date': date_str,
                    'success': result['success'],
                    'pdf_path': result.get('pdf_path'),
                    's3_url': result.get('s3_url'),
                    'error': result.get('error')
                })
                
                if result['success']:
                    success_count += 1
            
            logger.info(f"✅ Generated {success_count}/{len(missing_dates)} PDFs")
            
            return {
                'success': True,
                'generated_count': success_count,
                'total_missing': len(missing_dates),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Failed to generate missing PDFs: {e}")
            return {'success': False, 'error': str(e)}
    
    def close(self):
        """Close database connections"""
        if self.storage:
            self.storage.close()


def generate_audit_pdf_for_date(recon_date: str, output_dir: Optional[str] = None, 
                               s3_config: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Convenience function to generate audit PDF for a specific date
    
    Args:
        recon_date: Date string (YYYY-MM-DD)
        output_dir: Optional output directory
        s3_config: Optional S3 configuration
        
    Returns:
        Generation results dictionary
    """
    from datetime import datetime
    
    date_obj = datetime.strptime(recon_date, '%Y-%m-%d').date()
    
    generator = AuditReportGenerator(s3_config=s3_config)
    
    try:
        output_path = None
        if output_dir:
            output_path = f"{output_dir}/recon_{date_obj.strftime('%Y%m%d')}.pdf"
        
        return generator.generate_audit_pdf(date_obj, output_path)
        
    finally:
        generator.close()


if __name__ == "__main__":
    # Example usage
    from datetime import date, timedelta
    
    test_date = date.today() - timedelta(days=1)
    result = generate_audit_pdf_for_date(test_date.isoformat())
    print(f"PDF generation result: {result}")