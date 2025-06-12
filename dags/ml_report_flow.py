"""
Prefect flow for automated ML SHAP report generation.

Loads latest trained models, generates SHAP feature importance reports,
and stores results as artifacts for dashboard integration.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

logger = logging.getLogger(__name__)


@task(name="find_latest_models", description="Find latest trained ML models")
def find_latest_models(models_dir: str = "models") -> Dict[str, str]:
    """
    Find the most recent trained ML models.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Dictionary mapping algorithm to model file path
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return {}
    
    # Find model files by algorithm
    latest_models = {}
    
    # LightGBM models (.txt files)
    lgb_files = list(models_path.glob("lgbm_*.txt"))
    if lgb_files:
        latest_lgb = max(lgb_files, key=lambda x: x.stat().st_mtime)
        latest_models['lightgbm'] = str(latest_lgb)
        logger.info(f"Found LightGBM model: {latest_lgb}")
    
    # XGBoost models (.json files)
    xgb_files = list(models_path.glob("xgb_*.json"))
    if xgb_files:
        latest_xgb = max(xgb_files, key=lambda x: x.stat().st_mtime)
        latest_models['xgboost'] = str(latest_xgb)
        logger.info(f"Found XGBoost model: {latest_xgb}")
    
    logger.info(f"Found {len(latest_models)} trained models")
    return latest_models


@task(name="load_recent_features", description="Load recent feature data for SHAP analysis")
def load_recent_features(features_dir: str = "data/features", 
                        lookback_days: int = 60) -> Optional[str]:
    """
    Load recent feature data for SHAP analysis.
    
    Args:
        features_dir: Directory containing feature files
        lookback_days: Days of recent data to include
        
    Returns:
        Path to combined feature file or None if no data found
    """
    features_path = Path(features_dir)
    
    if not features_path.exists():
        logger.warning(f"Features directory not found: {features_dir}")
        return None
    
    # Find recent feature files
    recent_files = []
    for days_back in range(lookback_days):
        check_date = datetime.now().date() - timedelta(days=days_back)
        date_str = check_date.strftime('%Y-%m-%d')
        
        csv_file = features_path / f"features_{date_str}.csv"
        parquet_file = features_path / f"features_{date_str}.parquet"
        
        if csv_file.exists():
            recent_files.append(csv_file)
        elif parquet_file.exists():
            recent_files.append(parquet_file)
        
        # Limit to reasonable number of files
        if len(recent_files) >= 30:
            break
    
    if not recent_files:
        logger.warning(f"No recent feature files found in {features_dir}")
        return None
    
    logger.info(f"Found {len(recent_files)} recent feature files")
    
    # For simplicity, return the most recent file path
    # In practice, you might want to combine multiple files
    latest_file = max(recent_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using latest feature file: {latest_file}")
    
    return str(latest_file)


@task(name="generate_shap_report", description="Generate SHAP feature importance report")
def generate_shap_report(model_path: str, 
                        features_file: str,
                        algorithm: str,
                        output_dir: str = "reports/shap") -> Dict[str, str]:
    """
    Generate SHAP report for a trained model.
    
    Args:
        model_path: Path to trained model
        features_file: Path to feature data file
        algorithm: Algorithm type (lightgbm or xgboost)
        output_dir: Output directory for reports
        
    Returns:
        Dictionary with report metadata
    """
    try:
        from mech_exo.ml.report_ml import make_shap_report
        import pandas as pd
        
        logger.info(f"Generating SHAP report for {algorithm} model")
        
        # Load feature data
        if features_file.endswith('.csv'):
            feature_df = pd.read_csv(features_file)
        else:
            feature_df = pd.read_parquet(features_file)
        
        # Prepare feature matrix (remove metadata columns)
        feature_cols = [col for col in feature_df.columns 
                       if col not in ['symbol', 'feature_date', 'feature_count']]
        X = feature_df[feature_cols].copy()
        
        logger.info(f"Feature matrix: {len(X)} samples, {len(X.columns)} features")
        
        # Generate output filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(model_path).stem
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        png_path = str(output_path / f"shap_summary_{model_name}_{timestamp}.png")
        html_path = str(output_path / f"shap_force_{model_name}_{timestamp}.html")
        
        # Generate SHAP report
        metadata = make_shap_report(
            model_path=model_path,
            feature_matrix=X,
            out_html=html_path,
            out_png=png_path,
            top_k=20,
            algorithm=algorithm
        )
        
        # Add task-specific metadata
        metadata.update({
            'task_timestamp': datetime.now().isoformat(),
            'algorithm': algorithm,
            'features_file': features_file,
            'samples_used': len(X)
        })
        
        logger.info(f"SHAP report generated successfully for {algorithm}")
        logger.info(f"  PNG: {png_path} ({metadata['png_size_mb']:.2f} MB)")
        logger.info(f"  HTML: {html_path} ({metadata['html_size_mb']:.2f} MB)")
        
        return metadata
        
    except ImportError as e:
        logger.error(f"SHAP dependencies not available: {e}")
        return {'error': f"Import error: {e}"}
    except Exception as e:
        logger.error(f"SHAP report generation failed: {e}")
        return {'error': str(e)}


@task(name="store_report_metadata", description="Store SHAP report metadata in database")
def store_report_metadata(report_metadata: Dict, algorithm: str) -> bool:
    """
    Store SHAP report metadata in DuckDB for dashboard access.
    
    Args:
        report_metadata: Report metadata from SHAP generation
        algorithm: Algorithm type
        
    Returns:
        Success status
    """
    try:
        from mech_exo.datasource.storage import DataStorage
        import pandas as pd
        
        if 'error' in report_metadata:
            logger.error(f"Cannot store failed report: {report_metadata['error']}")
            return False
        
        # Create report record
        report_record = {
            'report_date': datetime.now().date(),
            'algorithm': algorithm,
            'model_path': report_metadata['model_path'],
            'png_path': report_metadata['png_path'],
            'html_path': report_metadata['html_path'],
            'samples_analyzed': report_metadata['samples_analyzed'],
            'features_count': report_metadata['features_count'],
            'png_size_mb': report_metadata['png_size_mb'],
            'html_size_mb': report_metadata['html_size_mb'],
            'top_features': json.dumps(report_metadata['top_features']),
            'created_at': datetime.now(),
            'status': 'completed'
        }
        
        # Store in database
        storage = DataStorage()
        
        # Create table if not exists
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS ml_reports (
            report_date DATE,
            algorithm VARCHAR,
            model_path VARCHAR,
            png_path VARCHAR,
            html_path VARCHAR,
            samples_analyzed INTEGER,
            features_count INTEGER,
            png_size_mb FLOAT,
            html_size_mb FLOAT,
            top_features VARCHAR,
            created_at TIMESTAMP,
            status VARCHAR,
            PRIMARY KEY (report_date, algorithm)
        )
        """
        
        storage.execute_query(create_table_sql)
        
        # Insert report record
        df = pd.DataFrame([report_record])
        storage.store_data('ml_reports', df, if_exists='replace')
        
        storage.close()
        
        logger.info(f"Stored {algorithm} report metadata in database")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store report metadata: {e}")
        return False


@task(name="create_report_artifacts", description="Create Prefect artifacts for SHAP reports")
def create_report_artifacts(reports_metadata: List[Dict]) -> bool:
    """
    Create Prefect artifacts for generated SHAP reports.
    
    Args:
        reports_metadata: List of report metadata dictionaries
        
    Returns:
        Success status
    """
    try:
        # Create summary markdown artifact
        markdown_content = "# ML SHAP Feature Importance Reports\n\n"
        markdown_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        successful_reports = [r for r in reports_metadata if 'error' not in r]
        failed_reports = [r for r in reports_metadata if 'error' in r]
        
        if successful_reports:
            markdown_content += "## Generated Reports\n\n"
            
            for report in successful_reports:
                algorithm = report['algorithm']
                samples = report['samples_analyzed']
                features = report['features_count']
                
                markdown_content += f"### {algorithm.upper()} Model Report\n\n"
                markdown_content += f"- **Samples Analyzed**: {samples:,}\n"
                markdown_content += f"- **Features**: {features}\n"
                markdown_content += f"- **PNG Report**: `{report['png_path']}` ({report['png_size_mb']:.2f} MB)\n"
                markdown_content += f"- **HTML Interactive**: `{report['html_path']}` ({report['html_size_mb']:.2f} MB)\n"
                
                if 'top_features' in report and report['top_features']:
                    markdown_content += f"- **Top 5 Features**: {', '.join(report['top_features'][:5])}\n"
                
                markdown_content += "\n"
        
        if failed_reports:
            markdown_content += "## Failed Reports\n\n"
            for report in failed_reports:
                markdown_content += f"- **Error**: {report['error']}\n"
            markdown_content += "\n"
        
        markdown_content += f"## Summary\n\n"
        markdown_content += f"- **Successful**: {len(successful_reports)}\n"
        markdown_content += f"- **Failed**: {len(failed_reports)}\n"
        markdown_content += f"- **Total**: {len(reports_metadata)}\n"
        
        # Create the artifact
        create_markdown_artifact(
            key="ml-shap-reports",
            markdown=markdown_content,
            description="ML SHAP feature importance analysis reports"
        )
        
        logger.info(f"Created Prefect artifact for {len(reports_metadata)} reports")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create report artifacts: {e}")
        return False


@flow(name="ml-shap-reports", description="Generate ML SHAP feature importance reports")
def ml_shap_reports_flow(models_dir: str = "models",
                        features_dir: str = "data/features", 
                        output_dir: str = "reports/shap",
                        lookback_days: int = 60) -> Dict[str, int]:
    """
    Main flow for generating ML SHAP reports.
    
    Args:
        models_dir: Directory containing trained models
        features_dir: Directory containing feature files
        output_dir: Output directory for reports
        lookback_days: Days of recent data to analyze
        
    Returns:
        Summary statistics
    """
    logger.info("🔍 Starting ML SHAP reports flow...")
    
    # Find latest trained models
    models = find_latest_models(models_dir)
    
    if not models:
        logger.warning("No trained models found - skipping SHAP analysis")
        return {'total': 0, 'successful': 0, 'failed': 0}
    
    # Load recent feature data
    features_file = load_recent_features(features_dir, lookback_days)
    
    if not features_file:
        logger.warning("No recent feature data found - skipping SHAP analysis")
        return {'total': 0, 'successful': 0, 'failed': 0}
    
    # Generate SHAP reports for each model
    reports_metadata = []
    
    for algorithm, model_path in models.items():
        logger.info(f"Processing {algorithm} model: {model_path}")
        
        # Generate SHAP report
        metadata = generate_shap_report(
            model_path=model_path,
            features_file=features_file,
            algorithm=algorithm,
            output_dir=output_dir
        )
        
        metadata['algorithm'] = algorithm
        reports_metadata.append(metadata)
        
        # Store metadata in database if successful
        if 'error' not in metadata:
            store_report_metadata(metadata, algorithm)
    
    # Create Prefect artifacts
    create_report_artifacts(reports_metadata)
    
    # Summary statistics
    successful = len([r for r in reports_metadata if 'error' not in r])
    failed = len([r for r in reports_metadata if 'error' in r])
    
    logger.info(f"✅ ML SHAP reports flow completed")
    logger.info(f"   Successful: {successful}")
    logger.info(f"   Failed: {failed}")
    logger.info(f"   Total: {len(reports_metadata)}")
    
    return {
        'total': len(reports_metadata),
        'successful': successful,
        'failed': failed
    }


# Optional: Schedule the flow to run daily
if __name__ == "__main__":
    # Run the flow manually for testing
    result = ml_shap_reports_flow()
    print(f"Flow result: {result}")