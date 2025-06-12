"""
Dashboard hooks for ML SHAP reports integration.

Provides data access functions for displaying ML feature importance
and SHAP analysis results in the main dashboard.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class MLReportsDashboardHook:
    """
    Dashboard hook for ML SHAP reports and feature importance data.
    
    Provides standardized interface for dashboard components to access
    ML model interpretability results.
    """
    
    def __init__(self, storage=None):
        """
        Initialize dashboard hook.
        
        Args:
            storage: Optional DataStorage instance. Created if None.
        """
        self.storage = storage
        if self.storage is None:
            try:
                from mech_exo.datasource.storage import DataStorage
                self.storage = DataStorage()
            except Exception as e:
                logger.warning(f"Could not initialize DataStorage: {e}")
                self.storage = None
    
    def get_latest_shap_reports(self, days_back: int = 7) -> pd.DataFrame:
        """
        Get latest SHAP reports from the database.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            DataFrame with SHAP report metadata
        """
        if self.storage is None:
            logger.warning("No storage available - returning empty DataFrame")
            return pd.DataFrame()
        
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days_back)
            
            query = """
            SELECT * FROM ml_reports 
            WHERE report_date >= ? 
            ORDER BY report_date DESC, algorithm
            """
            
            result = self.storage.execute_query(query, params=[cutoff_date])
            
            if result.empty:
                logger.info("No recent SHAP reports found")
                return pd.DataFrame()
            
            # Parse top_features JSON
            if 'top_features' in result.columns:
                result['top_features_list'] = result['top_features'].apply(
                    lambda x: json.loads(x) if x else []
                )
            
            logger.info(f"Retrieved {len(result)} SHAP reports")
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve SHAP reports: {e}")
            return pd.DataFrame()
    
    def get_shap_report_summary(self) -> Dict:
        """
        Get summary statistics for SHAP reports.
        
        Returns:
            Dictionary with summary statistics
        """
        reports_df = self.get_latest_shap_reports(days_back=30)
        
        if reports_df.empty:
            return {
                'total_reports': 0,
                'algorithms': [],
                'last_generated': None,
                'avg_features': 0,
                'avg_samples': 0
            }
        
        summary = {
            'total_reports': len(reports_df),
            'algorithms': reports_df['algorithm'].unique().tolist(),
            'last_generated': reports_df['report_date'].max(),
            'avg_features': int(reports_df['features_count'].mean()) if 'features_count' in reports_df.columns else 0,
            'avg_samples': int(reports_df['samples_analyzed'].mean()) if 'samples_analyzed' in reports_df.columns else 0
        }
        
        # Add per-algorithm stats
        algo_stats = {}
        for algo in summary['algorithms']:
            algo_data = reports_df[reports_df['algorithm'] == algo]
            algo_stats[algo] = {
                'count': len(algo_data),
                'last_generated': algo_data['report_date'].max(),
                'avg_features': int(algo_data['features_count'].mean()) if 'features_count' in algo_data.columns else 0
            }
        
        summary['by_algorithm'] = algo_stats
        
        return summary
    
    def get_top_features_across_models(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get most important features across all recent models.
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance aggregated across models
        """
        reports_df = self.get_latest_shap_reports(days_back=7)
        
        if reports_df.empty or 'top_features_list' not in reports_df.columns:
            return pd.DataFrame()
        
        # Aggregate feature mentions across models
        feature_counts = {}
        feature_positions = {}
        
        for _, report in reports_df.iterrows():
            features = report['top_features_list']
            algorithm = report['algorithm']
            
            for i, feature in enumerate(features[:top_k]):
                if feature not in feature_counts:
                    feature_counts[feature] = 0
                    feature_positions[feature] = []
                
                feature_counts[feature] += 1
                feature_positions[feature].append({
                    'algorithm': algorithm,
                    'position': i + 1,
                    'date': report['report_date']
                })
        
        # Create summary DataFrame
        feature_data = []
        for feature, count in feature_counts.items():
            positions = feature_positions[feature]
            avg_position = sum(p['position'] for p in positions) / len(positions)
            algorithms = list(set(p['algorithm'] for p in positions))
            
            feature_data.append({
                'feature': feature,
                'mention_count': count,
                'avg_position': avg_position,
                'algorithms': ', '.join(algorithms),
                'importance_score': count / avg_position  # Higher is better
            })
        
        if not feature_data:
            return pd.DataFrame()
        
        # Sort by importance score
        result_df = pd.DataFrame(feature_data)
        result_df = result_df.sort_values('importance_score', ascending=False)
        result_df = result_df.reset_index(drop=True)
        result_df['rank'] = range(1, len(result_df) + 1)
        
        return result_df.head(top_k)
    
    def get_shap_file_paths(self, algorithm: str = None, 
                          report_date: str = None) -> Dict[str, str]:
        """
        Get file paths for SHAP reports.
        
        Args:
            algorithm: Filter by algorithm (optional)
            report_date: Filter by specific date (YYYY-MM-DD, optional)
            
        Returns:
            Dictionary with PNG and HTML file paths
        """
        reports_df = self.get_latest_shap_reports(days_back=30)
        
        if reports_df.empty:
            return {}
        
        # Apply filters
        if algorithm:
            reports_df = reports_df[reports_df['algorithm'] == algorithm]
        
        if report_date:
            target_date = pd.to_datetime(report_date).date()
            reports_df = reports_df[reports_df['report_date'] == target_date]
        
        if reports_df.empty:
            return {}
        
        # Get most recent report
        latest_report = reports_df.iloc[0]
        
        file_paths = {}
        
        # Check if files exist
        if 'png_path' in latest_report and latest_report['png_path']:
            png_path = Path(latest_report['png_path'])
            if png_path.exists():
                file_paths['png'] = str(png_path)
        
        if 'html_path' in latest_report and latest_report['html_path']:
            html_path = Path(latest_report['html_path'])
            if html_path.exists():
                file_paths['html'] = str(html_path)
        
        return file_paths
    
    def get_model_performance_comparison(self) -> pd.DataFrame:
        """
        Compare performance metrics across different ML models.
        
        Returns:
            DataFrame with model performance comparison
        """
        # This would integrate with model metrics stored during training
        # For now, return structure showing what this would look like
        
        try:
            # Look for recent model metrics files
            models_dir = Path("models")
            if not models_dir.exists():
                return pd.DataFrame()
            
            metrics_files = list(models_dir.glob("metrics_*.json"))
            
            if not metrics_files:
                return pd.DataFrame()
            
            # Load recent metrics
            comparison_data = []
            
            for metrics_file in sorted(metrics_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    comparison_data.append({
                        'algorithm': metrics.get('algorithm', 'unknown'),
                        'timestamp': metrics.get('timestamp', ''),
                        'best_auc': metrics.get('best_auc', 0),
                        'mean_auc': metrics.get('metrics', {}).get('mean_auc', 0),
                        'mean_ic': metrics.get('metrics', {}).get('mean_ic', 0),
                        'training_samples': metrics.get('training_samples', 0),
                        'features': metrics.get('features', 0),
                        'cv_folds': metrics.get('cv_folds', 0)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to load metrics from {metrics_file}: {e}")
            
            if comparison_data:
                return pd.DataFrame(comparison_data)
            
        except Exception as e:
            logger.error(f"Failed to load model performance data: {e}")
        
        return pd.DataFrame()
    
    def close(self):
        """Close database connection."""
        if self.storage:
            self.storage.close()


def get_ml_dashboard_data() -> Dict:
    """
    Convenience function to get all ML dashboard data.
    
    Returns:
        Dictionary with all ML dashboard components
    """
    hook = MLReportsDashboardHook()
    
    try:
        data = {
            'shap_summary': hook.get_shap_report_summary(),
            'top_features': hook.get_top_features_across_models(top_k=10),
            'recent_reports': hook.get_latest_shap_reports(days_back=7),
            'model_comparison': hook.get_model_performance_comparison()
        }
        
        # Convert DataFrames to dictionaries for JSON serialization
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                data[key] = value.to_dict('records') if not value.empty else []
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to get ML dashboard data: {e}")
        return {
            'shap_summary': {},
            'top_features': [],
            'recent_reports': [],
            'model_comparison': []
        }
    
    finally:
        hook.close()


def get_shap_report_for_display(algorithm: str = None) -> Dict:
    """
    Get SHAP report data formatted for dashboard display.
    
    Args:
        algorithm: Algorithm to get report for (optional)
        
    Returns:
        Dictionary with report data and file paths
    """
    hook = MLReportsDashboardHook()
    
    try:
        # Get file paths
        file_paths = hook.get_shap_file_paths(algorithm=algorithm)
        
        # Get summary data
        summary = hook.get_shap_report_summary()
        
        # Get top features
        top_features = hook.get_top_features_across_models(top_k=20)
        
        return {
            'file_paths': file_paths,
            'summary': summary,
            'top_features': top_features.to_dict('records') if not top_features.empty else [],
            'algorithm': algorithm,
            'available': len(file_paths) > 0
        }
        
    except Exception as e:
        logger.error(f"Failed to get SHAP report for display: {e}")
        return {
            'file_paths': {},
            'summary': {},
            'top_features': [],
            'algorithm': algorithm,
            'available': False,
            'error': str(e)
        }
    
    finally:
        hook.close()


if __name__ == "__main__":
    # Test the dashboard hooks
    print("🧪 Testing ML dashboard hooks...")
    
    hook = MLReportsDashboardHook()
    
    # Test summary
    summary = hook.get_shap_report_summary()
    print(f"SHAP summary: {summary}")
    
    # Test top features
    top_features = hook.get_top_features_across_models()
    print(f"Top features shape: {top_features.shape if not top_features.empty else 'empty'}")
    
    # Test file paths
    file_paths = hook.get_shap_file_paths()
    print(f"File paths: {file_paths}")
    
    hook.close()
    print("✅ Dashboard hooks tested")