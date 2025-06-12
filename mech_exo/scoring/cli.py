"""
CLI interface for idea scoring and ranking.

Provides command-line access to the IdeaScorer with ML integration support.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .scorer import IdeaScorer


def score_cli(symbols: Optional[List[str]] = None,
             use_ml: bool = False,
             ml_scores_file: Optional[str] = None,
             output_file: str = "idea_scores.csv",
             config_path: str = "config/factors.yml",
             top_n: Optional[int] = None,
             verbose: bool = False) -> dict:
    """
    CLI function for idea scoring.
    
    Args:
        symbols: List of symbols to score (None = all universe)
        use_ml: Whether to integrate ML scores
        ml_scores_file: Path to ML scores CSV file
        output_file: Output CSV file path
        config_path: Path to factors configuration
        top_n: Return only top N results
        verbose: Enable verbose logging
        
    Returns:
        Scoring metadata
    """
    # Set up logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Starting idea scoring...")
    logger.info(f"   Use ML: {use_ml}")
    logger.info(f"   Config: {config_path}")
    logger.info(f"   Output: {output_file}")
    
    if symbols:
        logger.info(f"   Symbols: {len(symbols)} specified ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    else:
        logger.info("   Symbols: Full universe")
    
    if use_ml and ml_scores_file:
        logger.info(f"   ML scores file: {ml_scores_file}")
    
    try:
        # Initialize scorer
        scorer = IdeaScorer(config_path=config_path, use_ml=use_ml)
        
        # Generate scores
        if symbols:
            results = scorer.score(symbols, ml_scores_file=ml_scores_file)
        else:
            results = scorer.rank_universe()
        
        if results.empty:
            logger.warning("No scores generated")
            return {
                'success': False,
                'message': 'No scores generated',
                'results': 0
            }
        
        # Limit to top N if specified
        if top_n and top_n > 0:
            results = results.head(top_n)
            logger.info(f"Limited to top {top_n} results")
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results.to_csv(output_path, index=False)
        
        # Create metadata
        metadata = {
            'success': True,
            'timestamp': pd.Timestamp.now().isoformat(),
            'results': len(results),
            'use_ml': use_ml,
            'output_file': str(output_path),
            'file_size': output_path.stat().st_size,
            'top_symbols': results.head(5)['symbol'].tolist(),
            'top_scores': results.head(5)['composite_score'].tolist(),
            'config_path': config_path
        }
        
        # Add ML-specific metadata
        if use_ml and 'ml_rank' in results.columns:
            metadata.update({
                'ml_weight_used': results['ml_weight_used'].iloc[0] if 'ml_weight_used' in results.columns else None,
                'ml_scores_available': True
            })
        elif use_ml:
            metadata['ml_scores_available'] = False
        
        logger.info(f"✅ Idea scoring completed successfully!")
        logger.info(f"   Results: {metadata['results']:,}")
        logger.info(f"   ML integration: {metadata['use_ml']}")
        logger.info(f"   Output: {metadata['output_file']} ({metadata['file_size']} bytes)")
        
        logger.info(f"\n🏆 Top 5 Ideas:")
        for i, (symbol, score) in enumerate(zip(metadata['top_symbols'], metadata['top_scores']), 1):
            logger.info(f"   {i}. {symbol}: {score:.2f}")
        
        if use_ml and metadata.get('ml_scores_available'):
            logger.info(f"   ML weight: {metadata.get('ml_weight_used', 'N/A')}")
        
        scorer.close()
        
        return metadata
        
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return {
            'success': False,
            'message': str(e),
            'results': 0
        }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Idea Scoring and Ranking")
    
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                       help="Symbols to score (default: all universe)")
    parser.add_argument("--use-ml", action="store_true", default=False,
                       help="Integrate ML scores into ranking")
    parser.add_argument("--ml-scores", type=str, default=None,
                       help="Path to ML scores CSV file")
    parser.add_argument("--output", type=str, default="idea_scores.csv",
                       help="Output CSV file (default: idea_scores.csv)")
    parser.add_argument("--config", type=str, default="config/factors.yml",
                       help="Factors configuration file")
    parser.add_argument("--top", type=int, default=None,
                       help="Return only top N results")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Run scoring
    result = score_cli(
        symbols=args.symbols,
        use_ml=args.use_ml,
        ml_scores_file=args.ml_scores,
        output_file=args.output,
        config_path=args.config,
        top_n=args.top,
        verbose=args.verbose
    )
    
    if result['success']:
        print(f"\n🎯 Scoring completed! Results saved to {result['output_file']}")
        
        if args.use_ml and result.get('ml_scores_available'):
            print(f"📊 ML integration enabled with weight {result.get('ml_weight_used', 'N/A')}")
        elif args.use_ml:
            print("⚠️  ML requested but scores not available - using traditional scoring")
        
        sys.exit(0)
    else:
        print(f"❌ Scoring failed: {result['message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()