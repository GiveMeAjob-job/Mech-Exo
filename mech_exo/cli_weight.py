"""
CLI utility for ML weight adjustment testing and management
"""

import argparse
import sys
import logging
from typing import Optional

from .scoring.weight_utils import (
    compute_new_weight,
    get_current_ml_weight,
    auto_adjust_ml_weight,
    format_weight_change_summary,
    validate_weight_bounds
)

logger = logging.getLogger(__name__)


def cmd_weight_adjust(args):
    """Handle weight adjustment command"""
    try:
        # Validate current weight (only weights have 0.0-0.50 bounds, not Sharpe ratios)
        if args.current is not None:
            current_valid, current_err = validate_weight_bounds(args.current)
            if not current_valid:
                print(f"❌ Invalid current weight: {current_err}")
                return 1
        
        # Basic validation for Sharpe ratios (allow any reasonable range)
        if args.baseline is None or args.ml is None:
            print("❌ Both baseline and ML Sharpe ratios are required")
            return 1
        
        # Get current weight if not provided
        current_weight = args.current if args.current is not None else get_current_ml_weight()
        
        print(f"🔧 ML Weight Adjustment Analysis")
        print(f"📊 Baseline Sharpe: {args.baseline:.3f}")
        print(f"📊 ML Sharpe: {args.ml:.3f}")
        print(f"⚖️ Current weight: {current_weight:.3f}")
        print()
        
        # Compute new weight
        new_weight, rule = compute_new_weight(
            baseline_sharpe=args.baseline,
            ml_sharpe=args.ml,
            current_w=current_weight,
            step=args.step,
            upper=args.upper,
            lower=args.lower,
            up_thresh=args.up_thresh,
            down_thresh=args.down_thresh
        )
        
        # Calculate change
        weight_change = new_weight - current_weight
        change_direction = "↗️" if weight_change > 0 else "↘️" if weight_change < 0 else "➡️"
        
        # Display results
        print(f"📈 Performance delta: {args.ml - args.baseline:+.3f}")
        print(f"🎯 Adjustment rule: {rule}")
        print(f"⚖️ New weight: {new_weight:.3f} {change_direction} ({weight_change:+.3f})")
        print()
        
        if abs(weight_change) > 0.001:
            if args.dry_run:
                print("🔍 DRY RUN MODE - No changes will be made")
                print(f"✅ Would adjust weight: {current_weight:.3f} → {new_weight:.3f}")
            else:
                print("⚠️ This would update config/factors.yml")
                print("Use --dry-run to simulate without making changes")
        else:
            print("ℹ️ No weight change recommended")
        
        # Show notification preview
        if abs(weight_change) > 0.001:
            print("\n📱 Notification preview:")
            summary = format_weight_change_summary(
                current_weight, new_weight, args.ml, args.baseline, rule
            )
            for line in summary.split('\n'):
                print(f"   {line}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Weight adjustment failed: {e}")
        return 1


def cmd_auto_adjust(args):
    """Handle automatic weight adjustment"""
    try:
        print(f"🤖 Automatic ML Weight Adjustment")
        print(f"🔍 Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
        print()
        
        # Run auto adjustment
        result = auto_adjust_ml_weight(dry_run=args.dry_run)
        
        if result['success']:
            print(f"📊 Current weight: {result['current_weight']:.3f}")
            print(f"📊 New weight: {result['new_weight']:.3f}")
            print(f"🎯 Rule: {result['adjustment_rule']}")
            
            if 'baseline_sharpe' in result and 'ml_sharpe' in result:
                print(f"📈 ML Sharpe: {result['ml_sharpe']:.3f}")
                print(f"📈 Baseline Sharpe: {result['baseline_sharpe']:.3f}")
                print(f"📈 Delta: {result['sharpe_diff']:+.3f}")
            
            if result['changed']:
                if args.dry_run:
                    print("✅ DRY RUN: Weight would be adjusted")
                else:
                    print("✅ Weight adjusted and logged")
                    if result.get('config_updated'):
                        print("✅ Configuration file updated")
                    if result.get('change_logged'):
                        print("✅ Change logged to history")
            else:
                print("ℹ️ No weight adjustment needed")
            
        else:
            print(f"❌ Auto adjustment failed: {result.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Auto adjustment failed: {e}")
        return 1


def cmd_current_weight(args):
    """Display current ML weight"""
    try:
        current_weight = get_current_ml_weight()
        print(f"⚖️ Current ML weight: {current_weight:.3f}")
        
        # Show validation
        is_valid, error_msg = validate_weight_bounds(current_weight)
        if is_valid:
            print("✅ Weight within valid range (0.0 - 0.50)")
        else:
            print(f"⚠️ Weight validation: {error_msg}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to get current weight: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ML Weight Adjustment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test weight adjustment with specific Sharpe ratios
  python -m mech_exo.cli_weight adjust --baseline 1.20 --ml 1.35 --current 0.30 --dry-run
  
  # Auto-adjust using live metrics (dry run)
  python -m mech_exo.cli_weight auto --dry-run
  
  # Check current weight
  python -m mech_exo.cli_weight current
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Manual weight adjustment command
    adjust_parser = subparsers.add_parser('adjust', help='Test weight adjustment algorithm')
    adjust_parser.add_argument('--baseline', type=float, required=True,
                              help='Baseline strategy Sharpe ratio')
    adjust_parser.add_argument('--ml', type=float, required=True,
                              help='ML strategy Sharpe ratio')
    adjust_parser.add_argument('--current', type=float,
                              help='Current ML weight (default: read from config)')
    adjust_parser.add_argument('--step', type=float, default=0.05,
                              help='Adjustment step size (default: 0.05)')
    adjust_parser.add_argument('--upper', type=float, default=0.50,
                              help='Maximum weight (default: 0.50)')
    adjust_parser.add_argument('--lower', type=float, default=0.0,
                              help='Minimum weight (default: 0.0)')
    adjust_parser.add_argument('--up-thresh', type=float, default=0.10,
                              help='Upward adjustment threshold (default: 0.10)')
    adjust_parser.add_argument('--down-thresh', type=float, default=-0.05,
                              help='Downward adjustment threshold (default: -0.05)')
    adjust_parser.add_argument('--dry-run', action='store_true',
                              help='Simulate without making changes')
    adjust_parser.set_defaults(func=cmd_weight_adjust)
    
    # Auto adjustment command
    auto_parser = subparsers.add_parser('auto', help='Auto-adjust weight using live metrics')
    auto_parser.add_argument('--dry-run', action='store_true',
                            help='Simulate without making changes')
    auto_parser.set_defaults(func=cmd_auto_adjust)
    
    # Current weight command
    current_parser = subparsers.add_parser('current', help='Show current ML weight')
    current_parser.set_defaults(func=cmd_current_weight)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())