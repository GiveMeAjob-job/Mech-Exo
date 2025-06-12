#!/usr/bin/env python3
"""
Live Data Sanity Check and Threshold Review

Validates current canary configuration and reviews thresholds against
live data to ensure reasonable auto-disable behavior.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime, date, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mech_exo.execution.allocation import get_allocation_config, check_hysteresis_trigger
from mech_exo.reporting.query import get_ab_test_summary
from mech_exo.datasource.storage import DataStorage


def validate_allocation_config() -> Dict:
    """Validate current allocation configuration"""
    print("📋 Validating Allocation Configuration")
    print("=" * 50)
    
    config = get_allocation_config()
    
    validation_results = {
        'config_valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    # Check canary allocation percentage
    allocation_pct = config.get('canary_allocation', 0.1)
    print(f"Canary allocation: {allocation_pct:.1%}")
    
    if allocation_pct > 0.25:
        validation_results['warnings'].append(f"High canary allocation ({allocation_pct:.1%}) - consider reducing to < 25%")
    elif allocation_pct < 0.05:
        validation_results['warnings'].append(f"Low canary allocation ({allocation_pct:.1%}) - may need longer time for statistical significance")
    
    # Check disable rule configuration
    disable_rule = config.get('disable_rule', {})
    
    sharpe_threshold = disable_rule.get('sharpe_low', 0.0)
    confirm_days = disable_rule.get('confirm_days', 2)
    max_dd_pct = disable_rule.get('max_dd_pct', 2.0)
    min_observations = disable_rule.get('min_observations', 21)
    
    print(f"Sharpe threshold: {sharpe_threshold}")
    print(f"Confirm days (hysteresis): {confirm_days}")
    print(f"Max drawdown threshold: {max_dd_pct:.1%}")
    print(f"Min observations: {min_observations}")
    
    # Validate thresholds
    if sharpe_threshold < -0.5:
        validation_results['warnings'].append(f"Very low Sharpe threshold ({sharpe_threshold}) - may be too aggressive")
    elif sharpe_threshold > 0.5:
        validation_results['warnings'].append(f"High Sharpe threshold ({sharpe_threshold}) - may be too conservative")
    
    if confirm_days < 2:
        validation_results['errors'].append("Hysteresis confirm_days should be >= 2 for protection against false positives")
    elif confirm_days > 5:
        validation_results['warnings'].append(f"High confirm_days ({confirm_days}) - may delay necessary auto-disable")
    
    if min_observations < 15:
        validation_results['warnings'].append(f"Low min_observations ({min_observations}) - may not provide reliable Sharpe calculation")
    elif min_observations > 60:
        validation_results['warnings'].append(f"High min_observations ({min_observations}) - may delay auto-disable too long")
    
    # Check if config has errors
    if 'error' in config:
        validation_results['errors'].append(f"Config loading error: {config['error']}")
        validation_results['config_valid'] = False
    
    return validation_results


def check_historical_performance(days_back: int = 60) -> Dict:
    """Check historical canary performance to validate thresholds"""
    print(f"\n📈 Historical Performance Analysis ({days_back} days)")
    print("=" * 50)
    
    try:
        # Get recent A/B test summary
        summary = get_ab_test_summary(days=days_back)
        
        performance_analysis = {
            'data_available': summary['days_analyzed'] > 0,
            'avg_sharpe_diff': summary.get('sharpe_diff', 0.0),
            'status': summary.get('status_badge', 'UNKNOWN'),
            'days_analyzed': summary['days_analyzed'],
            'would_trigger_scenarios': []
        }
        
        print(f"Days with data: {summary['days_analyzed']}")
        print(f"Current status: {summary['status_badge']}")
        print(f"Average Sharpe difference: {summary['sharpe_diff']:+.3f}")
        print(f"Canary NAV: ${summary.get('canary_nav', 0):,.0f}")
        print(f"Base NAV: ${summary.get('base_nav', 0):,.0f}")
        
        # Get detailed performance data from database
        storage = DataStorage()
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        query = """
            SELECT date, canary_sharpe_30d, base_sharpe_30d, sharpe_diff, canary_enabled
            FROM canary_performance 
            WHERE date >= ? AND date <= ?
            ORDER BY date DESC
            LIMIT 30
        """
        
        result = storage.conn.execute(query, [str(start_date), str(end_date)]).fetchall()
        storage.close()
        
        if result:
            print(f"\nRecent performance (last {len(result)} days):")
            
            # Analyze threshold scenarios
            config = get_allocation_config()
            disable_rule = config.get('disable_rule', {})
            sharpe_threshold = disable_rule.get('sharpe_low', 0.0)
            
            breach_count = 0
            consecutive_breaches = 0
            max_consecutive = 0
            
            for row in result:
                date_str, canary_sharpe, base_sharpe, sharpe_diff, enabled = row
                
                if canary_sharpe is not None and canary_sharpe < sharpe_threshold:
                    breach_count += 1
                    consecutive_breaches += 1
                    max_consecutive = max(max_consecutive, consecutive_breaches)
                    
                    print(f"  {date_str}: Canary {canary_sharpe:.3f} < {sharpe_threshold:.3f} ⚠️")
                else:
                    consecutive_breaches = 0
                    if canary_sharpe is not None:
                        print(f"  {date_str}: Canary {canary_sharpe:.3f} >= {sharpe_threshold:.3f} ✅")
            
            performance_analysis['breach_days'] = breach_count
            performance_analysis['max_consecutive_breaches'] = max_consecutive
            
            print(f"\nThreshold analysis:")
            print(f"  Total breach days: {breach_count}/{len(result)}")
            print(f"  Max consecutive breaches: {max_consecutive}")
            print(f"  Would trigger auto-disable: {'Yes' if max_consecutive >= disable_rule.get('confirm_days', 2) else 'No'}")
            
        else:
            print("⚠️ No historical performance data available")
            performance_analysis['data_available'] = False
        
        return performance_analysis
        
    except Exception as e:
        print(f"❌ Error analyzing historical performance: {e}")
        return {
            'data_available': False,
            'error': str(e)
        }


def test_hysteresis_scenarios() -> Dict:
    """Test various Sharpe scenarios against hysteresis logic"""
    print(f"\n🧪 Hysteresis Logic Testing")
    print("=" * 50)
    
    test_scenarios = [
        {'sharpe': 0.1, 'description': 'Good performance'},
        {'sharpe': 0.05, 'description': 'Marginal performance'},
        {'sharpe': -0.05, 'description': 'Slight underperformance'},
        {'sharpe': -0.15, 'description': 'Poor performance'},
        {'sharpe': -0.25, 'description': 'Very poor performance'}
    ]
    
    scenario_results = []
    
    for scenario in test_scenarios:
        sharpe = scenario['sharpe']
        description = scenario['description']
        
        print(f"\nTesting: {description} (Sharpe: {sharpe:+.3f})")
        
        # Reset breach counter before testing
        from mech_exo.execution.allocation import reset_breach_counter
        reset_breach_counter()
        
        # Simulate multiple days
        results = []
        for day in range(1, 4):
            hysteresis_result = check_hysteresis_trigger(sharpe)
            results.append({
                'day': day,
                'breach_days': hysteresis_result['current_breach_days'],
                'should_trigger': hysteresis_result['should_trigger']
            })
            
            print(f"  Day {day}: Breach days {hysteresis_result['current_breach_days']}, "
                  f"Trigger: {hysteresis_result['should_trigger']}")
        
        scenario_results.append({
            'scenario': scenario,
            'results': results
        })
    
    # Reset breach counter after testing
    from mech_exo.execution.allocation import reset_breach_counter
    reset_breach_counter()
    
    return {'scenarios': scenario_results}


def check_data_quality() -> Dict:
    """Check data quality and availability"""
    print(f"\n📊 Data Quality Check")
    print("=" * 50)
    
    try:
        storage = DataStorage()
        
        # Check canary_performance table
        perf_count = storage.conn.execute("SELECT COUNT(*) FROM canary_performance").fetchone()[0]
        print(f"Performance records: {perf_count}")
        
        if perf_count > 0:
            # Get date range
            date_range = storage.conn.execute(
                "SELECT MIN(date), MAX(date) FROM canary_performance"
            ).fetchone()
            
            print(f"Date range: {date_range[0]} to {date_range[1]}")
            
            # Check recent data quality
            recent_data = storage.conn.execute("""
                SELECT date, canary_sharpe_30d, base_sharpe_30d, days_in_window
                FROM canary_performance 
                WHERE date >= DATE('now', '-30 days')
                ORDER BY date DESC
                LIMIT 10
            """).fetchall()
            
            print(f"Recent data quality (last 10 days):")
            for row in recent_data:
                date_str, canary_sharpe, base_sharpe, days_in_window = row
                quality = "good" if days_in_window >= 21 else "fair" if days_in_window >= 15 else "poor"
                print(f"  {date_str}: {days_in_window} days window, quality: {quality}")
        
        # Check fills data
        fills_count = storage.conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        print(f"Total fill records: {fills_count}")
        
        if fills_count > 0:
            # Check tag distribution
            tag_dist = storage.conn.execute("""
                SELECT tag, COUNT(*) as count, SUM(ABS(gross_value)) as total_value
                FROM fills 
                WHERE DATE(filled_at) >= DATE('now', '-30 days')
                GROUP BY tag
            """).fetchall()
            
            print(f"Recent fill distribution (30 days):")
            for row in tag_dist:
                tag, count, total_value = row
                print(f"  {tag}: {count} fills, ${total_value:,.0f} notional")
        
        storage.close()
        
        return {
            'performance_records': perf_count,
            'fills_records': fills_count,
            'data_quality': 'good' if perf_count > 30 else 'limited'
        }
        
    except Exception as e:
        print(f"❌ Error checking data quality: {e}")
        return {
            'error': str(e),
            'data_quality': 'error'
        }


def generate_recommendations(validation_results: Dict, performance_analysis: Dict, data_quality: Dict) -> List[str]:
    """Generate configuration recommendations based on analysis"""
    recommendations = []
    
    # Configuration recommendations
    if validation_results['config_valid']:
        if len(validation_results['warnings']) == 0:
            recommendations.append("✅ Current configuration appears well-balanced")
        
        for warning in validation_results['warnings']:
            recommendations.append(f"⚠️ Configuration: {warning}")
    
    # Performance-based recommendations
    if performance_analysis.get('data_available', False):
        max_consecutive = performance_analysis.get('max_consecutive_breaches', 0)
        
        if max_consecutive == 0:
            recommendations.append("✅ No threshold breaches detected in recent history")
        elif max_consecutive == 1:
            recommendations.append("👍 Single-day breaches detected - hysteresis protection working")
        elif max_consecutive >= 2:
            recommendations.append(f"⚠️ {max_consecutive} consecutive breaches detected - auto-disable would have triggered")
    
    # Data quality recommendations
    data_qual = data_quality.get('data_quality', 'unknown')
    if data_qual == 'limited':
        recommendations.append("📅 Limited historical data - consider running longer before adjusting thresholds")
    elif data_qual == 'error':
        recommendations.append("❌ Data quality check failed - verify database connectivity")
    
    return recommendations


def main():
    """Run complete canary threshold review"""
    print("🧪 Canary A/B Testing - Live Data Sanity Check & Threshold Review")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Validate configuration
    validation_results = validate_allocation_config()
    
    # 2. Check historical performance
    performance_analysis = check_historical_performance(days_back=60)
    
    # 3. Test hysteresis scenarios
    hysteresis_results = test_hysteresis_scenarios()
    
    # 4. Check data quality
    data_quality = check_data_quality()
    
    # 5. Generate recommendations
    recommendations = generate_recommendations(validation_results, performance_analysis, data_quality)
    
    # Summary
    print(f"\n📋 Summary & Recommendations")
    print("=" * 50)
    
    for rec in recommendations:
        print(f"  {rec}")
    
    # Overall health score
    config_score = 1.0 if validation_results['config_valid'] and len(validation_results['errors']) == 0 else 0.5
    data_score = 1.0 if data_quality.get('data_quality') == 'good' else 0.5
    perf_score = 1.0 if performance_analysis.get('data_available') else 0.0
    
    overall_score = (config_score + data_score + perf_score) / 3.0
    
    print(f"\n🎯 Overall Health Score: {overall_score:.1%}")
    
    if overall_score >= 0.8:
        print("✅ Canary A/B testing system is healthy and ready for live trading")
    elif overall_score >= 0.6:
        print("⚠️ Canary A/B testing system is mostly ready - address warnings above")
    else:
        print("❌ Canary A/B testing system needs attention before live trading")
    
    return overall_score >= 0.6


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)