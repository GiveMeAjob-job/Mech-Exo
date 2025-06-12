"""
Unit tests for Optuna Monitor dashboard components

Tests the query functions, data processing, and dashboard callback logic
for the Optuna optimization monitoring interface.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import plotly.graph_objects as go


def test_get_optuna_results_empty():
    """Test get_optuna_results with empty database"""
    
    with patch('mech_exo.datasource.storage.DataStorage') as mock_storage:
        # Mock empty result
        mock_conn = Mock()
        mock_storage.return_value.conn = mock_conn
        
        # Mock empty DataFrame
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            
            from mech_exo.reporting.query import get_optuna_results
            
            result = get_optuna_results()
            
            # Verify empty DataFrame with correct columns
            assert isinstance(result, pd.DataFrame)
            assert result.empty
            expected_columns = [
                'trial_id', 'trial_number', 'study_name', 'sharpe_ratio', 
                'max_drawdown', 'constraint_violations', 'constraints_satisfied',
                'calculation_date', 'elapsed_time_seconds', 'data_points',
                'sampler', 'pruner', 'status', 'trial_duration_min', 'constraint_status'
            ]
            for col in expected_columns:
                assert col in result.columns


def test_get_optuna_results_with_data():
    """Test get_optuna_results with sample trial data"""
    
    # Create sample trial data
    sample_data = {
        'trial_id': ['trial_1', 'trial_2', 'trial_3'],
        'trial_number': [1, 2, 3],
        'study_name': ['test_study', 'test_study', 'test_study'],
        'sharpe_ratio': [0.5, 0.8, 1.2],
        'max_drawdown': [0.1, 0.08, 0.05],
        'constraint_violations': [1, 0, 0],
        'constraints_satisfied': [False, True, True],
        'calculation_date': [date.today()] * 3,
        'elapsed_time_seconds': [120.0, 95.0, 110.0],
        'data_points': [1000, 1000, 1000],
        'sampler': ['TPESampler', 'TPESampler', 'TPESampler'],
        'pruner': ['MedianPruner', 'MedianPruner', 'MedianPruner'],
        'status': ['COMPLETE', 'COMPLETE', 'COMPLETE']
    }
    
    with patch('mech_exo.datasource.storage.DataStorage') as mock_storage:
        mock_conn = Mock()
        mock_storage.return_value.conn = mock_conn
        
        # Mock DataFrame with sample data
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame(sample_data)
            
            from mech_exo.reporting.query import get_optuna_results
            
            result = get_optuna_results(limit=10)
            
            # Verify data processing
            assert len(result) == 3
            assert 'trial_duration_min' in result.columns
            assert 'constraint_status' in result.columns
            
            # Check derived columns
            assert result['trial_duration_min'].iloc[0] == 2.0  # 120 seconds / 60
            assert result['constraint_status'].iloc[0] == 'Violated'
            assert result['constraint_status'].iloc[1] == 'Satisfied'
            
            # Verify sorting by trial number
            assert result['trial_number'].tolist() == [1, 2, 3]


def test_optuna_data_update_callback():
    """Test Optuna data update callback function"""
    
    # Mock the query function
    sample_trials = pd.DataFrame({
        'trial_number': [1, 2, 3],
        'study_name': ['test_study', 'test_study', 'other_study'],
        'sharpe_ratio': [0.5, 0.8, 1.2],
        'calculation_date': [datetime.now().date()] * 3,
        'constraints_satisfied': [True, True, False]
    })
    
    with patch('mech_exo.reporting.dash_layout.optuna_monitor.get_optuna_results') as mock_query:
        mock_query.return_value = sample_trials
        
        from mech_exo.reporting.dash_layout.optuna_monitor import update_optuna_data
        
        # Test with specific study selection
        data_dict, status_text, status_class, last_updated = update_optuna_data(
            n_intervals=1, 
            selected_study='test_study'
        )
        
        # Verify data filtering
        assert len(data_dict['trials']) == 2  # Only test_study trials
        assert 'test_study' in data_dict['studies']
        assert 'other_study' in data_dict['studies']
        
        # Verify status
        assert '2 trials' in status_text
        assert 'bg-success' in status_class  # Recent activity
        assert 'Updated:' in last_updated


def test_optuna_data_update_callback_no_data():
    """Test Optuna data update callback with no data"""
    
    with patch('mech_exo.reporting.dash_layout.optuna_monitor.get_optuna_results') as mock_query:
        mock_query.return_value = pd.DataFrame()  # Empty DataFrame
        
        from mech_exo.reporting.dash_layout.optuna_monitor import update_optuna_data
        
        data_dict, status_text, status_class, last_updated = update_optuna_data(
            n_intervals=1, 
            selected_study='any_study'
        )
        
        # Verify empty data handling
        assert data_dict['trials'] == []
        assert data_dict['studies'] == []
        assert data_dict['error'] == 'No data'
        assert status_text == 'No Data'
        assert 'bg-warning' in status_class


def test_sharpe_chart_creation():
    """Test Sharpe ratio chart creation"""
    
    from mech_exo.reporting.dash_layout.optuna_monitor import update_sharpe_chart
    
    # Test with empty data
    empty_data = {'trials': [], 'error': 'No data'}
    fig = update_sharpe_chart(empty_data)
    
    assert isinstance(fig, go.Figure)
    assert 'No optimization data available' in str(fig.to_dict())
    
    # Test with sample data
    sample_data = {
        'trials': [
            {
                'trial_number': 1,
                'sharpe_ratio': 0.5,
                'max_drawdown': 0.1,
                'constraint_status': 'Violated'
            },
            {
                'trial_number': 2,
                'sharpe_ratio': 0.8,
                'max_drawdown': 0.08,
                'constraint_status': 'Satisfied'
            },
            {
                'trial_number': 3,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.05,
                'constraint_status': 'Satisfied'
            }
        ],
        'error': None
    }
    
    fig = update_sharpe_chart(sample_data)
    
    assert isinstance(fig, go.Figure)
    # Check that data traces are created
    fig_dict = fig.to_dict()
    assert len(fig_dict['data']) >= 1  # At least one trace for the line chart
    assert 'Sharpe Ratio Optimization Progress' in fig_dict['layout']['title']['text']


def test_importance_chart_creation():
    """Test parameter importance chart creation"""
    
    from mech_exo.reporting.dash_layout.optuna_monitor import update_importance_chart
    
    # Test with mock data (importance chart uses mock data in MVP)
    sample_data = {
        'trials': [{'trial_number': 1, 'sharpe_ratio': 0.5}],
        'error': None
    }
    
    fig = update_importance_chart(sample_data, 'test_study')
    
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    
    # Should have horizontal bar chart
    assert len(fig_dict['data']) == 1
    assert fig_dict['data'][0]['type'] == 'bar'
    assert fig_dict['data'][0]['orientation'] == 'h'
    assert 'Parameter Importance' in fig_dict['layout']['title']['text']


def test_summary_and_trials_update():
    """Test summary statistics and trials table update"""
    
    from mech_exo.reporting.dash_layout.optuna_monitor import update_summary_and_trials
    
    # Test with empty data
    empty_data = {'trials': [], 'error': 'No data'}
    summary, trials = update_summary_and_trials(empty_data)
    
    # Should return components indicating no data
    assert 'No optimization data available' in str(summary)
    assert 'No trial data available' in str(trials)
    
    # Test with sample data
    sample_data = {
        'trials': [
            {
                'trial_number': 1,
                'sharpe_ratio': 0.5,
                'max_drawdown': 0.1,
                'constraints_satisfied': False,
                'constraint_status': 'Violated',
                'trial_duration_min': 2.0
            },
            {
                'trial_number': 2,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.05,
                'constraints_satisfied': True,
                'constraint_status': 'Satisfied',
                'trial_duration_min': 1.8
            }
        ],
        'error': None
    }
    
    summary, trials = update_summary_and_trials(sample_data)
    
    # Verify summary component has statistics
    summary_str = str(summary)
    assert '2' in summary_str  # Total trials
    assert '1.2' in summary_str or '1.20' in summary_str  # Best Sharpe
    
    # Verify trials table component
    trials_str = str(trials)
    assert 'Trial' in trials_str
    assert 'Sharpe' in trials_str
    assert 'Constraints' in trials_str


def test_dashboard_layout_creation():
    """Test that Optuna monitor layout can be created without errors"""
    
    from mech_exo.reporting.dash_layout.optuna_monitor import create_optuna_monitor_layout
    
    layout = create_optuna_monitor_layout()
    
    # Verify layout is a Dash component
    assert hasattr(layout, 'children')
    
    # Convert to string representation and check for key elements
    layout_str = str(layout)
    assert 'Optuna Optimization Monitor' in layout_str
    assert 'Study:' in layout_str
    assert 'Open Optuna Dashboard' in layout_str
    assert 'Sharpe Ratio vs. Trial Number' in layout_str
    assert 'Parameter Importance' in layout_str


@pytest.fixture
def sample_optuna_trials():
    """Fixture providing sample Optuna trial data for testing"""
    return pd.DataFrame({
        'trial_id': ['trial_1', 'trial_2', 'trial_3', 'trial_4', 'trial_5'],
        'trial_number': [1, 2, 3, 4, 5],
        'study_name': ['factor_opt', 'factor_opt', 'factor_opt', 'test_study', 'test_study'],
        'sharpe_ratio': [0.2, 0.5, 0.8, 1.2, 1.5],
        'max_drawdown': [0.15, 0.12, 0.08, 0.05, 0.03],
        'constraint_violations': [2, 1, 0, 0, 0],
        'constraints_satisfied': [False, False, True, True, True],
        'calculation_date': [date.today() - timedelta(days=i) for i in range(5)],
        'elapsed_time_seconds': [150.0, 120.0, 95.0, 110.0, 105.0],
        'data_points': [1000] * 5,
        'sampler': ['TPESampler'] * 5,
        'pruner': ['MedianPruner'] * 5,
        'status': ['COMPLETE'] * 5
    })


def test_integration_with_sample_data(sample_optuna_trials):
    """Integration test with sample Optuna trial data"""
    
    with patch('mech_exo.datasource.storage.DataStorage') as mock_storage:
        mock_conn = Mock()
        mock_storage.return_value.conn = mock_conn
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = sample_optuna_trials
            
            from mech_exo.reporting.query import get_optuna_results
            from mech_exo.reporting.dash_layout.optuna_monitor import (
                update_optuna_data, update_sharpe_chart, update_summary_and_trials
            )
            
            # Test query function
            results = get_optuna_results(limit=50)
            assert len(results) == 5
            assert 'trial_duration_min' in results.columns
            
            # Test data update callback
            data_dict, status_text, status_class, last_updated = update_optuna_data(
                n_intervals=1, 
                selected_study='factor_opt'
            )
            
            # Should filter to 3 factor_opt trials
            assert len(data_dict['trials']) == 3
            assert '3 trials' in status_text
            
            # Test chart creation
            fig = update_sharpe_chart(data_dict)
            assert isinstance(fig, go.Figure)
            
            # Test summary update
            summary, trials = update_summary_and_trials(data_dict)
            summary_str = str(summary)
            assert '3' in summary_str  # Total trials
            assert '0.8' in summary_str or '0.80' in summary_str  # Best Sharpe from filtered data


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])