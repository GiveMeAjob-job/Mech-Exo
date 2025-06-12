"""
Integration tests for Prefect factor optimization flow
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime
import yaml

try:
    from prefect.testing.utilities import prefect_test_harness
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    
    # Mock prefect_test_harness for testing without Prefect
    class MockTestHarness:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    def prefect_test_harness():
        return MockTestHarness()


@pytest.fixture
def temp_study_file():
    """Create temporary study file"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        yield f.name
    
    # Cleanup
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except FileNotFoundError:
        pass


def test_factor_optimization_flow_basic(temp_study_file, temp_output_dir):
    """Test basic factor optimization flow execution"""
    
    # Set environment for testing
    os.environ['TELEGRAM_DRY_RUN'] = 'true'
    
    try:
        from dags.factor_opt_flow import factor_optimization_flow
        
        # Use Prefect test harness if available
        with prefect_test_harness():
            print("🧪 Testing factor optimization flow...")
            
            # Run flow with minimal parameters for testing
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            result = factor_optimization_flow(
                n_trials=3,  # Small number for fast testing
                n_jobs=1,    # Single job to avoid parallelization issues
                stage=False, # Skip staging for test
                send_telegram=True,  # Will use dry-run mode
                notify_progress=False,
                study_name=f"test_study_{unique_id}"  # Unique study per test run
            )
            
            # Verify flow completed successfully
            assert result is not None, "Flow should return results"
            assert result['status'] == 'success', f"Flow should succeed, got: {result}"
            assert 'run_id' in result, "Result should contain run_id"
            assert 'best_sharpe' in result, "Result should contain best_sharpe"
            assert 'total_trials' in result, "Result should contain total_trials"
            assert result['total_trials'] == 3, f"Should have 3 trials, got {result['total_trials']}"
            
            # Verify YAML file was created
            yaml_path = result['yaml_path']
            assert Path(yaml_path).exists(), f"YAML file should exist: {yaml_path}"
            
            # Verify YAML structure
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            assert 'metadata' in yaml_data, "YAML should have metadata"
            assert 'factors' in yaml_data, "YAML should have factors"
            assert 'hyperparameters' in yaml_data, "YAML should have hyperparameters"
            
            # Check metadata completeness
            metadata = yaml_data['metadata']
            assert 'run_id' in metadata, "Metadata should have run_id"
            assert 'best_sharpe_ratio' in metadata, "Metadata should have best_sharpe_ratio"
            assert 'optimization_method' in metadata, "Metadata should have optimization_method"
            assert metadata['optimization_method'] == 'optuna_tpe_prefect', "Should use Prefect optimization method"
            
            print(f"✅ Flow test passed: run_id={result['run_id'][:8]}, best_sharpe={result['best_sharpe']:.4f}")
            
    finally:
        # Clean up environment
        if 'TELEGRAM_DRY_RUN' in os.environ:
            del os.environ['TELEGRAM_DRY_RUN']


def test_individual_flow_tasks():
    """Test individual flow tasks"""
    
    # Set up test environment
    os.environ['TELEGRAM_DRY_RUN'] = 'true'
    
    try:
        from dags.factor_opt_flow import (
            run_optuna_batch, export_yaml, store_opt_result, 
            notify_telegram, promote_yaml_to_staging
        )
        import uuid
        
        print("🧪 Testing individual flow tasks...")
        
        # Test run_optuna_batch
        run_id = str(uuid.uuid4())
        unique_study_name = f"test_individual_{str(uuid.uuid4())[:8]}"
        optimization_results = run_optuna_batch(
            n_trials=2,
            n_jobs=1,
            run_id=run_id,
            study_name=unique_study_name
        )
        
        assert optimization_results is not None, "Optimization should return results"
        assert optimization_results['run_id'] == run_id, "Should preserve run_id"
        assert 'best_value' in optimization_results, "Should have best_value"
        assert 'total_trials' in optimization_results, "Should have total_trials"
        
        print(f"✅ Optimization task passed: {optimization_results['total_trials']} trials")
        
        # Test export_yaml
        export_info = export_yaml(optimization_results)
        
        assert export_info is not None, "Export should return info"
        assert 'yaml_path' in export_info, "Should have yaml_path"
        assert 'file_size' in export_info, "Should have file_size"
        assert Path(export_info['yaml_path']).exists(), "YAML file should exist"
        
        print(f"✅ Export task passed: {export_info['yaml_path']}")
        
        # Test store_opt_result
        store_success = store_opt_result(optimization_results, export_info)
        
        assert store_success, "Store should succeed"
        
        print("✅ Store task passed")
        
        # Test notify_telegram (dry-run)
        telegram_success = notify_telegram(optimization_results, export_info, send_file=False)
        
        assert telegram_success, "Telegram notification should succeed in dry-run"
        
        print("✅ Telegram task passed")
        
        # Test promote_yaml_to_staging (disabled)
        staging_success = promote_yaml_to_staging(export_info, stage=False)
        
        assert staging_success, "Staging should succeed when disabled"
        
        print("✅ Staging task passed")
        
        print("🎯 All individual tasks passed!")
        
    finally:
        # Clean up environment
        if 'TELEGRAM_DRY_RUN' in os.environ:
            del os.environ['TELEGRAM_DRY_RUN']


def test_telegram_message_formatting():
    """Test Telegram message formatting"""
    
    from dags.factor_opt_flow import _create_telegram_message
    
    # Mock optimization results
    optimization_results = {
        'run_id': 'test-run-12345678',
        'best_value': 1.2847,
        'total_trials': 50,
        'elapsed_time': 3600,  # 1 hour
        'best_user_attrs': {
            'max_drawdown': 0.08,
            'constraint_violations': 0,
            'constraints_satisfied': True
        }
    }
    
    export_info = {
        'yaml_path': 'factors_opt_2025-06-09.yml',
        'file_size_mb': 0.5
    }
    
    message = _create_telegram_message(optimization_results, export_info)
    
    # Verify message content
    assert '🎯' in message, "Message should have optimization emoji"
    assert '1.2847' in message, "Message should include Sharpe ratio"
    assert '50' in message, "Message should include trial count"
    assert '60.0 minutes' in message, "Message should include elapsed time"
    assert 'factors\\_opt\\_2025\\-06\\-09\\.yml' in message, "Message should include escaped filename"
    assert 'Satisfied' in message, "Message should show constraint status"
    assert 'test-run' in message, "Message should include run ID prefix"
    
    print("✅ Telegram message formatting test passed")


def test_yaml_export_structure():
    """Test YAML export structure"""
    
    from dags.factor_opt_flow import export_yaml
    import uuid
    
    # Mock optimization results
    optimization_results = {
        'run_id': str(uuid.uuid4()),
        'study_name': 'test_study',
        'best_trial_number': 42,
        'best_value': 1.23,
        'total_trials': 50,
        'elapsed_time': 1800,
        'sampler': 'TPESampler',
        'pruner': 'MedianPruner',
        'data_points': 5000,
        'best_params': {
            'weight_pe_ratio': 0.1234,
            'weight_return_on_equity': -0.5678,
            'cash_pct': 0.15,
            'stop_loss_pct': 0.08
        },
        'best_user_attrs': {
            'max_drawdown': 0.08,
            'constraints_satisfied': True,
            'constraint_violations': 0
        }
    }
    
    with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as f:
        temp_file = f.name
    
    try:
        export_info = export_yaml(optimization_results, temp_file)
        
        # Verify export info
        assert export_info['yaml_path'] == temp_file
        assert export_info['factor_count'] == 2  # pe_ratio and return_on_equity
        assert export_info['hyperparameter_count'] == 2  # cash_pct and stop_loss_pct
        
        # Load and verify YAML
        with open(temp_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Check structure
        assert 'metadata' in yaml_data
        assert 'factors' in yaml_data
        assert 'hyperparameters' in yaml_data
        
        # Check metadata
        metadata = yaml_data['metadata']
        assert metadata['optimization_method'] == 'optuna_tpe_prefect'
        assert metadata['best_sharpe_ratio'] == 1.23
        assert metadata['total_trials'] == 50
        assert metadata['constraints_satisfied'] == True
        
        # Check factors
        factors = yaml_data['factors']
        assert 'fundamental' in factors
        assert 'pe_ratio' in factors['fundamental']
        assert factors['fundamental']['pe_ratio']['weight'] == 0.1234
        
        # Check hyperparameters
        hyperparams = yaml_data['hyperparameters']
        assert hyperparams['cash_pct'] == 0.15
        assert hyperparams['stop_loss_pct'] == 0.08
        
        print("✅ YAML export structure test passed")
        
    finally:
        try:
            os.unlink(temp_file)
        except FileNotFoundError:
            pass


def test_database_storage():
    """Test database storage functionality"""
    
    from dags.factor_opt_flow import store_opt_result
    import uuid
    import tempfile
    
    # Mock data
    optimization_results = {
        'run_id': str(uuid.uuid4()),
        'best_value': 1.5,
        'best_trial_number': 25,
        'total_trials': 100,
        'elapsed_time': 7200,
        'n_jobs': 4,
        'data_points': 10000,
        'study_name': 'test_study',
        'sampler': 'TPESampler',
        'pruner': 'MedianPruner',
        'best_user_attrs': {
            'max_drawdown': 0.09,
            'constraints_satisfied': True,
            'constraint_violations': 1,
            'total_return': 0.25,
            'volatility': 0.15
        }
    }
    
    export_info = {
        'yaml_path': 'test/factors.yml',
        'file_size': 2048
    }
    
    # Test database storage
    success = store_opt_result(optimization_results, export_info)
    
    assert success, "Database storage should succeed"
    
    # Verify data was stored (would need actual DB connection to test)
    print("✅ Database storage test passed")


if __name__ == "__main__":
    # Run tests directly
    print("🧪 Running Prefect factor optimization flow tests...")
    
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as study_f:
        study_file = study_f.name
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Run tests
        test_factor_optimization_flow_basic(study_file, temp_dir)
        test_individual_flow_tasks()
        test_telegram_message_formatting()
        test_yaml_export_structure()
        test_database_storage()
        
        print("🎯 All Prefect flow tests passed!")
        
    finally:
        # Cleanup
        try:
            os.unlink(study_file)
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass