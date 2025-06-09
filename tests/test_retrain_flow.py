"""
Tests for Strategy Retrain Flow

Tests the retrain flow structure and basic functionality
without requiring Prefect runtime dependencies.
"""

import pytest
import ast
from pathlib import Path


def test_retrain_flow_structure():
    """Test that retrain flow has correct structure"""
    flow_file = Path("dags/retrain_flow.py")
    assert flow_file.exists(), "Retrain flow file should exist"
    
    # Parse the file
    with open(flow_file, 'r') as f:
        tree = ast.parse(f.read())
    
    # Find flows and tasks
    flows = []
    tasks = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'id'):
                    if decorator.func.id == 'flow':
                        flows.append(node.name)
                    elif decorator.func.id == 'task':
                        tasks.append(node.name)
    
    # Verify flow structure
    assert len(flows) == 1, "Should have exactly one flow"
    assert "strategy_retrain_flow" in flows, "Should have strategy_retrain_flow"
    
    # Verify required tasks
    expected_tasks = [
        "check_drift_breach",
        "load_retrain_data", 
        "refit_factors",
        "validate_retrained_strategy",
        "deploy_new_factors",
        "send_retrain_notification"
    ]
    
    for task in expected_tasks:
        assert task in tasks, f"Should have {task} task"
    
    assert len(tasks) >= len(expected_tasks), f"Should have at least {len(expected_tasks)} tasks"


def test_retrain_flow_imports():
    """Test that retrain flow has correct imports"""
    flow_file = Path("dags/retrain_flow.py")
    
    with open(flow_file, 'r') as f:
        content = f.read()
    
    # Check for required imports
    required_imports = [
        "from prefect import flow, task",
        "from datetime import datetime, date",
        "from pathlib import Path",
        "import logging"
    ]
    
    for import_stmt in required_imports:
        assert import_stmt in content, f"Should have import: {import_stmt}"


def test_retrain_flow_docstrings():
    """Test that key functions have docstrings"""
    flow_file = Path("dags/retrain_flow.py")
    
    with open(flow_file, 'r') as f:
        tree = ast.parse(f.read())
    
    # Check main flow has docstring
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "strategy_retrain_flow":
            assert ast.get_docstring(node) is not None, "Main flow should have docstring"
            docstring = ast.get_docstring(node)
            assert "drift-triggered" in docstring.lower(), "Docstring should mention drift triggering"


def test_retrain_flow_parameters():
    """Test that main flow has expected parameters"""
    flow_file = Path("dags/retrain_flow.py")
    
    with open(flow_file, 'r') as f:
        tree = ast.parse(f.read())
    
    # Find main flow function
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "strategy_retrain_flow":
            # Check parameters
            args = [arg.arg for arg in node.args.args]
            
            expected_params = ["target_date", "lookback_months", "force_retrain"]
            for param in expected_params:
                assert param in args, f"Flow should have {param} parameter"


if __name__ == "__main__":
    # Run tests manually
    print("🧪 Testing Retrain Flow Structure...")
    
    try:
        test_retrain_flow_structure()
        print("✅ Flow structure test passed")
        
        test_retrain_flow_imports() 
        print("✅ Imports test passed")
        
        test_retrain_flow_docstrings()
        print("✅ Docstrings test passed")
        
        test_retrain_flow_parameters()
        print("✅ Parameters test passed")
        
        print("🎉 All retrain flow tests passed!")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")