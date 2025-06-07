"""
Basic tests to verify project setup
"""

import pytest
from mech_exo import __version__


def test_version():
    """Test that version is correctly set"""
    assert __version__ == "0.1.0"


def test_imports():
    """Test that main package imports work"""
    import mech_exo
    import mech_exo.datasource
    import mech_exo.scoring
    import mech_exo.sizing
    import mech_exo.risk
    import mech_exo.execution
    import mech_exo.reporting
    import mech_exo.backtest
    import mech_exo.utils
    
    # If we get here without ImportError, the imports work
    assert True