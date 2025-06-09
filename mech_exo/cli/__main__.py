"""
CLI package main entry point
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mech_exo.cli import main

if __name__ == "__main__":
    main()