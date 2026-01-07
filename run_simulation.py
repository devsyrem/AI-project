#!/usr/bin/env python3
"""
Startup script for Predator: Badlands simulation.
Handles import path setup and launches the main application.
"""
import sys
import os

# Add the src directory to the Python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_dir)

if __name__ == "__main__":
    # Import and run the main module
    from main import main
    main()