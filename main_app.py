#!/usr/bin/env python3
"""
Power AI - Main Application Entry Point for Cloud Deployment
Optimized for Docker and GCP Cloud Run deployment
"""

import os
import sys
from pathlib import Path

# Add the tools directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "tools"))

def main():
    """Main application entry point"""
    print("🚀 Starting Power AI Application...")
    
    # Import and run the dashboard
    from dash_frontend import main as dashboard_main
    
    # Run the dashboard
    dashboard_main()

if __name__ == "__main__":
    main() 