#!/usr/bin/env python3
"""
🚀 POWER AI COMPLETE SYSTEM ORCHESTRATOR 🚀
Master controller for the complete ML and Dashboard system
"""

import sys
from pathlib import Path
import subprocess
import time

def main():
    print("🚀 POWER AI SYSTEM - COMPLETE ML & DASHBOARD SUITE")
    print("=" * 60)
    print("Choose your adventure:")
    print("1. 🤖 Run ML Engine (train models, predictions, anomalies)")
    print("2. 🎨 Generate Interactive Visualizations")
    print("3. 📊 Create ML Visualizations")
    print("4. 🌐 Launch Dash Dashboard (interactive web app)")
    print("5. 🛠️ Run Utilities & Configuration")
    print("6. 🎯 RUN EVERYTHING (complete system)")
    print("7. ❌ Exit")
    
    choice = input("\nEnter your choice (1-7): ").strip()
    
    if choice == "1":
        print("\n🤖 Running ML Engine...")
        subprocess.run([sys.executable, "tools/ml_engine.py"])
        
    elif choice == "2":
        print("\n🎨 Generating Interactive Visualizations...")
        subprocess.run([sys.executable, "tools/interactive_viz.py"])
        
    elif choice == "3":
        print("\n📊 Creating ML Visualizations...")
        subprocess.run([sys.executable, "tools/ml_visualizations.py"])
        
    elif choice == "4":
        print("\n🌐 Launching Dash Dashboard...")
        print("Dashboard will be available at: http://localhost:8050")
        print("Press Ctrl+C to stop the dashboard")
        subprocess.run([sys.executable, "tools/dash_frontend.py"])
        
    elif choice == "5":
        print("\n🛠️ Running Utilities...")
        subprocess.run([sys.executable, "tools/additional_utilities.py"])
        
    elif choice == "6":
        print("\n🎯 RUNNING COMPLETE SYSTEM...")
        print("This will run all components in sequence")
        
        print("\n1/4 🤖 Running ML Engine...")
        subprocess.run([sys.executable, "tools/ml_engine.py"])
        
        print("\n2/4 🎨 Generating Interactive Visualizations...")
        subprocess.run([sys.executable, "tools/interactive_viz.py"])
        
        print("\n3/4 📊 Creating ML Visualizations...")
        subprocess.run([sys.executable, "tools/ml_visualizations.py"])
        
        print("\n4/4 🌐 Starting Dashboard...")
        print("Dashboard available at: http://localhost:8050")
        subprocess.run([sys.executable, "tools/dash_frontend.py"])
        
    elif choice == "7":
        print("👋 Goodbye!")
        return
        
    else:
        print("❌ Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()
