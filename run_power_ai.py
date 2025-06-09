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
    print("1. 🤖 Run Basic ML Engine (simple models)")
    print("2. 🎯 Run ADVANCED ML Engine (sophisticated analysis)")
    print("3. 🎨 Generate Interactive Visualizations")
    print("4. 🔬 Generate ADVANCED Visualizations")
    print("5. 📊 Create ML Visualizations")
    print("6. 🌐 Launch Dash Dashboard (interactive web app)")
    print("7. 🛠️ Run Utilities & Configuration")
    print("8. 🚀 RUN COMPLETE ADVANCED SYSTEM")
    print("9. 🔄 RUN EVERYTHING (all components)")
    print("0. ❌ Exit")
    
    choice = input("\nEnter your choice (0-9): ").strip()
    
    if choice == "1":
        print("\n🤖 Running Basic ML Engine...")
        subprocess.run([sys.executable, "tools/ml_engine.py"])
        
    elif choice == "2":
        print("\n🎯 Running ADVANCED ML Engine...")
        print("This uses sophisticated electrical engineering features,")
        print("XGBoost, time series analysis, and advanced anomaly detection")
        subprocess.run([sys.executable, "tools/advanced_ml_engine.py"])
        
    elif choice == "3":
        print("\n🎨 Generating Interactive Visualizations...")
        subprocess.run([sys.executable, "tools/interactive_viz.py"])
        
    elif choice == "4":
        print("\n🔬 Generating ADVANCED Visualizations...")
        print("Power quality dashboards, electrical analysis, PDU monitoring")
        subprocess.run([sys.executable, "tools/advanced_visualizations.py"])
        
    elif choice == "5":
        print("\n📊 Creating ML Visualizations...")
        subprocess.run([sys.executable, "tools/ml_visualizations.py"])
        
    elif choice == "6":
        print("\n🌐 Launching Dash Dashboard...")
        print("Dashboard will be available at: http://localhost:8050")
        print("Press Ctrl+C to stop the dashboard")
        subprocess.run([sys.executable, "tools/dash_frontend.py"])
        
    elif choice == "7":
        print("\n🛠️ Running Utilities...")
        subprocess.run([sys.executable, "tools/additional_utilities.py"])
        
    elif choice == "8":
        print("\n🚀 RUNNING COMPLETE ADVANCED SYSTEM...")
        print("This will run the advanced ML engine plus visualizations")
        
        print("\n1/4 🎯 Running Advanced ML Engine...")
        subprocess.run([sys.executable, "tools/advanced_ml_engine.py"])
        
        print("\n2/4 🔬 Generating Advanced Visualizations...")
        subprocess.run([sys.executable, "tools/advanced_visualizations.py"])
        
        print("\n3/4 🎨 Generating Interactive Visualizations...")
        subprocess.run([sys.executable, "tools/interactive_viz.py"])
        
        print("\n4/4 📊 Creating ML Visualizations...")
        subprocess.run([sys.executable, "tools/ml_visualizations.py"])
        
        print("\n✅ Advanced system complete! Check outputs folder for results.")
        
    elif choice == "9":
        print("\n🔄 RUNNING EVERYTHING...")
        print("This will run ALL components in sequence")
        
        print("\n1/6 🎯 Running Advanced ML Engine...")
        subprocess.run([sys.executable, "tools/advanced_ml_engine.py"])
        
        print("\n2/6 🤖 Running Basic ML Engine (for comparison)...")
        subprocess.run([sys.executable, "tools/ml_engine.py"])
        
        print("\n3/6 🔬 Generating Advanced Visualizations...")
        subprocess.run([sys.executable, "tools/advanced_visualizations.py"])
        
        print("\n4/6 🎨 Generating Interactive Visualizations...")
        subprocess.run([sys.executable, "tools/interactive_viz.py"])
        
        print("\n5/6 📊 Creating ML Visualizations...")
        subprocess.run([sys.executable, "tools/ml_visualizations.py"])
        
        print("\n6/6 🌐 Starting Dashboard...")
        print("Dashboard available at: http://localhost:8050")
        subprocess.run([sys.executable, "tools/dash_frontend.py"])
        
    elif choice == "0":
        print("👋 Goodbye!")
        return
        
    else:
        print("❌ Invalid choice. Please try again.")
        main()

def show_system_info():
    """Show information about the Power AI system"""
    print("\n" + "=" * 60)
    print("🔋 POWER AI SYSTEM INFORMATION")
    print("=" * 60)
    print("📊 Data Sources:")
    print("   • UPS systems (voltage, current, power, load)")
    print("   • Energy meters (primary and secondary)")
    print("   • Power Distribution Units (PDUs 1-8)")
    print("   • Battery monitoring systems")
    print("   • Environmental sensors")
    print("\n🤖 ML Capabilities:")
    print("   • Power consumption forecasting")
    print("   • Anomaly detection")
    print("   • Load balancing optimization")
    print("   • Energy efficiency analysis")
    print("   • Predictive maintenance")
    print("\n📈 Advanced Features:")
    print("   • Electrical engineering feature extraction")
    print("   • Time series analysis with lag features")
    print("   • XGBoost and Random Forest models")
    print("   • Cross-validation with time series splits")
    print("   • Feature selection and scaling")
    print("   • Multi-method anomaly detection")
    print("\n🎯 Outputs:")
    print("   • Interactive dashboards")
    print("   • Predictive models")
    print("   • Anomaly reports")
    print("   • Optimization recommendations")
    print("   • Real-time visualizations")
    print("=" * 60)

if __name__ == "__main__":
    # Show system info first
    show_system_info()
    
    # Then run main menu
    main()
