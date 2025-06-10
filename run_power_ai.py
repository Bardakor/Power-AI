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
    print("3. 🔬 Run MLOps ADVANCED Analysis (correlation + optimization)")
    print("4. 🎨 Generate Interactive Visualizations")
    print("5. 🔬 Generate ADVANCED Visualizations")
    print("6. 📊 Create ML Visualizations")
    print("7. 🌐 Launch Dash Dashboard (interactive web app)")
    print("8. 🛠️ Run Utilities & Configuration")
    print("9. 🚀 RUN COMPLETE ADVANCED SYSTEM")
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
        print("\n🔬 Running MLOps ADVANCED Analysis...")
        print("Comprehensive ML pipeline with:")
        print("• Feature correlation analysis and removal")
        print("• Multi-method feature selection")
        print("• Hyperparameter optimization")
        print("• Cross-validation with time series")
        print("• Advanced XGBoost optimization")
        print("• Comprehensive reporting")
        subprocess.run([sys.executable, "tools/mlops_advanced_engine.py"])
        
    elif choice == "4":
        print("\n🎨 Generating Interactive Visualizations...")
        subprocess.run([sys.executable, "tools/interactive_viz.py"])
        
    elif choice == "5":
        print("\n🔬 Generating ADVANCED Visualizations...")
        print("Power quality dashboards, electrical analysis, PDU monitoring")
        subprocess.run([sys.executable, "tools/advanced_visualizations.py"])
        
    elif choice == "6":
        print("\n📊 Creating ML Visualizations...")
        subprocess.run([sys.executable, "tools/ml_visualizations.py"])
        
    elif choice == "7":
        print("\n🌐 Launching Dash Dashboard...")
        print("Dashboard will be available at: http://localhost:8050")
        print("Press Ctrl+C to stop the dashboard")
        subprocess.run([sys.executable, "tools/dash_frontend.py"])
        
    elif choice == "8":
        print("\n🛠️ Running Utilities...")
        subprocess.run([sys.executable, "tools/additional_utilities.py"])
        
    elif choice == "9":
        print("\n🚀 RUNNING COMPLETE ADVANCED SYSTEM...")
        print("This will run the MLOps engine plus all visualizations")
        
        print("\n1/5 🔬 Running MLOps Advanced Analysis...")
        subprocess.run([sys.executable, "tools/mlops_advanced_engine.py"])
        
        print("\n2/5 🎯 Running Advanced ML Engine...")
        subprocess.run([sys.executable, "tools/advanced_ml_engine.py"])
        
        print("\n3/5 🔬 Generating Advanced Visualizations...")
        subprocess.run([sys.executable, "tools/advanced_visualizations.py"])
        
        print("\n4/5 🎨 Generating Interactive Visualizations...")
        subprocess.run([sys.executable, "tools/interactive_viz.py"])
        
        print("\n5/5 📊 Creating ML Visualizations...")
        subprocess.run([sys.executable, "tools/ml_visualizations.py"])
        
        print("\n✅ Complete advanced system finished! Check outputs folder for results.")
        
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
    print("   • XGBoost optimization with hyperparameter tuning")
    print("   • Cross-validation with time series splits")
    print("   • Multi-method feature selection")
    print("   • Correlation analysis and removal")
    print("   • MLOps pipeline with comprehensive reporting")
    print("\n🎯 Outputs:")
    print("   • Interactive dashboards")
    print("   • Optimized predictive models")
    print("   • Anomaly reports")
    print("   • Feature importance analysis")
    print("   • Correlation matrices")
    print("   • Real-time visualizations")
    print("=" * 60)

if __name__ == "__main__":
    # Show system info first
    show_system_info()
    
    # Then run main menu
    main()
