#!/usr/bin/env python3
"""
Power AI - Complete Data Analysis Suite

This is the main entry point for all data exploration and analysis tools.
Run this script to see all available options and get started with your analysis.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


class PowerAIManager:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.csv_dir = self.base_dir / "outputs/csv_data"
        self.available_tools = {
            'quick': 'Quick overview of CSV files',
            'explore': 'Comprehensive data exploration with visualizations',
            'power': 'Advanced power system analysis with dashboards',
            'notebook': 'Launch Jupyter notebook for interactive analysis',
            'summary': 'Show summary of available tools and results'
        }
    
    def check_data_availability(self):
        """Check if CSV data is available"""
        if not self.csv_dir.exists():
            print("❌ No CSV data found!")
            print("   Please run the parser first: python main.py")
            return False
        
        csv_files = list(self.csv_dir.glob("*/*.csv"))
        if not csv_files:
            print("❌ No CSV files found in subdirectories!")
            print("   Please run the parser first: python main.py")
            return False
        
        return True
    
    def show_status(self):
        """Show current status and available data"""
        print("🔌 POWER AI - DATA ANALYSIS STATUS")
        print("=" * 50)
        
        # Check for source data
        data_dir = self.base_dir / "data"
        db_files = list(data_dir.glob("*.db")) if data_dir.exists() else []
        
        print(f"\n📁 SOURCE DATA:")
        if db_files:
            for db_file in db_files:
                size_mb = db_file.stat().st_size / 1024**2
                print(f"  ✅ {db_file.name} ({size_mb:.1f} MB)")
        else:
            print("  ❌ No SQLite database files found in data/")
        
        # Check for CSV exports
        print(f"\n📊 EXPORTED DATA:")
        if self.csv_dir.exists():
            csv_dirs = [d for d in self.csv_dir.iterdir() if d.is_dir()]
            if csv_dirs:
                for csv_dir in csv_dirs:
                    csv_files = list(csv_dir.glob("*.csv"))
                    if csv_files:
                        csv_file = csv_files[0]
                        size_mb = csv_file.stat().st_size / 1024**2
                        # Count rows quickly
                        with open(csv_file, 'r') as f:
                            row_count = sum(1 for _ in f) - 1
                        print(f"  ✅ {csv_dir.name}: {row_count:,} rows ({size_mb:.1f} MB)")
            else:
                print("  ❌ No CSV subdirectories found")
        else:
            print("  ❌ CSV output directory not found")
        
        # Check for analysis results
        print(f"\n📈 ANALYSIS RESULTS:")
        
        # Basic exploration results
        explore_dir = self.base_dir / "outputs/exploration"
        if explore_dir.exists():
            result_files = list(explore_dir.glob("*.txt"))
            viz_dirs = [d for d in explore_dir.iterdir() if d.is_dir()]
            print(f"  ✅ Basic exploration: {len(result_files)} reports, {len(viz_dirs)} visualization sets")
        else:
            print("  ❌ No basic exploration results")
        
        # Power system analysis results
        power_dir = self.base_dir / "outputs/power_analysis"
        if power_dir.exists():
            dashboards = list(power_dir.glob("*.png"))
            reports = list(power_dir.glob("*.txt"))
            print(f"  ✅ Power analysis: {len(dashboards)} dashboards, {len(reports)} reports")
        else:
            print("  ❌ No power system analysis results")
        
        print(f"\n🛠️  AVAILABLE TOOLS:")
        for tool, description in self.available_tools.items():
            print(f"  • {tool:10} - {description}")
    
    def run_quick_analysis(self):
        """Run quick overview"""
        if not self.check_data_availability():
            return
        
        print("🚀 Running Quick Analysis...")
        subprocess.run([sys.executable, "tools/quick_explore.py"], cwd=self.base_dir)
    
    def run_full_exploration(self):
        """Run comprehensive exploration"""
        if not self.check_data_availability():
            return
        
        print("🚀 Running Full Data Exploration...")
        subprocess.run([sys.executable, "tools/explore_data.py"], cwd=self.base_dir)
    
    def run_power_analysis(self):
        """Run power system analysis"""
        if not self.check_data_availability():
            return
        
        print("🚀 Running Power System Analysis...")
        subprocess.run([sys.executable, "tools/power_analysis.py"], cwd=self.base_dir)
    
    def launch_notebook(self):
        """Launch Jupyter notebook"""
        notebook_file = self.base_dir / "notebooks/data_exploration.ipynb"
        if not notebook_file.exists():
            print("❌ Notebook file not found!")
            return
        
        print("🚀 Launching Jupyter Notebook...")
        print("   Opening notebooks/data_exploration.ipynb...")
        try:
            subprocess.run(["jupyter", "notebook", str(notebook_file)], cwd=self.base_dir)
        except FileNotFoundError:
            print("❌ Jupyter not found! Install it with: pip install jupyter")
    
    def show_summary(self):
        """Show summary of tools and results"""
        subprocess.run([sys.executable, "tools/exploration_summary.py"], cwd=self.base_dir)
    
    def run_interactive_menu(self):
        """Run interactive menu"""
        while True:
            print("\n🔌 POWER AI - INTERACTIVE MENU")
            print("=" * 40)
            print("1. Show Status")
            print("2. Quick Analysis")
            print("3. Full Exploration")
            print("4. Power System Analysis")
            print("5. Launch Jupyter Notebook")
            print("6. Show Summary")
            print("7. Run All Analyses")
            print("0. Exit")
            
            try:
                choice = input("\nSelect option (0-7): ").strip()
                
                if choice == '0':
                    print("👋 Goodbye!")
                    break
                elif choice == '1':
                    self.show_status()
                elif choice == '2':
                    self.run_quick_analysis()
                elif choice == '3':
                    self.run_full_exploration()
                elif choice == '4':
                    self.run_power_analysis()
                elif choice == '5':
                    self.launch_notebook()
                elif choice == '6':
                    self.show_summary()
                elif choice == '7':
                    print("🚀 Running All Analyses...")
                    self.run_quick_analysis()
                    print("\n" + "="*50)
                    self.run_full_exploration()
                    print("\n" + "="*50)
                    self.run_power_analysis()
                    print("\n✅ All analyses complete!")
                else:
                    print("❌ Invalid option! Please select 0-7.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("Press Enter to continue...")


def main():
    parser = argparse.ArgumentParser(description='Power AI - Complete Data Analysis Suite')
    parser.add_argument('command', nargs='?', choices=['quick', 'explore', 'power', 'notebook', 'summary', 'status'], 
                       help='Command to run')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run interactive menu')
    
    args = parser.parse_args()
    
    manager = PowerAIManager()
    
    if args.interactive or not args.command:
        manager.run_interactive_menu()
    elif args.command == 'status':
        manager.show_status()
    elif args.command == 'quick':
        manager.run_quick_analysis()
    elif args.command == 'explore':
        manager.run_full_exploration()
    elif args.command == 'power':
        manager.run_power_analysis()
    elif args.command == 'notebook':
        manager.launch_notebook()
    elif args.command == 'summary':
        manager.show_summary()


if __name__ == "__main__":
    main()
