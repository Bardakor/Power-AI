#!/usr/bin/env python3
"""
Comprehensive Analysis Runner for Power AI
Runs all available analysis engines and generates comprehensive PDF report
"""

import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

class ComprehensiveAnalysisRunner:
    """Run all analysis engines and generate comprehensive PDF report"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.results = {}
        
    def run_power_system_analysis(self):
        """Run power system analysis"""
        print("🔌 Running Power System Analysis...")
        try:
            import sys
            sys.path.append('.')
            from tools.power_analysis import PowerSystemAnalyzer
            analyzer = PowerSystemAnalyzer()
            analyzer.run_analysis(sample_size=30000)
            self.results['power_analysis'] = True
            print("✅ Power system analysis completed")
        except Exception as e:
            print(f"❌ Power system analysis failed: {e}")
            self.results['power_analysis'] = False
    
    def run_mlops_analysis(self):
        """Run MLOps advanced analysis"""
        print("🤖 Running MLOps Advanced Analysis...")
        try:
            result = subprocess.run([sys.executable, "tools/mlops_advanced_engine.py"], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                self.results['mlops_analysis'] = True
                print("✅ MLOps analysis completed")
            else:
                print(f"❌ MLOps analysis failed: {result.stderr}")
                self.results['mlops_analysis'] = False
        except Exception as e:
            print(f"❌ MLOps analysis failed: {e}")
            self.results['mlops_analysis'] = False
    
    def run_advanced_ml(self):
        """Run advanced ML analysis"""
        print("🧠 Running Advanced ML Analysis...")
        try:
            result = subprocess.run([sys.executable, "tools/advanced_ml_engine.py"], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                self.results['advanced_ml'] = True
                print("✅ Advanced ML analysis completed")
            else:
                print(f"❌ Advanced ML analysis failed: {result.stderr}")
                self.results['advanced_ml'] = False
        except Exception as e:
            print(f"❌ Advanced ML analysis failed: {e}")
            self.results['advanced_ml'] = False
    
    def run_data_exploration(self):
        """Run data exploration"""
        print("🔍 Running Data Exploration...")
        try:
            result = subprocess.run([sys.executable, "tools/explore_data.py"], 
                                  capture_output=True, text=True, timeout=180)
            if result.returncode == 0:
                self.results['data_exploration'] = True
                print("✅ Data exploration completed")
            else:
                print(f"❌ Data exploration failed: {result.stderr}")
                self.results['data_exploration'] = False
        except Exception as e:
            print(f"❌ Data exploration failed: {e}")
            self.results['data_exploration'] = False
    
    def run_advanced_visualizations(self):
        """Run advanced visualizations"""
        print("🎨 Running Advanced Visualizations...")
        try:
            result = subprocess.run([sys.executable, "tools/advanced_visualizations.py"], 
                                  capture_output=True, text=True, timeout=240)
            if result.returncode == 0:
                self.results['advanced_viz'] = True
                print("✅ Advanced visualizations completed")
            else:
                print(f"❌ Advanced visualizations failed: {result.stderr}")
                self.results['advanced_viz'] = False
        except Exception as e:
            print(f"❌ Advanced visualizations failed: {e}")
            self.results['advanced_viz'] = False
    
    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        print("📄 Generating Comprehensive PDF Report...")
        try:
            import sys
            sys.path.append('.')
            from tools.pdf_report_generator import PowerAIPDFReportGenerator
            generator = PowerAIPDFReportGenerator()
            generator.load_and_analyze_data(sample_size=50000)  # Larger sample for final report
            report_path = generator.generate_pdf_report()
            self.results['pdf_report'] = str(report_path)
            print(f"✅ PDF report generated: {report_path}")
            return report_path
        except Exception as e:
            print(f"❌ PDF report generation failed: {e}")
            self.results['pdf_report'] = False
            return None
    
    def run_comprehensive_analysis(self):
        """Run complete comprehensive analysis"""
        print("🚀 POWER AI COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Data Exploration (Foundation)
        self.run_data_exploration()
        
        # Step 2: Power System Analysis (Core Analysis)
        self.run_power_system_analysis()
        
        # Step 3: Advanced ML Analysis (Parallel processing)
        print("\n🔄 Running ML Analyses in Parallel...")
        self.run_advanced_ml()
        self.run_mlops_analysis()
        
        # Step 4: Visualizations
        self.run_advanced_visualizations()
        
        # Step 5: Generate Final PDF Report
        print("\n📋 Generating Final Comprehensive Report...")
        final_report = self.generate_pdf_report()
        
        # Summary
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Results summary
        print("\n📊 ANALYSIS RESULTS SUMMARY:")
        for analysis, status in self.results.items():
            if analysis == 'pdf_report' and status:
                print(f"  ✅ {analysis}: {status}")
            elif status:
                print(f"  ✅ {analysis}: Success")
            else:
                print(f"  ❌ {analysis}: Failed")
        
        successful_analyses = sum(1 for status in self.results.values() if status)
        total_analyses = len(self.results)
        success_rate = (successful_analyses / total_analyses) * 100
        
        print(f"\nSuccess Rate: {successful_analyses}/{total_analyses} ({success_rate:.1f}%)")
        
        if final_report:
            print(f"\n📄 FINAL REPORT: {final_report}")
            print("📁 Check the outputs/ directory for all generated files")
        
        return self.results

def run_quick_analysis():
    """Run a quick analysis with essential components only"""
    print("⚡ POWER AI QUICK ANALYSIS")
    print("=" * 40)
    
    runner = ComprehensiveAnalysisRunner()
    
    # Essential analyses only
    runner.run_power_system_analysis()
    runner.generate_pdf_report()
    
    print("✅ Quick analysis complete!")
    return runner.results

def run_full_analysis():
    """Run the complete comprehensive analysis"""
    runner = ComprehensiveAnalysisRunner()
    return runner.run_comprehensive_analysis()

def main():
    """Main entry point with user selection"""
    print("🔌 Power AI Analysis Runner")
    print("=" * 40)
    print("1. Quick Analysis (Essential components)")
    print("2. Comprehensive Analysis (All engines)")
    print("3. PDF Report Only")
    print("4. Test All Components")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        run_quick_analysis()
    elif choice == "2":
        run_full_analysis()
    elif choice == "3":
        print("📄 Generating PDF Report Only...")
        import sys
        sys.path.append('.')
        from tools.pdf_report_generator import PowerAIPDFReportGenerator
        generator = PowerAIPDFReportGenerator()
        generator.load_and_analyze_data(sample_size=40000)
        report_path = generator.generate_pdf_report()
        print(f"✅ PDF report generated: {report_path}")
    elif choice == "4":
        print("🧪 Testing All Components...")
        subprocess.run([sys.executable, "test_integration.py"])
    else:
        print("Invalid choice. Running quick analysis...")
        run_quick_analysis()

if __name__ == "__main__":
    main() 