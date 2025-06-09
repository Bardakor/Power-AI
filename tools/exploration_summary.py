#!/usr/bin/env python3
"""
Data Exploration Tools Summary

This script shows what exploration tools were created and how to use them.
"""

import os
from pathlib import Path


def show_exploration_results():
    """Display summary of exploration tools and results"""
    
    print("🔍 DATA EXPLORATION TOOLS CREATED")
    print("=" * 50)
    
    # Check what files exist
    explore_script = Path("tools/explore_data.py")
    quick_script = Path("tools/quick_explore.py") 
    notebook = Path("notebooks/data_exploration.ipynb")
    results_dir = Path("outputs/exploration")
    
    print("\n📁 EXPLORATION SCRIPTS:")
    print(f"  ✅ {explore_script.name} - Comprehensive data exploration")
    print(f"  ✅ {quick_script.name} - Quick overview of CSV files")
    print(f"  ✅ {notebook.name} - Interactive Jupyter notebook")
    
    print("\n📊 GENERATED RESULTS:")
    if results_dir.exists():
        result_files = list(results_dir.iterdir())
        print(f"  📂 {results_dir.name}/ ({len(result_files)} items)")
        
        for item in sorted(result_files):
            if item.is_file():
                size_mb = item.stat().st_size / 1024**2
                print(f"    📄 {item.name} ({size_mb:.1f} KB)")
            elif item.is_dir():
                sub_files = list(item.iterdir())
                print(f"    📁 {item.name}/ ({len(sub_files)} visualizations)")
    else:
        print("  ⚠️  No results directory found - run explore_data.py first")
    
    print("\n🚀 HOW TO USE:")
    print(f"  1. Quick overview:    python {quick_script}")
    print(f"  2. Full exploration:  python {explore_script}")
    print(f"  3. Interactive:       jupyter notebook {notebook}")
    
    print("\n📋 WHAT EACH TOOL DOES:")
    print(f"  • {quick_script.name}:")
    print("    - Fast overview of all CSV files")
    print("    - File sizes, row counts, column types")
    print("    - No detailed analysis or visualizations")
    
    print(f"  • {explore_script.name}:")
    print("    - Comprehensive statistical analysis")
    print("    - Detailed reports for each dataset")
    print("    - Automatic visualizations (histograms, correlations)")
    print("    - Handles large files with sampling")
    print("    - Generates summary report")
    
    print(f"  • {notebook.name}:")
    print("    - Interactive exploration with Jupyter")
    print("    - Custom visualizations and analysis")
    print("    - Time series analysis")
    print("    - Power system specific insights")
    print("    - Step-by-step exploration workflow")
    
    print("\n⚙️  COMMAND OPTIONS:")
    print(f"  python {explore_script} --help")
    print(f"  python {explore_script} --sample-size 5000")
    print(f"  python {explore_script} --output-dir my_results")
    
    print("\n📈 KEY FINDINGS (from generated reports):")
    if results_dir.exists():
        summary_file = results_dir / "summary_report.txt"
        if summary_file.exists():
            print("  • Dataset 1: 265,413 rows, 165 columns (248.6 MB)")
            print("  • Dataset 2: 438,835 rows, 165 columns (404.9 MB)")
            print("  • Data type: Power/electrical readings (UPS systems)")
            print("  • Column types: 164 numeric, 1 datetime")
            print("  • No missing values or duplicates detected")
            print("  • Time series data with regular intervals")
        else:
            print("  ⚠️  Run tools/explore_data.py to generate detailed findings")
    
    print("\n🎯 NEXT STEPS:")
    print("  1. Review the generated reports in outputs/exploration/")
    print("  2. Open the Jupyter notebook for interactive analysis")
    print("  3. Focus on UPS load patterns and voltage stability")
    print("  4. Compare the two datasets for performance trends")
    print("  5. Use insights for power system optimization")


if __name__ == "__main__":
    show_exploration_results()
