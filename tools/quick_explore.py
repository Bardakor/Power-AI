#!/usr/bin/env python3
"""
Quick Data Explorer - Fast overview of CSV files

This script provides a quick summary of all CSV files in subdirectories.
"""

import pandas as pd
import os
from pathlib import Path
import argparse


def quick_explore(csv_output_dir="outputs/csv_data"):
    """Quick exploration of all CSV files"""
    csv_dir = Path(csv_output_dir)
    
    print(f"🔍 Quick Data Exploration")
    print(f"{'='*50}")
    
    if not csv_dir.exists():
        print(f"Directory {csv_dir} not found!")
        return
    
    csv_files = []
    for subdir in csv_dir.iterdir():
        if subdir.is_dir():
            for csv_file in subdir.glob("*.csv"):
                csv_files.append({
                    'path': csv_file,
                    'dataset': subdir.name,
                    'table': csv_file.stem
                })
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):\n")
    
    for i, csv_info in enumerate(csv_files, 1):
        print(f"{i}. Dataset: {csv_info['dataset']}")
        print(f"   Table: {csv_info['table']}")
        
        try:
            # Get file size
            file_size_mb = csv_info['path'].stat().st_size / 1024**2
            
            # Count lines (fast way)
            with open(csv_info['path'], 'r') as f:
                line_count = sum(1 for _ in f) - 1  # -1 for header
            
            # Read just the header to get column info
            df_head = pd.read_csv(csv_info['path'], nrows=5)
            
            print(f"   Size: {file_size_mb:.1f} MB")
            print(f"   Rows: {line_count:,}")
            print(f"   Columns: {len(df_head.columns)}")
            
            # Show column types
            numeric_cols = len(df_head.select_dtypes(include=['int64', 'float64']).columns)
            text_cols = len(df_head.select_dtypes(include=['object']).columns)
            
            print(f"   Column Types: {numeric_cols} numeric, {text_cols} text")
            
            # Show first few column names
            col_sample = list(df_head.columns[:5])
            if len(df_head.columns) > 5:
                col_sample.append('...')
            print(f"   Sample Columns: {', '.join(col_sample)}")
            
        except Exception as e:
            print(f"   Error reading file: {e}")
        
        print()


def main():
    parser = argparse.ArgumentParser(description='Quick exploration of CSV files')
    parser.add_argument('--csv-dir', default='outputs/csv_data',
                       help='Directory containing CSV subdirectories (default: outputs/csv_data)')
    
    args = parser.parse_args()
    quick_explore(args.csv_dir)


if __name__ == "__main__":
    main()
