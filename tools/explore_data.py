#!/usr/bin/env python3
"""
Data Exploration Script for CSV Files

This script explores CSV files in subdirectories under csv_output/
and generates comprehensive analysis reports.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class DataExplorer:
    def __init__(self, csv_output_dir="outputs/csv_data", output_dir="outputs/exploration"):
        self.csv_output_dir = Path(csv_output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def find_csv_files(self):
        """Find all CSV files in subdirectories"""
        csv_files = []
        for subdir in self.csv_output_dir.iterdir():
            if subdir.is_dir():
                for csv_file in subdir.glob("*.csv"):
                    csv_files.append({
                        'path': csv_file,
                        'dataset': subdir.name,
                        'table': csv_file.stem
                    })
        return csv_files
    
    def load_data_sample(self, file_path, sample_size=10000):
        """Load a sample of the data for exploration"""
        try:
            # First, get the total number of rows
            total_rows = sum(1 for line in open(file_path)) - 1  # -1 for header
            
            if total_rows <= sample_size:
                # If file is small enough, load all data
                df = pd.read_csv(file_path)
                print(f"  Loaded all {total_rows:,} rows")
            else:
                # Load a random sample
                skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                                  total_rows - sample_size, 
                                                  replace=False))
                df = pd.read_csv(file_path, skiprows=skip_rows)
                print(f"  Loaded sample of {len(df):,} rows from {total_rows:,} total rows")
            
            return df, total_rows
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            return None, 0
    
    def basic_info(self, df, dataset_name, table_name, total_rows):
        """Generate basic information about the dataset"""
        info = {
            'dataset': dataset_name,
            'table': table_name,
            'total_rows': total_rows,
            'sample_rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
        return info
    
    def numeric_analysis(self, df):
        """Analyze numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {}
        
        analysis = {}
        for col in numeric_cols:
            analysis[col] = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'zeros': (df[col] == 0).sum(),
                'negative': (df[col] < 0).sum()
            }
        return analysis
    
    def categorical_analysis(self, df, max_categories=20):
        """Analyze categorical columns"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            return {}
        
        analysis = {}
        for col in categorical_cols:
            unique_count = df[col].nunique()
            value_counts = df[col].value_counts()
            
            analysis[col] = {
                'unique_count': unique_count,
                'most_frequent': value_counts.head(max_categories).to_dict(),
                'least_frequent': value_counts.tail(5).to_dict() if unique_count > 5 else {},
                'mode': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
            }
        return analysis
    
    def datetime_analysis(self, df):
        """Try to identify and analyze datetime columns"""
        datetime_analysis = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    sample = df[col].dropna().head(1000)
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() > len(sample) * 0.8:  # 80% success rate
                        # This looks like a datetime column
                        full_parsed = pd.to_datetime(df[col], errors='coerce')
                        datetime_analysis[col] = {
                            'min_date': full_parsed.min(),
                            'max_date': full_parsed.max(),
                            'date_range_days': (full_parsed.max() - full_parsed.min()).days,
                            'null_dates': full_parsed.isna().sum(),
                            'unique_dates': full_parsed.nunique()
                        }
                except:
                    continue
        
        return datetime_analysis
    
    def create_visualizations(self, df, dataset_name, table_name):
        """Create visualizations for the data"""
        viz_dir = self.output_dir / f"{dataset_name}_{table_name}_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Numeric columns distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    df[col].hist(bins=50, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'numeric_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title(f'Correlation Matrix - {dataset_name} - {table_name}')
            plt.tight_layout()
            plt.savefig(viz_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Categorical columns (top categories)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:4]:  # Limit to first 4 categorical columns
                if df[col].nunique() <= 20:  # Only plot if reasonable number of categories
                    plt.figure(figsize=(12, 6))
                    value_counts = df[col].value_counts().head(15)
                    value_counts.plot(kind='bar')
                    plt.title(f'Top Categories in {col}')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(viz_dir / f'{col}_categories.png', dpi=300, bbox_inches='tight')
                    plt.close()
        
        return viz_dir
    
    def generate_report(self, info, numeric_analysis, categorical_analysis, datetime_analysis, viz_dir):
        """Generate a comprehensive text report"""
        report_path = self.output_dir / f"{info['dataset']}_{info['table']}_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"DATA EXPLORATION REPORT\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Dataset: {info['dataset']}\n")
            f.write(f"Table: {info['table']}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic Information
            f.write(f"BASIC INFORMATION\n")
            f.write(f"{'-'*20}\n")
            f.write(f"Total Rows: {info['total_rows']:,}\n")
            f.write(f"Sample Rows: {info['sample_rows']:,}\n")
            f.write(f"Columns: {info['columns']}\n")
            f.write(f"Memory Usage: {info['memory_usage_mb']:.2f} MB\n")
            f.write(f"Duplicate Rows: {info['duplicate_rows']:,}\n\n")
            
            # Column Information
            f.write(f"COLUMN INFORMATION\n")
            f.write(f"{'-'*20}\n")
            for col, dtype in info['data_types'].items():
                missing = info['missing_values'][col]
                missing_pct = (missing / info['sample_rows']) * 100
                f.write(f"{col:30} | {str(dtype):15} | Missing: {missing:,} ({missing_pct:.1f}%)\n")
            f.write("\n")
            
            # Numeric Analysis
            if numeric_analysis:
                f.write(f"NUMERIC ANALYSIS\n")
                f.write(f"{'-'*20}\n")
                for col, stats in numeric_analysis.items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  Count: {stats['count']:,}\n")
                    f.write(f"  Mean: {stats['mean']:.2f}\n")
                    f.write(f"  Median: {stats['median']:.2f}\n")
                    f.write(f"  Std Dev: {stats['std']:.2f}\n")
                    f.write(f"  Min: {stats['min']:.2f}\n")
                    f.write(f"  Max: {stats['max']:.2f}\n")
                    f.write(f"  Q25: {stats['q25']:.2f}\n")
                    f.write(f"  Q75: {stats['q75']:.2f}\n")
                    f.write(f"  Skewness: {stats['skewness']:.2f}\n")
                    f.write(f"  Zeros: {stats['zeros']:,}\n")
                    f.write(f"  Negative: {stats['negative']:,}\n")
                f.write("\n")
            
            # Categorical Analysis
            if categorical_analysis:
                f.write(f"CATEGORICAL ANALYSIS\n")
                f.write(f"{'-'*20}\n")
                for col, stats in categorical_analysis.items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  Unique Values: {stats['unique_count']:,}\n")
                    f.write(f"  Mode: {stats['mode']}\n")
                    f.write(f"  Top Values:\n")
                    for value, count in list(stats['most_frequent'].items())[:10]:
                        f.write(f"    {value}: {count:,}\n")
                f.write("\n")
            
            # Datetime Analysis
            if datetime_analysis:
                f.write(f"DATETIME ANALYSIS\n")
                f.write(f"{'-'*20}\n")
                for col, stats in datetime_analysis.items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  Date Range: {stats['min_date']} to {stats['max_date']}\n")
                    f.write(f"  Range (days): {stats['date_range_days']:,}\n")
                    f.write(f"  Unique Dates: {stats['unique_dates']:,}\n")
                    f.write(f"  Null Dates: {stats['null_dates']:,}\n")
                f.write("\n")
            
            f.write(f"VISUALIZATIONS\n")
            f.write(f"{'-'*20}\n")
            f.write(f"Visualizations saved to: {viz_dir}\n")
        
        return report_path
    
    def explore_dataset(self, csv_info, sample_size=10000):
        """Explore a single dataset"""
        print(f"\nExploring {csv_info['dataset']} - {csv_info['table']}...")
        
        # Load data sample
        df, total_rows = self.load_data_sample(csv_info['path'], sample_size)
        if df is None:
            return
        
        # Basic information
        info = self.basic_info(df, csv_info['dataset'], csv_info['table'], total_rows)
        
        # Detailed analysis
        print("  Analyzing numeric columns...")
        numeric_analysis = self.numeric_analysis(df)
        
        print("  Analyzing categorical columns...")
        categorical_analysis = self.categorical_analysis(df)
        
        print("  Analyzing datetime columns...")
        datetime_analysis = self.datetime_analysis(df)
        
        print("  Creating visualizations...")
        viz_dir = self.create_visualizations(df, csv_info['dataset'], csv_info['table'])
        
        print("  Generating report...")
        report_path = self.generate_report(info, numeric_analysis, categorical_analysis, 
                                         datetime_analysis, viz_dir)
        
        print(f"  Report saved: {report_path}")
        return info
    
    def run_exploration(self, sample_size=10000):
        """Run exploration on all CSV files"""
        print("🔍 Starting Data Exploration...")
        print(f"Output directory: {self.output_dir}")
        
        csv_files = self.find_csv_files()
        if not csv_files:
            print("No CSV files found in subdirectories!")
            return
        
        print(f"Found {len(csv_files)} CSV file(s) to explore:")
        for csv_info in csv_files:
            print(f"  - {csv_info['dataset']}/{csv_info['table']}.csv")
        
        # Create summary
        all_info = []
        for csv_info in csv_files:
            info = self.explore_dataset(csv_info, sample_size)
            if info:
                all_info.append(info)
        
        # Generate summary report
        self.generate_summary_report(all_info)
        
        print(f"\n✅ Exploration complete! Results saved to: {self.output_dir}")
    
    def generate_summary_report(self, all_info):
        """Generate a summary report across all datasets"""
        summary_path = self.output_dir / "summary_report.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"DATA EXPLORATION SUMMARY\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Datasets: {len(all_info)}\n\n")
            
            for info in all_info:
                f.write(f"Dataset: {info['dataset']} - {info['table']}\n")
                f.write(f"  Rows: {info['total_rows']:,}\n")
                f.write(f"  Columns: {info['columns']}\n")
                f.write(f"  Memory: {info['memory_usage_mb']:.2f} MB\n")
                f.write(f"  Duplicates: {info['duplicate_rows']:,}\n\n")


def main():
    parser = argparse.ArgumentParser(description='Explore CSV data files')
    parser.add_argument('--csv-dir', default='outputs/csv_data', 
                       help='Directory containing CSV subdirectories (default: outputs/csv_data)')
    parser.add_argument('--output-dir', default='outputs/exploration',
                       help='Directory to save exploration results (default: outputs/exploration)')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Sample size for large files (default: 10000)')
    
    args = parser.parse_args()
    
    explorer = DataExplorer(args.csv_dir, args.output_dir)
    explorer.run_exploration(args.sample_size)


if __name__ == "__main__":
    main()
