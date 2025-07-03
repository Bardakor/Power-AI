#!/usr/bin/env python3
"""
mini_parser.py

Find all SQLite databases named *.db in the data directory,
dump every table in each to mini CSV files (5MB max each) for easy handling.
CSVs will be organized into subdirectories with "mini_" prefix.

Usage:
    python mini_parser.py --input-dir /path/to/dbs --output-dir /path/to/mini_csvs
"""

import os
import glob
import argparse
import sqlite3
import pandas as pd
import math

def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def export_db_tables_to_mini_csv(db_path: str, output_dir: str, max_size_mb=5):
    """
    Connect to the SQLite database at db_path, read all tables,
    and write each as mini CSV files (5MB max) into output_dir.
    """
    db_name = os.path.splitext(os.path.basename(db_path))[0]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get list of user tables (skip SQLite internal tables)
    cursor.execute("""
        SELECT name 
          FROM sqlite_master 
         WHERE type='table' 
           AND name NOT LIKE 'sqlite_%';
    """)
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        print(f"  [!] No tables found in {db_name}, skipping.")
    else:
        for table in tables:
            print(f"  -> Exporting table '{table}' as mini files...")
            
            # Get total row count first
            count_query = f'SELECT COUNT(*) FROM "{table}"'
            cursor.execute(count_query)
            total_rows = cursor.fetchone()[0]
            
            if total_rows == 0:
                print(f"     ↳ Table '{table}' is empty, skipping.")
                continue
            
            # Read a small sample to estimate row size
            sample_df = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT 1000', conn)
            
            if len(sample_df) == 0:
                continue
                
            # Estimate bytes per row by creating a temporary CSV
            temp_csv = os.path.join(output_dir, "temp_sample.csv")
            sample_df.to_csv(temp_csv, index=False)
            sample_size_mb = get_file_size_mb(temp_csv)
            bytes_per_row = (sample_size_mb * 1024 * 1024) / len(sample_df)
            
            # Clean up temp file
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            
            # Calculate rows per 5MB file
            max_bytes = max_size_mb * 1024 * 1024
            rows_per_file = max(1000, int(max_bytes / bytes_per_row * 0.8))  # 80% safety margin
            
            # Calculate number of files needed
            num_files = math.ceil(total_rows / rows_per_file)
            
            print(f"     ↳ Splitting {total_rows:,} rows into {num_files} mini files (~{rows_per_file:,} rows each)")
            
            # Export in chunks
            for file_num in range(num_files):
                offset = file_num * rows_per_file
                
                # Read chunk from database
                chunk_query = f'SELECT * FROM "{table}" LIMIT {rows_per_file} OFFSET {offset}'
                chunk_df = pd.read_sql_query(chunk_query, conn)
                
                if len(chunk_df) == 0:
                    break
                
                # Create mini CSV filename
                if num_files == 1:
                    csv_filename = f"mini_{table}.csv"
                else:
                    csv_filename = f"mini_{table}_part{file_num + 1:03d}.csv"
                
                csv_path = os.path.join(output_dir, csv_filename)
                chunk_df.to_csv(csv_path, index=False)
                
                actual_size = get_file_size_mb(csv_path)
                print(f"       → {csv_filename}: {len(chunk_df):,} rows, {actual_size:.1f}MB")

    conn.close()

def main():
    parser = argparse.ArgumentParser(
        description="Export all tables from *.db SQLite files to mini CSV files (5MB max each), organizing them into subdirectories."
    )
    parser.add_argument(
        "--input-dir", "-i",
        default=".",
        help="Base directory containing the 'data' subdirectory with *.db files (default: current directory)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs/mini_csv_data",
        help="Directory where mini CSV files will be written (default: ./outputs/mini_csv_data)"
    )
    parser.add_argument(
        "--max-size", "-s",
        type=float,
        default=5.0,
        help="Maximum size per CSV file in MB (default: 5.0)"
    )
    args = parser.parse_args()

    # Ensure the main output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .db files in the data subdirectory
    pattern = os.path.join(args.input_dir, "data", "*.db")
    db_files = glob.glob(pattern)
    
    if not db_files:
        print(f"[!] No .db files found matching {pattern}")
        print(f"    Please ensure your .db files are in the 'data' subdirectory.")
        print(f"    Current input directory: {os.path.abspath(args.input_dir)}")
        return

    print(f"Found {len(db_files)} database(s) to process:")
    for db_file_path in db_files:
        print(f" - {db_file_path}")
    print()

    # Process each database
    for db_file_path in db_files:
        db_name_for_subdir = os.path.splitext(os.path.basename(db_file_path))[0]
        print(f"Processing '{db_name_for_subdir}'...")
        
        # Create a subdirectory for this specific database's mini CSVs
        current_db_output_dir = os.path.join(args.output_dir, f"mini_{db_name_for_subdir}")
        os.makedirs(current_db_output_dir, exist_ok=True)
        print(f"  Outputting mini CSVs to: {os.path.abspath(current_db_output_dir)}")
        
        try:
            export_db_tables_to_mini_csv(db_file_path, current_db_output_dir, args.max_size)
        except Exception as e:
            print(f"  [ERROR] Failed to process {db_file_path}: {e}")
        print()

    print("Done! Mini CSV files are in subdirectories under:", os.path.abspath(args.output_dir))

if __name__ == "__main__":
    main() 