#!/usr/bin/env python3
"""
parser.py

Find all SQLite databases named leituras*.db in a directory,
dump every table in each to a CSV file for easy pandas loading.
CSVs will be organized into subdirectories named after their source database.

Usage:
    python parser.py --input-dir /path/to/dbs --output-dir /path/to/csvs
"""

import os
import glob
import argparse
import sqlite3
import pandas as pd

def export_db_tables_to_csv(db_path: str, output_dir: str):
    """
    Connect to the SQLite database at db_path, read all tables,
    and write each as a CSV into output_dir.
    The output_dir is expected to be a directory specific to this db.
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
            print(f"  -> Exporting table '{table}'...")
            # Read entire table into a DataFrame
            # Ensure table names with spaces or special characters are handled by quoting.
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
            # Build a safe CSV filename: <table>.csv (output_dir is already db-specific)
            csv_filename = f"{table}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            # Write out without the pandas index
            df.to_csv(csv_path, index=False)
            print(f"     ↳ Wrote {len(df):,} rows to {csv_path}")

    conn.close()

def parser():
    parser = argparse.ArgumentParser(
        description="Export all tables from leituras*.db SQLite files to CSV, organizing them into subdirectories."
    )
    parser.add_argument(
        "--input-dir", "-i",
        default=".",
        help="Base directory containing the 'data' subdirectory with leituras*.db files (default: current directory, expecting ./data/leituras*.db)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="csv_output",
        help="Directory where database-specific subdirectories and CSVs will be written (default: ./csv_output)"
    )
    args = parser.parse_args()

    # Ensure the main output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct the pattern to find .db files within the 'data' subdirectory of the input_dir
    # If input_dir is '.', pattern becomes './data/leituras*.db'
    pattern = os.path.join(args.input_dir, "data", "leituras*.db")
    db_files = glob.glob(pattern)
    
    if not db_files:
        print(f"[!] No files matching {pattern}")
        print(f"    Please ensure your .db files are in a 'data' subdirectory relative to your input directory.")
        print(f"    Current input directory: {os.path.abspath(args.input_dir)}")
        return

    print(f"Found {len(db_files)} database(s) to process:")
    for db_file_path in db_files: # Renamed 'db' to 'db_file_path' for clarity
        print(f" - {db_file_path}")
    print()

    # Process each database
    for db_file_path in db_files:
        db_name_for_subdir = os.path.splitext(os.path.basename(db_file_path))[0]
        print(f"Processing '{db_name_for_subdir}'...")
        
        # Create a subdirectory for this specific database's CSVs
        current_db_output_dir = os.path.join(args.output_dir, db_name_for_subdir)
        os.makedirs(current_db_output_dir, exist_ok=True)
        print(f"  Outputting CSVs to: {os.path.abspath(current_db_output_dir)}")
        
        try:
            export_db_tables_to_csv(db_file_path, current_db_output_dir)
        except Exception as e:
            print(f"  [ERROR] Failed to process {db_file_path}: {e}")
        print()

    print("Done! CSV files are in subdirectories under:", os.path.abspath(args.output_dir))


