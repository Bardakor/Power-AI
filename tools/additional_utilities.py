#!/usr/bin/env python3
"""Additional Utilities - Supporting functionality in 300 lines"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sqlite3
import time
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

class PowerAIUtilities:
    def __init__(self, base_dir="outputs"):
        self.base_dir = Path(base_dir)
        self.config_file = self.base_dir / "config.json"
        self.setup_logging()
        self.load_config()
        
    def setup_logging(self):
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "power_ai.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self):
        default_config = {
            "data_refresh_interval": 300,  # seconds
            "anomaly_threshold": 0.1,
            "prediction_horizon": 24,  # hours
            "alert_thresholds": {
                "high_load": 90,
                "low_voltage": 220,
                "high_voltage": 240,
                "power_factor": 0.85
            },
            "dashboard_settings": {
                "auto_refresh": True,
                "theme": "bootstrap",
                "charts_per_page": 8
            }
        }
        
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update_config(self, **kwargs):
        self.config.update(kwargs)
        self.save_config()
        self.logger.info("Configuration updated")
    
    def real_time_data_simulator(self, base_data, duration_hours=1, frequency_seconds=30):
        """Simulate real-time data stream for testing"""
        if base_data.empty:
            return
        
        base_stats = {
            'load_mean': base_data['ups_load'].mean(),
            'load_std': base_data['ups_load'].std(),
            'voltage_mean': base_data[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].mean().mean(),
            'voltage_std': base_data[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].std().mean()
        }
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            # Generate synthetic data point
            new_data = {
                'timestamp': datetime.now(),
                'ups_load': np.random.normal(base_stats['load_mean'], base_stats['load_std']),
                'ups_va_out': np.random.normal(base_stats['voltage_mean'], base_stats['voltage_std']),
                'ups_vb_out': np.random.normal(base_stats['voltage_mean'], base_stats['voltage_std']),
                'ups_vc_out': np.random.normal(base_stats['voltage_mean'], base_stats['voltage_std'])
            }
            
            # Add some realistic variations
            hour = new_data['timestamp'].hour
            if 9 <= hour <= 17:  # Business hours
                new_data['ups_load'] *= 1.2
            elif 22 <= hour or hour <= 6:  # Night hours
                new_data['ups_load'] *= 0.7
            
            yield new_data
            time.sleep(frequency_seconds)
            
    def export_analysis_results(self, results, format='json'):
        """Export analysis results in various formats"""
        export_dir = self.base_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = export_dir / f"analysis_results_{timestamp}.json"
            with open(filename, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2, default=str)
                
        elif format == 'csv':
            for dataset_name, data in results.items():
                if 'future' in data and hasattr(data['future'], 'to_csv'):
                    filename = export_dir / f"forecast_{dataset_name}_{timestamp}.csv"
                    data['future'].to_csv(filename)
                    
        elif format == 'excel':
            filename = export_dir / f"power_ai_report_{timestamp}.xlsx"
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for dataset_name, data in results.items():
                    if 'future' in data and hasattr(data['future'], 'to_excel'):
                        data['future'].to_excel(writer, sheet_name=f"Forecast_{dataset_name[:15]}")
        
        self.logger.info(f"Results exported to {filename}")
        return filename
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return obj
    
    def import_external_data(self, file_path, format='csv'):
        """Import data from external sources"""
        file_path = Path(file_path)
        
        try:
            if format == 'csv':
                df = pd.read_csv(file_path)
            elif format == 'excel':
                df = pd.read_excel(file_path)
            elif format == 'sqlite':
                conn = sqlite3.connect(file_path)
                df = pd.read_sql_query("SELECT * FROM leituras", conn)
                conn.close()
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Successfully imported {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to import data from {file_path}: {e}")
            return None
    
    def performance_monitor(self, func):
        """Decorator to monitor function performance"""
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                self.logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                self.logger.error(f"{func.__name__} failed after {duration:.2f} seconds: {e}")
                raise
        return wrapper
    
    def alert_system(self, data, alert_type="all"):
        """Generate alerts based on thresholds"""
        alerts = []
        thresholds = self.config.get("alert_thresholds", {})
        
        # Default thresholds if not in config
        default_thresholds = {
            "high_load": 90,
            "low_voltage": 220,
            "high_voltage": 240,
            "power_factor": 0.85
        }
        
        # Merge defaults with config
        for key, value in default_thresholds.items():
            if key not in thresholds:
                thresholds[key] = value
        
        if alert_type in ["all", "load"]:
            high_load_data = data[data['ups_load'] > thresholds['high_load']]
            if not high_load_data.empty:
                alerts.append({
                    'type': 'HIGH_LOAD',
                    'severity': 'WARNING',
                    'message': f"High UPS load detected: {high_load_data['ups_load'].max():.1f}%",
                    'count': len(high_load_data),
                    'timestamp': datetime.now().isoformat()
                })
        
        if alert_type in ["all", "voltage"]:
            voltage_cols = ['ups_va_out', 'ups_vb_out', 'ups_vc_out']
            available_voltage_cols = [col for col in voltage_cols if col in data.columns]
            
            if available_voltage_cols:
                avg_voltage = data[available_voltage_cols].mean(axis=1)
                
                low_voltage_data = avg_voltage[avg_voltage < thresholds['low_voltage']]
                high_voltage_data = avg_voltage[avg_voltage > thresholds['high_voltage']]
                
                if not low_voltage_data.empty:
                    alerts.append({
                        'type': 'LOW_VOLTAGE',
                        'severity': 'CRITICAL',
                        'message': f"Low voltage detected: {low_voltage_data.min():.1f}V",
                        'count': len(low_voltage_data),
                        'timestamp': datetime.now().isoformat()
                    })
                
                if not high_voltage_data.empty:
                    alerts.append({
                        'type': 'HIGH_VOLTAGE',
                        'severity': 'WARNING',
                        'message': f"High voltage detected: {high_voltage_data.max():.1f}V",
                        'count': len(high_voltage_data),
                        'timestamp': datetime.now().isoformat()
                    })
        
        return alerts
    
    def data_quality_check(self, df):
        """Perform data quality checks"""
        quality_report = {
            'total_records': len(df),
            'missing_data': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_ranges': {},
            'timestamp': datetime.now().isoformat()
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            quality_report['numeric_ranges'][col] = {
                'min': float(df[col].min()) if not df[col].empty else None,
                'max': float(df[col].max()) if not df[col].empty else None,
                'mean': float(df[col].mean()) if not df[col].empty else None,
                'std': float(df[col].std()) if not df[col].empty else None
            }
        
        return quality_report
    
    def optimize_data_storage(self, df, compression='gzip'):
        """Optimize DataFrame for storage"""
        optimized_df = df.copy()
        
        # Convert float64 to float32 where possible
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                optimized_df[col] = optimized_df[col].astype(np.float32)
        
        # Convert int64 to smaller int types where possible
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
        
        return optimized_df
    
    def generate_summary_report(self, results):
        """Generate comprehensive summary report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_overview': {
                'total_datasets': len(results),
                'datasets': list(results.keys())
            },
            'performance_metrics': {},
            'alerts_summary': {},
            'recommendations': []
        }
        
        for dataset_name, data in results.items():
            if 'consumption' in data:
                report['performance_metrics'][dataset_name] = {
                    'model_accuracy': data['consumption'].get('r2', 0),
                    'prediction_error': data['consumption'].get('mae', 0),
                    'anomaly_rate': data.get('anomalies', 0)
                }
            
            if 'optimizations' in data:
                report['recommendations'].extend(data['optimizations'])
        
        return report

class DataStreamManager:
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.data_buffer = []
        
    def add_data_point(self, data_point):
        self.data_buffer.append(data_point)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
    
    def get_recent_data(self, num_points=100):
        return self.data_buffer[-num_points:] if self.data_buffer else []
    
    def clear_buffer(self):
        self.data_buffer.clear()

def main():
    utilities = PowerAIUtilities()
    print("🛠️ Power AI Utilities initialized")
    
    # Example usage
    sample_config = {
        "alert_thresholds": {
            "high_load": 85,
            "low_voltage": 215,
            "high_voltage": 245
        }
    }
    utilities.update_config(**sample_config)
    print("✅ Configuration updated")

if __name__ == "__main__":
    main()
