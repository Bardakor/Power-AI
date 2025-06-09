#!/usr/bin/env python3
"""
Advanced Power System Analysis

This script provides specialized analysis for power infrastructure monitoring,
focusing on UPS performance, power quality, and energy efficiency metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class PowerSystemAnalyzer:
    def __init__(self, csv_output_dir="outputs/csv_data", output_dir="outputs/power_analysis"):
        self.csv_output_dir = Path(csv_output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("Set2")
        
    def load_datasets(self, sample_size=20000):
        """Load both datasets with larger samples for better analysis"""
        datasets = {}
        
        for subdir in self.csv_output_dir.iterdir():
            if subdir.is_dir():
                csv_file = subdir / 'leituras.csv'
                if csv_file.exists():
                    print(f"Loading {subdir.name}...")
                    
                    # Get total rows
                    total_rows = sum(1 for line in open(csv_file)) - 1
                    
                    if total_rows <= sample_size:
                        df = pd.read_csv(csv_file)
                    else:
                        # Load random sample
                        skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                                          total_rows - sample_size, 
                                                          replace=False))
                        df = pd.read_csv(csv_file, skiprows=skip_rows)
                    
                    # Parse datetime
                    df['datetime'] = pd.to_datetime(df['data_hora'])
                    df = df.set_index('datetime').sort_index()
                    
                    datasets[subdir.name] = {
                        'data': df,
                        'total_rows': total_rows,
                        'sample_size': len(df)
                    }
                    
                    print(f"  Loaded {len(df):,} rows from {total_rows:,} total")
        
        return datasets
    
    def analyze_ups_performance(self, datasets):
        """Analyze UPS system performance and reliability"""
        print("\n🔋 UPS Performance Analysis")
        print("=" * 40)
        
        ups_analysis = {}
        
        for name, dataset in datasets.items():
            df = dataset['data']
            
            # UPS key metrics
            ups_metrics = {
                'load_avg': df['ups_load'].mean(),
                'load_max': df['ups_load'].max(),
                'load_min': df['ups_load'].min(),
                'load_std': df['ups_load'].std(),
                'efficiency': self.calculate_ups_efficiency(df),
                'voltage_stability': self.analyze_voltage_stability(df),
                'frequency_stability': self.analyze_frequency_stability(df),
                'battery_health': self.analyze_battery_health(df)
            }
            
            ups_analysis[name] = ups_metrics
            
            print(f"\n{name}:")
            print(f"  Load: {ups_metrics['load_avg']:.1f}% avg, {ups_metrics['load_max']:.1f}% max")
            print(f"  Efficiency: {ups_metrics['efficiency']:.2f}%")
            print(f"  Voltage Stability: {ups_metrics['voltage_stability']:.3f}")
            print(f"  Frequency Stability: {ups_metrics['frequency_stability']:.3f}")
        
        return ups_analysis
    
    def calculate_ups_efficiency(self, df):
        """Calculate UPS efficiency based on input/output power"""
        try:
            # Calculate total input and output power
            input_power = df[['ups_pa', 'ups_pb', 'ups_pc']].sum(axis=1)
            output_power_cols = [col for col in df.columns if 'ups_load' in col and col.endswith('_out')]
            
            if len(output_power_cols) > 0:
                output_power = df[output_power_cols].sum(axis=1)
            else:
                # Estimate output power from load percentage and VA rating
                estimated_va = df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].mean(axis=1) * 3
                output_power = estimated_va * (df['ups_load'] / 100)
            
            efficiency = (output_power / input_power * 100).mean()
            return efficiency if not np.isnan(efficiency) else 0
        except:
            return 0
    
    def analyze_voltage_stability(self, df):
        """Calculate voltage stability coefficient of variation"""
        try:
            voltage_cols = ['ups_va_out', 'ups_vb_out', 'ups_vc_out']
            voltage_data = df[voltage_cols].mean(axis=1)
            cv = voltage_data.std() / voltage_data.mean()
            return cv
        except:
            return 0
    
    def analyze_frequency_stability(self, df):
        """Calculate frequency stability"""
        try:
            freq_std = df['ups_hz_out'].std()
            freq_mean = df['ups_hz_out'].mean()
            return freq_std / freq_mean if freq_mean > 0 else 0
        except:
            return 0
    
    def analyze_battery_health(self, df):
        """Analyze battery charging/discharging patterns"""
        try:
            charge_cycles = 0
            discharge_events = 0
            
            if 'ups_bat_i_charge' in df.columns and 'ups_bat_i_discharge' in df.columns:
                charge_current = df['ups_bat_i_charge']
                discharge_current = df['ups_bat_i_discharge']
                
                # Count charging cycles
                charging = charge_current > 0
                charge_cycles = charging.diff().sum() / 2
                
                # Count discharge events
                discharging = discharge_current > 0
                discharge_events = discharging.sum()
            
            return {'charge_cycles': charge_cycles, 'discharge_events': discharge_events}
        except:
            return {'charge_cycles': 0, 'discharge_events': 0}
    
    def analyze_power_quality(self, datasets):
        """Analyze power quality metrics"""
        print("\n⚡ Power Quality Analysis")
        print("=" * 40)
        
        pq_analysis = {}
        
        for name, dataset in datasets.items():
            df = dataset['data']
            
            # Power factor analysis
            pf_cols = [col for col in df.columns if 'fp' in col.lower()]
            thd_cols = [col for col in df.columns if 'thd' in col.lower()]
            
            metrics = {
                'power_factor_avg': df[pf_cols].mean().mean() if pf_cols else 0,
                'power_factor_min': df[pf_cols].min().min() if pf_cols else 0,
                'voltage_imbalance': self.calculate_voltage_imbalance(df),
                'current_imbalance': self.calculate_current_imbalance(df),
                'harmonic_distortion': df[thd_cols].mean().mean() if thd_cols else 0
            }
            
            pq_analysis[name] = metrics
            
            print(f"\n{name}:")
            print(f"  Power Factor: {metrics['power_factor_avg']:.3f} avg")
            print(f"  Voltage Imbalance: {metrics['voltage_imbalance']:.2f}%")
            print(f"  Current Imbalance: {metrics['current_imbalance']:.2f}%")
        
        return pq_analysis
    
    def calculate_voltage_imbalance(self, df):
        """Calculate three-phase voltage imbalance"""
        try:
            # Use UPS output voltages
            va = df['ups_va_out']
            vb = df['ups_vb_out'] 
            vc = df['ups_vc_out']
            
            v_avg = (va + vb + vc) / 3
            max_deviation = np.maximum(np.maximum(np.abs(va - v_avg), 
                                                 np.abs(vb - v_avg)), 
                                     np.abs(vc - v_avg))
            imbalance = (max_deviation / v_avg * 100).mean()
            return imbalance if not np.isnan(imbalance) else 0
        except:
            return 0
    
    def calculate_current_imbalance(self, df):
        """Calculate three-phase current imbalance"""
        try:
            ia = df['ups_ia_out']
            ib = df['ups_ib_out']
            ic = df['ups_ic_out']
            
            i_avg = (ia + ib + ic) / 3
            max_deviation = np.maximum(np.maximum(np.abs(ia - i_avg), 
                                                 np.abs(ib - i_avg)), 
                                     np.abs(ic - i_avg))
            imbalance = (max_deviation / i_avg * 100).mean()
            return imbalance if not np.isnan(imbalance) else 0
        except:
            return 0
    
    def analyze_energy_consumption(self, datasets):
        """Analyze energy consumption patterns"""
        print("\n📊 Energy Consumption Analysis")
        print("=" * 40)
        
        energy_analysis = {}
        
        for name, dataset in datasets.items():
            df = dataset['data']
            
            # Find energy columns
            kwh_cols = [col for col in df.columns if 'kwh' in col.lower()]
            pdu_kwh_cols = [col for col in kwh_cols if 'pdu' in col.lower()]
            met_kwh_cols = [col for col in kwh_cols if 'met' in col.lower()]
            
            # Calculate daily energy consumption if possible
            daily_consumption = self.calculate_daily_consumption(df, kwh_cols)
            
            metrics = {
                'total_kwh_meters': len(met_kwh_cols),
                'total_pdu_meters': len(pdu_kwh_cols),
                'daily_consumption_avg': daily_consumption.get('avg', 0),
                'daily_consumption_peak': daily_consumption.get('peak', 0),
                'consumption_trend': daily_consumption.get('trend', 0)
            }
            
            energy_analysis[name] = metrics
            
            print(f"\n{name}:")
            print(f"  Energy meters: {metrics['total_kwh_meters']} main, {metrics['total_pdu_meters']} PDU")
            print(f"  Daily consumption: {metrics['daily_consumption_avg']:.1f} kWh avg")
            print(f"  Peak consumption: {metrics['daily_consumption_peak']:.1f} kWh")
        
        return energy_analysis
    
    def calculate_daily_consumption(self, df, kwh_cols):
        """Calculate daily energy consumption patterns"""
        try:
            if not kwh_cols:
                return {'avg': 0, 'peak': 0, 'trend': 0}
            
            # Take main energy meter (first kwh column)
            energy_data = df[kwh_cols[0]]
            
            # Resample to daily and calculate differences
            daily_energy = energy_data.resample('D').last()
            daily_consumption = daily_energy.diff().dropna()
            
            return {
                'avg': daily_consumption.mean(),
                'peak': daily_consumption.max(),
                'trend': np.polyfit(range(len(daily_consumption)), daily_consumption, 1)[0]
            }
        except:
            return {'avg': 0, 'peak': 0, 'trend': 0}
    
    def create_dashboard_visualizations(self, datasets, ups_analysis, pq_analysis, energy_analysis):
        """Create comprehensive dashboard visualizations"""
        print("\n📈 Creating Dashboard Visualizations...")
        
        # 1. UPS Performance Dashboard
        self.create_ups_dashboard(datasets, ups_analysis)
        
        # 2. Power Quality Dashboard  
        self.create_power_quality_dashboard(datasets, pq_analysis)
        
        # 3. Energy Consumption Dashboard
        self.create_energy_dashboard(datasets, energy_analysis)
        
        # 4. System Comparison Dashboard
        self.create_comparison_dashboard(datasets, ups_analysis, pq_analysis)
    
    def create_ups_dashboard(self, datasets, ups_analysis):
        """Create UPS performance dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('UPS Performance Dashboard', fontsize=16, fontweight='bold')
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Load comparison
        loads = [analysis['load_avg'] for analysis in ups_analysis.values()]
        names = list(ups_analysis.keys())
        
        axes[0,0].bar(range(len(names)), loads, color=colors[:len(names)])
        axes[0,0].set_title('Average UPS Load')
        axes[0,0].set_ylabel('Load (%)')
        axes[0,0].set_xticks(range(len(names)))
        axes[0,0].set_xticklabels([name.split('_')[1][:6] for name in names], rotation=45)
        
        # Efficiency comparison
        efficiencies = [analysis['efficiency'] for analysis in ups_analysis.values()]
        axes[0,1].bar(range(len(names)), efficiencies, color=colors[:len(names)])
        axes[0,1].set_title('UPS Efficiency')
        axes[0,1].set_ylabel('Efficiency (%)')
        axes[0,1].set_xticks(range(len(names)))
        axes[0,1].set_xticklabels([name.split('_')[1][:6] for name in names], rotation=45)
        
        # Load distribution for first dataset
        first_dataset = list(datasets.values())[0]['data']
        axes[0,2].hist(first_dataset['ups_load'], bins=50, alpha=0.7, color=colors[0])
        axes[0,2].set_title('UPS Load Distribution')
        axes[0,2].set_xlabel('Load (%)')
        axes[0,2].set_ylabel('Frequency')
        
        # Load over time for both datasets
        for i, (name, dataset) in enumerate(datasets.items()):
            df = dataset['data']
            # Resample to hourly for cleaner plot
            hourly_load = df['ups_load'].resample('H').mean()
            axes[1,0].plot(hourly_load.index, hourly_load.values, 
                          label=name.split('_')[1][:6], alpha=0.8, color=colors[i])
        
        axes[1,0].set_title('UPS Load Over Time')
        axes[1,0].set_ylabel('Load (%)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Voltage stability
        voltage_stab = [analysis['voltage_stability'] for analysis in ups_analysis.values()]
        axes[1,1].bar(range(len(names)), voltage_stab, color=colors[:len(names)])
        axes[1,1].set_title('Voltage Stability (CV)')
        axes[1,1].set_ylabel('Coefficient of Variation')
        axes[1,1].set_xticks(range(len(names)))
        axes[1,1].set_xticklabels([name.split('_')[1][:6] for name in names], rotation=45)
        
        # Frequency stability
        freq_stab = [analysis['frequency_stability'] for analysis in ups_analysis.values()]
        axes[1,2].bar(range(len(names)), freq_stab, color=colors[:len(names)])
        axes[1,2].set_title('Frequency Stability')
        axes[1,2].set_ylabel('Relative Standard Deviation')
        axes[1,2].set_xticks(range(len(names)))
        axes[1,2].set_xticklabels([name.split('_')[1][:6] for name in names], rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ups_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_power_quality_dashboard(self, datasets, pq_analysis):
        """Create power quality dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Power Quality Dashboard', fontsize=16, fontweight='bold')
        
        colors = ['#2E86AB', '#A23B72']
        names = list(pq_analysis.keys())
        
        # Power factor comparison
        pf_values = [analysis['power_factor_avg'] for analysis in pq_analysis.values()]
        axes[0,0].bar(range(len(names)), pf_values, color=colors[:len(names)])
        axes[0,0].set_title('Average Power Factor')
        axes[0,0].set_ylabel('Power Factor')
        axes[0,0].set_xticks(range(len(names)))
        axes[0,0].set_xticklabels([name.split('_')[1][:6] for name in names], rotation=45)
        axes[0,0].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (0.9)')
        axes[0,0].legend()
        
        # Voltage imbalance
        v_imb = [analysis['voltage_imbalance'] for analysis in pq_analysis.values()]
        axes[0,1].bar(range(len(names)), v_imb, color=colors[:len(names)])
        axes[0,1].set_title('Voltage Imbalance')
        axes[0,1].set_ylabel('Imbalance (%)')
        axes[0,1].set_xticks(range(len(names)))
        axes[0,1].set_xticklabels([name.split('_')[1][:6] for name in names], rotation=45)
        axes[0,1].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Limit (2%)')
        axes[0,1].legend()
        
        # Voltage waveform (first dataset)
        first_dataset = list(datasets.values())[0]['data']
        sample_data = first_dataset.head(1000)  # First 1000 points
        axes[1,0].plot(sample_data.index, sample_data['ups_va_out'], alpha=0.7, label='Phase A')
        axes[1,0].plot(sample_data.index, sample_data['ups_vb_out'], alpha=0.7, label='Phase B')
        axes[1,0].plot(sample_data.index, sample_data['ups_vc_out'], alpha=0.7, label='Phase C')
        axes[1,0].set_title('Three-Phase Voltage (Sample)')
        axes[1,0].set_ylabel('Voltage (V)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Current imbalance
        c_imb = [analysis['current_imbalance'] for analysis in pq_analysis.values()]
        axes[1,1].bar(range(len(names)), c_imb, color=colors[:len(names)])
        axes[1,1].set_title('Current Imbalance')
        axes[1,1].set_ylabel('Imbalance (%)')
        axes[1,1].set_xticks(range(len(names)))
        axes[1,1].set_xticklabels([name.split('_')[1][:6] for name in names], rotation=45)
        axes[1,1].axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Limit (10%)')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'power_quality_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_energy_dashboard(self, datasets, energy_analysis):
        """Create energy consumption dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Energy Consumption Dashboard', fontsize=16, fontweight='bold')
        
        colors = ['#2E86AB', '#A23B72']
        names = list(energy_analysis.keys())
        
        # Daily consumption comparison
        daily_avg = [analysis['daily_consumption_avg'] for analysis in energy_analysis.values()]
        axes[0,0].bar(range(len(names)), daily_avg, color=colors[:len(names)])
        axes[0,0].set_title('Average Daily Consumption')
        axes[0,0].set_ylabel('kWh/day')
        axes[0,0].set_xticks(range(len(names)))
        axes[0,0].set_xticklabels([name.split('_')[1][:6] for name in names], rotation=45)
        
        # Peak consumption
        peak_consumption = [analysis['daily_consumption_peak'] for analysis in energy_analysis.values()]
        axes[0,1].bar(range(len(names)), peak_consumption, color=colors[:len(names)])
        axes[0,1].set_title('Peak Daily Consumption')
        axes[0,1].set_ylabel('kWh/day')
        axes[0,1].set_xticks(range(len(names)))
        axes[0,1].set_xticklabels([name.split('_')[1][:6] for name in names], rotation=45)
        
        # Energy trend over time (if data available)
        for i, (name, dataset) in enumerate(datasets.items()):
            df = dataset['data']
            kwh_cols = [col for col in df.columns if 'kwh' in col.lower() and 'met' in col.lower()]
            if kwh_cols:
                energy_data = df[kwh_cols[0]].resample('D').last()
                axes[1,0].plot(energy_data.index, energy_data.values, 
                              label=name.split('_')[1][:6], alpha=0.8, color=colors[i])
        
        axes[1,0].set_title('Cumulative Energy Consumption')
        axes[1,0].set_ylabel('kWh (Cumulative)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Energy meters count
        meter_counts = [analysis['total_kwh_meters'] for analysis in energy_analysis.values()]
        pdu_counts = [analysis['total_pdu_meters'] for analysis in energy_analysis.values()]
        
        x = np.arange(len(names))
        width = 0.35
        
        axes[1,1].bar(x - width/2, meter_counts, width, label='Main Meters', color=colors[0])
        axes[1,1].bar(x + width/2, pdu_counts, width, label='PDU Meters', color=colors[1])
        axes[1,1].set_title('Energy Monitoring Points')
        axes[1,1].set_ylabel('Number of Meters')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels([name.split('_')[1][:6] for name in names], rotation=45)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'energy_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_dashboard(self, datasets, ups_analysis, pq_analysis):
        """Create system comparison dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Performance Comparison', fontsize=16, fontweight='bold')
        
        names = list(ups_analysis.keys())
        short_names = [name.split('_')[1][:6] for name in names]
        colors = ['#2E86AB', '#A23B72']
        
        # Performance radar chart data
        metrics = ['Load', 'Efficiency', 'V Stability', 'F Stability', 'Power Factor']
        
        # Normalize metrics for radar chart
        loads = [analysis['load_avg']/100 for analysis in ups_analysis.values()]  # Normalize to 0-1
        efficiencies = [analysis['efficiency']/100 for analysis in ups_analysis.values()]
        v_stab = [1 - min(analysis['voltage_stability'], 1) for analysis in ups_analysis.values()]  # Invert (lower is better)
        f_stab = [1 - min(analysis['frequency_stability'], 1) for analysis in ups_analysis.values()]
        pf = [analysis['power_factor_avg'] for analysis in pq_analysis.values()]
        
        # Create comparison table
        comparison_data = {
            'Dataset': short_names,
            'UPS Load (%)': [f"{analysis['load_avg']:.1f}" for analysis in ups_analysis.values()],
            'Efficiency (%)': [f"{analysis['efficiency']:.1f}" for analysis in ups_analysis.values()],
            'V Imbalance (%)': [f"{analysis['voltage_imbalance']:.2f}" for analysis in pq_analysis.values()],
            'Power Factor': [f"{analysis['power_factor_avg']:.3f}" for analysis in pq_analysis.values()]
        }
        
        # Plot comparison table
        axes[0,0].axis('tight')
        axes[0,0].axis('off')
        table = axes[0,0].table(cellText=[list(row) for row in zip(*[comparison_data[col] for col in comparison_data.keys()])],
                               colLabels=list(comparison_data.keys()),
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[0,0].set_title('Performance Summary')
        
        # Time series comparison of key metric
        for i, (name, dataset) in enumerate(datasets.items()):
            df = dataset['data']
            hourly_load = df['ups_load'].resample('H').mean()
            axes[0,1].plot(hourly_load.index, hourly_load.values, 
                          label=short_names[i], alpha=0.8, color=colors[i])
        
        axes[0,1].set_title('UPS Load Comparison Over Time')
        axes[0,1].set_ylabel('Load (%)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Load distribution comparison
        for i, (name, dataset) in enumerate(datasets.items()):
            df = dataset['data']
            axes[1,0].hist(df['ups_load'], bins=30, alpha=0.6, 
                          label=short_names[i], color=colors[i], density=True)
        
        axes[1,0].set_title('Load Distribution Comparison')
        axes[1,0].set_xlabel('Load (%)')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # System health score calculation
        health_scores = []
        for i, name in enumerate(names):
            ups = ups_analysis[name]
            pq = pq_analysis[name]
            
            # Calculate composite health score (0-100)
            load_score = max(0, 100 - abs(ups['load_avg'] - 80))  # Optimal around 80%
            eff_score = ups['efficiency']
            voltage_score = max(0, 100 - pq['voltage_imbalance'] * 50)  # Penalize imbalance
            pf_score = pq['power_factor_avg'] * 100
            
            health_score = (load_score + eff_score + voltage_score + pf_score) / 4
            health_scores.append(health_score)
        
        axes[1,1].bar(range(len(names)), health_scores, color=colors[:len(names)])
        axes[1,1].set_title('System Health Score')
        axes[1,1].set_ylabel('Health Score (0-100)')
        axes[1,1].set_xticks(range(len(names)))
        axes[1,1].set_xticklabels(short_names, rotation=45)
        axes[1,1].axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Good (80+)')
        axes[1,1].axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Fair (60+)')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'system_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_executive_report(self, datasets, ups_analysis, pq_analysis, energy_analysis):
        """Generate executive summary report"""
        report_path = self.output_dir / 'executive_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("POWER SYSTEM EXECUTIVE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Period: Multiple datasets covering power infrastructure monitoring\n\n")
            
            f.write("🔋 UPS PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            for name, analysis in ups_analysis.items():
                period = name.split('_')[1] if '_' in name else name
                f.write(f"\nPeriod: {period}\n")
                f.write(f"  • Average Load: {analysis['load_avg']:.1f}% (Range: {analysis['load_min']:.1f}% - {analysis['load_max']:.1f}%)\n")
                f.write(f"  • System Efficiency: {analysis['efficiency']:.1f}%\n")
                f.write(f"  • Voltage Stability: {'Excellent' if analysis['voltage_stability'] < 0.01 else 'Good' if analysis['voltage_stability'] < 0.02 else 'Fair'}\n")
                f.write(f"  • Frequency Stability: {'Excellent' if analysis['frequency_stability'] < 0.001 else 'Good' if analysis['frequency_stability'] < 0.002 else 'Fair'}\n")
            
            f.write("\n⚡ POWER QUALITY ASSESSMENT\n")
            f.write("-" * 30 + "\n")
            for name, analysis in pq_analysis.items():
                period = name.split('_')[1] if '_' in name else name
                f.write(f"\nPeriod: {period}\n")
                f.write(f"  • Power Factor: {analysis['power_factor_avg']:.3f} ({'Excellent' if analysis['power_factor_avg'] > 0.95 else 'Good' if analysis['power_factor_avg'] > 0.9 else 'Needs Improvement'})\n")
                f.write(f"  • Voltage Imbalance: {analysis['voltage_imbalance']:.2f}% ({'Good' if analysis['voltage_imbalance'] < 2 else 'Concerning'})\n")
                f.write(f"  • Current Imbalance: {analysis['current_imbalance']:.2f}% ({'Good' if analysis['current_imbalance'] < 10 else 'Concerning'})\n")
            
            f.write("\n📊 ENERGY CONSUMPTION INSIGHTS\n")
            f.write("-" * 30 + "\n")
            for name, analysis in energy_analysis.items():
                period = name.split('_')[1] if '_' in name else name
                f.write(f"\nPeriod: {period}\n")
                f.write(f"  • Daily Average: {analysis['daily_consumption_avg']:.1f} kWh\n")
                f.write(f"  • Peak Daily: {analysis['daily_consumption_peak']:.1f} kWh\n")
                f.write(f"  • Monitoring Points: {analysis['total_kwh_meters']} main meters, {analysis['total_pdu_meters']} PDU meters\n")
                f.write(f"  • Consumption Trend: {'Increasing' if analysis['consumption_trend'] > 0 else 'Decreasing' if analysis['consumption_trend'] < 0 else 'Stable'}\n")
            
            f.write("\n🎯 KEY RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            
            # Generate recommendations based on analysis
            all_loads = [analysis['load_avg'] for analysis in ups_analysis.values()]
            all_pf = [analysis['power_factor_avg'] for analysis in pq_analysis.values()]
            all_v_imb = [analysis['voltage_imbalance'] for analysis in pq_analysis.values()]
            
            if max(all_loads) > 90:
                f.write("⚠️  HIGH LOAD: UPS approaching capacity limits. Consider load balancing or capacity expansion.\n")
            elif min(all_loads) < 20:
                f.write("💡 LOW LOAD: UPS operating at low efficiency. Consider rightsizing equipment.\n")
            
            if min(all_pf) < 0.9:
                f.write("⚠️  POWER FACTOR: Install power factor correction equipment to improve efficiency.\n")
            
            if max(all_v_imb) > 2:
                f.write("⚠️  VOLTAGE IMBALANCE: Investigate load distribution and phase balancing.\n")
            
            f.write("✅ MONITORING: Continue regular monitoring of all identified metrics.\n")
            f.write("📈 TRENDING: Implement predictive analytics for proactive maintenance.\n")
            
            f.write(f"\n📁 DETAILED REPORTS\n")
            f.write("-" * 30 + "\n")
            f.write("Generated visualizations:\n")
            f.write("  • ups_dashboard.png - UPS performance metrics\n")
            f.write("  • power_quality_dashboard.png - Power quality analysis\n")
            f.write("  • energy_dashboard.png - Energy consumption patterns\n")
            f.write("  • system_comparison.png - Comparative analysis\n")
        
        return report_path
    
    def run_analysis(self, sample_size=20000):
        """Run complete power system analysis"""
        print("🔌 POWER SYSTEM ANALYSIS")
        print("=" * 50)
        
        # Load datasets
        datasets = self.load_datasets(sample_size)
        
        if not datasets:
            print("No datasets found!")
            return
        
        print(f"\nAnalyzing {len(datasets)} dataset(s)...")
        
        # Run analyses
        ups_analysis = self.analyze_ups_performance(datasets)
        pq_analysis = self.analyze_power_quality(datasets)
        energy_analysis = self.analyze_energy_consumption(datasets)
        
        # Create visualizations
        self.create_dashboard_visualizations(datasets, ups_analysis, pq_analysis, energy_analysis)
        
        # Generate executive report
        report_path = self.generate_executive_report(datasets, ups_analysis, pq_analysis, energy_analysis)
        
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Dashboards saved to: {self.output_dir}")
        print(f"📋 Executive summary: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Advanced Power System Analysis')
    parser.add_argument('--csv-dir', default='outputs/csv_data',
                       help='Directory containing CSV subdirectories (default: outputs/csv_data)')
    parser.add_argument('--output-dir', default='outputs/power_analysis',
                       help='Directory to save analysis results (default: outputs/power_analysis)')
    parser.add_argument('--sample-size', type=int, default=20000,
                       help='Sample size for analysis (default: 20000)')
    
    args = parser.parse_args()
    
    analyzer = PowerSystemAnalyzer(args.csv_dir, args.output_dir)
    analyzer.run_analysis(args.sample_size)


if __name__ == "__main__":
    main()
