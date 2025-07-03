#!/usr/bin/env python3
"""
🎨 ADVANCED POWER AI VISUALIZATIONS
Sophisticated visualizations for electrical engineering analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedPowerVisualizer:
    def __init__(self, data_dir="outputs/csv_data", output_dir="outputs/advanced_viz"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load datasets for visualization"""
        datasets = {}
        for dataset_dir in self.data_dir.glob("*"):
            if dataset_dir.is_dir():
                csv_file = dataset_dir / "leituras.csv"
                if csv_file.exists():
                    print(f"📊 Loading {dataset_dir.name} for visualization...")
                    df = pd.read_csv(csv_file, nrows=10000)  # Sample for visualization
                    df['datetime'] = pd.to_datetime(df['data_hora'])
                    df = df.set_index('datetime').sort_index()
                    datasets[dataset_dir.name] = df
        return datasets
    
    def create_power_quality_dashboard(self, df, dataset_name):
        """Create comprehensive power quality dashboard"""
        print("⚡ Creating power quality dashboard...")
        
        # Calculate electrical features
        df['ups_total_power'] = df[['ups_pa', 'ups_pb', 'ups_pc']].sum(axis=1)
        df['ups_voltage_avg'] = df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].mean(axis=1)
        df['ups_current_avg'] = df[['ups_ia_out', 'ups_ib_out', 'ups_ic_out']].mean(axis=1)
        df['ups_power_factor'] = df['ups_total_power'] / (df['ups_voltage_avg'] * df['ups_current_avg'] + 1e-6)
        df['ups_power_imbalance'] = df[['ups_pa', 'ups_pb', 'ups_pc']].std(axis=1)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Power Consumption Over Time',
                'Voltage Stability',
                'Current Distribution',
                'Power Factor Analysis',
                'Load Balance',
                'System Efficiency'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Power over time
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ups_total_power'],
                mode='lines',
                name='Total Power',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Plot 2: Voltage stability
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ups_va_out'],
                mode='lines',
                name='Phase A',
                line=dict(color='red', width=1)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ups_vb_out'],
                mode='lines',
                name='Phase B',
                line=dict(color='green', width=1)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ups_vc_out'],
                mode='lines',
                name='Phase C',
                line=dict(color='blue', width=1)
            ),
            row=1, col=2
        )
        
        # Plot 3: Current distribution
        current_data = [df['ups_ia_out'].dropna(), df['ups_ib_out'].dropna(), df['ups_ic_out'].dropna()]
        current_labels = ['Phase A', 'Phase B', 'Phase C']
        colors = ['red', 'green', 'blue']
        
        for i, (data, label, color) in enumerate(zip(current_data, current_labels, colors)):
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=label,
                    opacity=0.7,
                    nbinsx=30,
                    marker_color=color
                ),
                row=2, col=1
            )
        
        # Plot 4: Power factor over time
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ups_power_factor'].clip(0, 2),  # Clip extreme values
                mode='lines',
                name='Power Factor',
                line=dict(color='purple', width=2)
            ),
            row=2, col=2
        )
        
        # Add power factor threshold line
        fig.add_hline(
            y=0.85,
            line_dash="dash",
            line_color="red",
            annotation_text="Minimum PF (0.85)",
            row=2, col=2
        )
        
        # Plot 5: Load balance (pie chart simulation with bar)
        load_data = [df['ups_pa'].mean(), df['ups_pb'].mean(), df['ups_pc'].mean()]
        load_labels = ['Phase A', 'Phase B', 'Phase C']
        
        fig.add_trace(
            go.Bar(
                x=load_labels,
                y=load_data,
                marker_color=['red', 'green', 'blue'],
                name='Average Load'
            ),
            row=3, col=1
        )
        
        # Plot 6: System efficiency
        df['efficiency'] = (df['ups_total_power'] / (df['ups_voltage_avg'] * df['ups_current_avg'] + 1e-6)).clip(0, 1.2)
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['efficiency'],
                mode='lines',
                name='Efficiency',
                line=dict(color='orange', width=2)
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Power Quality Dashboard - {dataset_name}',
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        # Save
        output_file = self.output_dir / f'{dataset_name}_power_quality_dashboard.html'
        fig.write_html(output_file)
        print(f"✅ Saved power quality dashboard: {output_file}")
        
        return fig
    
    def create_electrical_analysis_plots(self, df, dataset_name):
        """Create detailed electrical analysis plots"""
        print("🔌 Creating electrical analysis plots...")
        
        # Calculate features
        df['ups_total_power'] = df[['ups_pa', 'ups_pb', 'ups_pc']].sum(axis=1)
        df['ups_voltage_avg'] = df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].mean(axis=1)
        df['ups_current_avg'] = df[['ups_ia_out', 'ups_ib_out', 'ups_ic_out']].mean(axis=1)
        df['ups_power_imbalance'] = df[['ups_pa', 'ups_pb', 'ups_pc']].std(axis=1)
        df['hour'] = df.index.hour
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Electrical Analysis - {dataset_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Power vs Time with hourly patterns
        ax1 = axes[0, 0]
        df.resample('H')['ups_total_power'].mean().plot(ax=ax1, color='blue', linewidth=2)
        ax1.set_title('Hourly Average Power Consumption')
        ax1.set_ylabel('Power (W)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Voltage vs Current relationship
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['ups_voltage_avg'], df['ups_current_avg'], 
                            c=df['ups_total_power'], cmap='viridis', alpha=0.6, s=10)
        ax2.set_xlabel('Voltage (V)')
        ax2.set_ylabel('Current (A)')
        ax2.set_title('Voltage vs Current (colored by Power)')
        plt.colorbar(scatter, ax=ax2, label='Power (W)')
        
        # Plot 3: Power factor distribution
        ax3 = axes[0, 2]
        pf_data = df['ups_total_power'] / (df['ups_voltage_avg'] * df['ups_current_avg'] + 1e-6)
        pf_data = pf_data.clip(0, 2)  # Remove extreme values
        ax3.hist(pf_data.dropna(), bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(0.85, color='red', linestyle='--', linewidth=2, label='Min PF (0.85)')
        ax3.set_xlabel('Power Factor')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Power Factor Distribution')
        ax3.legend()
        
        # Plot 4: Load imbalance over time
        ax4 = axes[1, 0]
        df.resample('H')['ups_power_imbalance'].mean().plot(ax=ax4, color='red', linewidth=2)
        ax4.set_title('Power Imbalance (Hourly Average)')
        ax4.set_ylabel('Std Dev (W)')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Daily power pattern
        ax5 = axes[1, 1]
        hourly_avg = df.groupby('hour')['ups_total_power'].mean()
        hourly_std = df.groupby('hour')['ups_total_power'].std()
        ax5.errorbar(hourly_avg.index, hourly_avg.values, yerr=hourly_std.values, 
                    capsize=5, capthick=2, color='purple', linewidth=2)
        ax5.set_xlabel('Hour of Day')
        ax5.set_ylabel('Power (W)')
        ax5.set_title('Daily Power Consumption Pattern')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Phase balance analysis
        ax6 = axes[1, 2]
        phase_data = [df['ups_pa'].mean(), df['ups_pb'].mean(), df['ups_pc'].mean()]
        phase_labels = ['Phase A', 'Phase B', 'Phase C']
        colors = ['red', 'green', 'blue']
        bars = ax6.bar(phase_labels, phase_data, color=colors, alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Average Power (W)')
        ax6.set_title('Phase Load Balance')
        
        # Add value labels on bars
        for bar, value in zip(bars, phase_data):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save matplotlib plot
        output_file = self.output_dir / f'{dataset_name}_electrical_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved electrical analysis: {output_file}")
    
    def create_pdu_analysis(self, df, dataset_name):
        """Create PDU (Power Distribution Unit) analysis"""
        print("🔌 Creating PDU analysis...")
        
        # Get PDU columns
        pdu_current_cols = [col for col in df.columns if col.startswith('pdu') and col.endswith('_i')]
        pdu_kwh_cols = [col for col in df.columns if col.startswith('pdu') and col.endswith('_kwh')]
        pdu_fp_cols = [col for col in df.columns if col.startswith('pdu') and col.endswith('_fp')]
        
        if not pdu_current_cols:
            print("⚠️ No PDU data found")
            return
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'PDU Current Distribution',
                'PDU Energy Consumption',
                'PDU Power Factor',
                'PDU Load Balance'
            ]
        )
        
        # PDU Current over time
        for i, col in enumerate(pdu_current_cols):
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode='lines',
                        name=f'PDU {i+1}',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # PDU Energy consumption
        if pdu_kwh_cols:
            pdu_energy_avg = [df[col].mean() for col in pdu_kwh_cols if col in df.columns]
            pdu_labels = [f'PDU {i+1}' for i in range(len(pdu_energy_avg))]
            
            fig.add_trace(
                go.Bar(
                    x=pdu_labels,
                    y=pdu_energy_avg,
                    name='Average Energy',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # PDU Power Factor
        if pdu_fp_cols:
            for i, col in enumerate(pdu_fp_cols):
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[col],
                            mode='lines',
                            name=f'PDU {i+1} PF',
                            line=dict(width=1)
                        ),
                        row=2, col=1
                    )
        
        # PDU Load balance (current distribution)
        if pdu_current_cols:
            current_avg = [df[col].mean() for col in pdu_current_cols if col in df.columns]
            pdu_labels = [f'PDU {i+1}' for i in range(len(current_avg))]
            
            fig.add_trace(
                go.Pie(
                    labels=pdu_labels,
                    values=current_avg,
                    name="PDU Balance"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'PDU Analysis - {dataset_name}',
            height=800,
            showlegend=True
        )
        
        # Save
        output_file = self.output_dir / f'{dataset_name}_pdu_analysis.html'
        fig.write_html(output_file)
        print(f"✅ Saved PDU analysis: {output_file}")
    
    def create_anomaly_visualization(self, df, dataset_name):
        """Create anomaly detection visualization"""
        print("🚨 Creating anomaly visualization...")
        
        # Calculate key features for anomaly detection
        df['ups_total_power'] = df[['ups_pa', 'ups_pb', 'ups_pc']].sum(axis=1)
        df['ups_voltage_avg'] = df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].mean(axis=1)
        df['ups_current_avg'] = df[['ups_ia_out', 'ups_ib_out', 'ups_ic_out']].mean(axis=1)
        
        # Simple anomaly detection using z-score
        from scipy.stats import zscore
        
        features = ['ups_total_power', 'ups_voltage_avg', 'ups_current_avg', 'ups_load']
        df_features = df[features].fillna(0)
        
        # Calculate z-scores
        z_scores = np.abs(zscore(df_features, axis=0))
        df['is_anomaly'] = (z_scores > 3).any(axis=1)
        df['anomaly_score'] = z_scores.max(axis=1)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Power with Anomalies',
                'Anomaly Score Distribution',
                'Voltage with Anomalies',
                'Anomaly Timeline'
            ]
        )
        
        # Plot 1: Power with anomalies highlighted
        normal_data = df[~df['is_anomaly']]
        anomaly_data = df[df['is_anomaly']]
        
        fig.add_trace(
            go.Scatter(
                x=normal_data.index,
                y=normal_data['ups_total_power'],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=4, opacity=0.6)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data['ups_total_power'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=8, opacity=0.8)
            ),
            row=1, col=1
        )
        
        # Plot 2: Anomaly score distribution
        fig.add_trace(
            go.Histogram(
                x=df['anomaly_score'],
                name='Anomaly Score',
                nbinsx=50,
                marker_color='orange'
            ),
            row=1, col=2
        )
        
        # Plot 3: Voltage with anomalies
        fig.add_trace(
            go.Scatter(
                x=normal_data.index,
                y=normal_data['ups_voltage_avg'],
                mode='markers',
                name='Normal Voltage',
                marker=dict(color='green', size=4, opacity=0.6)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data['ups_voltage_avg'],
                mode='markers',
                name='Anomaly Voltage',
                marker=dict(color='red', size=8, opacity=0.8)
            ),
            row=2, col=1
        )
        
        # Plot 4: Anomaly timeline
        anomaly_counts = df.resample('D')['is_anomaly'].sum()
        
        fig.add_trace(
            go.Bar(
                x=anomaly_counts.index,
                y=anomaly_counts.values,
                name='Daily Anomalies',
                marker_color='red'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Anomaly Analysis - {dataset_name}',
            height=800,
            showlegend=True
        )
        
        # Save
        output_file = self.output_dir / f'{dataset_name}_anomaly_analysis.html'
        fig.write_html(output_file)
        print(f"✅ Saved anomaly analysis: {output_file}")
        
        # Print anomaly summary
        total_anomalies = df['is_anomaly'].sum()
        total_points = len(df)
        print(f"📊 Found {total_anomalies} anomalies out of {total_points} data points ({total_anomalies/total_points*100:.2f}%)")
    
    def generate_summary_report(self, datasets):
        """Generate a comprehensive summary report"""
        print("📋 Generating summary report...")
        
        summary_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Power AI Advanced Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2E86AB; border-bottom: 3px solid #2E86AB; }
                h2 { color: #A23B72; }
                .dataset { background: #f0f0f0; padding: 20px; margin: 20px 0; border-radius: 10px; }
                .metric { background: white; padding: 10px; margin: 10px; border-left: 4px solid #F18F01; }
                .good { border-left-color: #28a745; }
                .warning { border-left-color: #ffc107; }
                .danger { border-left-color: #dc3545; }
            </style>
        </head>
        <body>
            <h1>🔋 Power AI Advanced Analysis Report</h1>
            <p>Generated on: {}</p>
        """.format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        for name, df in datasets.items():
            # Calculate metrics
            df['ups_total_power'] = df[['ups_pa', 'ups_pb', 'ups_pc']].sum(axis=1)
            df['ups_voltage_avg'] = df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].mean(axis=1)
            df['ups_current_avg'] = df[['ups_ia_out', 'ups_ib_out', 'ups_ic_out']].mean(axis=1)
            df['ups_power_factor'] = df['ups_total_power'] / (df['ups_voltage_avg'] * df['ups_current_avg'] + 1e-6)
            
            avg_power = df['ups_total_power'].mean()
            avg_voltage = df['ups_voltage_avg'].mean()
            avg_pf = df['ups_power_factor'].clip(0, 2).mean()
            avg_load = df['ups_load'].mean()
            
            # Determine status
            pf_status = "good" if avg_pf > 0.85 else "warning" if avg_pf > 0.7 else "danger"
            voltage_status = "good" if 215 <= avg_voltage <= 225 else "warning"
            load_status = "good" if avg_load < 80 else "warning" if avg_load < 90 else "danger"
            
            summary_html += f"""
            <div class="dataset">
                <h2>📊 Dataset: {name}</h2>
                <div class="metric good">
                    <strong>Data Points:</strong> {len(df):,}
                </div>
                <div class="metric {pf_status}">
                    <strong>Average Power Factor:</strong> {avg_pf:.3f}
                </div>
                <div class="metric {voltage_status}">
                    <strong>Average Voltage:</strong> {avg_voltage:.1f} V
                </div>
                <div class="metric {load_status}">
                    <strong>Average Load:</strong> {avg_load:.1f}%
                </div>
                <div class="metric">
                    <strong>Average Power:</strong> {avg_power:.0f} W
                </div>
            </div>
            """
        
        summary_html += """
        </body>
        </html>
        """
        
        # Save report
        report_file = self.output_dir / 'advanced_analysis_report.html'
        with open(report_file, 'w') as f:
            f.write(summary_html)
        
        print(f"✅ Saved summary report: {report_file}")

def run_advanced_visualizations():
    """Run all advanced visualizations"""
    print("🚀 Starting Advanced Power AI Visualizations")
    print("=" * 60)
    
    visualizer = AdvancedPowerVisualizer()
    datasets = visualizer.load_data()
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    for name, df in datasets.items():
        print(f"\n🔍 Creating visualizations for: {name}")
        print("-" * 40)
        
        # Create all visualizations
        visualizer.create_power_quality_dashboard(df, name)
        visualizer.create_electrical_analysis_plots(df, name)
        visualizer.create_pdu_analysis(df, name)
        visualizer.create_anomaly_visualization(df, name)
    
    # Generate summary report
    visualizer.generate_summary_report(datasets)
    
    print("\n" + "=" * 60)
    print("🎯 ADVANCED VISUALIZATIONS COMPLETE!")
    print(f"📁 Check output folder: {visualizer.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    run_advanced_visualizations() 