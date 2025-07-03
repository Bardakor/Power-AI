#!/usr/bin/env python3
"""
Power AI PDF Report Generator
Creates comprehensive PDF reports with data analysis, visualizations, and recommendations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

class PowerAIPDFReportGenerator:
    """Comprehensive PDF Report Generator for Power AI Analysis"""
    
    def __init__(self, data_dir="outputs/csv_data", output_dir="outputs/reports"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis results storage
        self.datasets = {}
        self.power_analysis = {}
        self.ml_results = {}
        self.data_quality = {}
        self.recommendations = []
        
        # Modern color scheme
        self.colors = {
            'primary': '#1e3a8a',     # Deep blue
            'secondary': '#64748b',   # Slate gray
            'success': '#10b981',     # Emerald green
            'warning': '#f59e0b',     # Amber
            'danger': '#ef4444',      # Red
            'info': '#3b82f6'         # Blue
        }
        
    def load_and_analyze_data(self, sample_size=50000):
        """Load datasets and run comprehensive analysis"""
        print("📊 Loading datasets for comprehensive analysis...")
        
        # Load all available datasets
        for dataset_dir in self.data_dir.glob("*"):
            if dataset_dir.is_dir():
                csv_file = dataset_dir / "leituras.csv"
                if csv_file.exists():
                    try:
                        print(f"Loading {dataset_dir.name}...")
                        
                        # Get total rows for sampling
                        total_rows = sum(1 for line in open(csv_file)) - 1
                        
                        if total_rows <= sample_size:
                            df = pd.read_csv(csv_file)
                        else:
                            # Smart sampling
                            recent_sample = int(sample_size * 0.3)
                            random_sample = sample_size - recent_sample
                            
                            df_recent = pd.read_csv(csv_file, skiprows=range(1, total_rows - recent_sample + 1))
                            skip_rows = sorted(np.random.choice(range(1, total_rows - recent_sample), 
                                                              total_rows - recent_sample - random_sample, 
                                                              replace=False))
                            df_random = pd.read_csv(csv_file, skiprows=skip_rows, nrows=random_sample)
                            df = pd.concat([df_random, df_recent], ignore_index=True)
                        
                        # Parse datetime and set index
                        df['datetime'] = pd.to_datetime(df['data_hora'])
                        df = df.set_index('datetime').sort_index()
                        
                        # Store dataset with metadata
                        self.datasets[dataset_dir.name] = {
                            'data': df,
                            'total_rows': total_rows,
                            'sample_size': len(df),
                            'time_range': {
                                'start': df.index.min(),
                                'end': df.index.max(),
                                'duration_hours': (df.index.max() - df.index.min()).total_seconds() / 3600
                            }
                        }
                        
                        print(f"  ✅ Loaded {len(df):,} rows (from {total_rows:,} total)")
                        
                    except Exception as e:
                        print(f"  ❌ Error loading {dataset_dir.name}: {e}")
        
        print(f"📈 Running comprehensive analysis on {len(self.datasets)} datasets...")
        self._run_analysis()
        
    def _run_analysis(self):
        """Run comprehensive analysis"""
        
        # Power System Analysis
        print("⚡ Running power system analysis...")
        self._analyze_power_systems()
        
        # Data Quality Analysis
        print("🔍 Running data quality analysis...")
        self._analyze_data_quality()
        
        # Generate Recommendations
        print("💡 Generating recommendations...")
        self._generate_recommendations()
    
    def _analyze_power_systems(self):
        """Analyze power system metrics"""
        for name, dataset_info in self.datasets.items():
            df = dataset_info['data']
            
            # UPS Performance Analysis
            ups_metrics = {}
            if 'ups_load' in df.columns:
                ups_metrics = {
                    'load_avg': df['ups_load'].mean(),
                    'load_max': df['ups_load'].max(),
                    'load_min': df['ups_load'].min(),
                    'load_std': df['ups_load'].std()
                }
            
            # Power Quality Analysis
            pq_metrics = {}
            if all(col in df.columns for col in ['ups_va_out', 'ups_vb_out', 'ups_vc_out']):
                voltages = df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']]
                voltage_avg = voltages.mean(axis=1)
                voltage_imbalance = ((voltages.max(axis=1) - voltages.min(axis=1)) / voltage_avg * 100).mean()
                
                pq_metrics = {
                    'voltage_avg': voltage_avg.mean(),
                    'voltage_imbalance': voltage_imbalance,
                    'voltage_stability': voltage_avg.std()
                }
            
            self.power_analysis[name] = {
                'ups_performance': ups_metrics,
                'power_quality': pq_metrics
            }
    
    def _analyze_data_quality(self):
        """Analyze data quality metrics"""
        for name, dataset_info in self.datasets.items():
            df = dataset_info['data']
            
            missing_by_column = df.isnull().sum()
            duplicates = df.duplicated().sum()
            
            # Time series quality
            time_gaps = df.index.to_series().diff()
            
            self.data_quality[name] = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': missing_by_column.sum(),
                'duplicates': duplicates,
                'completeness': (1 - missing_by_column.sum() / (len(df) * len(df.columns))) * 100,
                'median_interval_minutes': time_gaps.median().total_seconds() / 60 if not time_gaps.empty else 0,
                'ups_columns': len([col for col in df.columns if 'ups' in col.lower()]),
                'meter_columns': len([col for col in df.columns if 'met' in col.lower()]),
                'pdu_columns': len([col for col in df.columns if 'pdu' in col.lower()])
            }
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Power system recommendations
        for dataset, analysis in self.power_analysis.items():
            ups_perf = analysis.get('ups_performance', {})
            load_avg = ups_perf.get('load_avg', 0)
            
            if load_avg > 90:
                recommendations.append({
                    'category': 'Critical',
                    'message': f"UPS load at {load_avg:.1f}% - approaching capacity limits",
                    'priority': 'High'
                })
            elif load_avg < 20:
                recommendations.append({
                    'category': 'Optimization',
                    'message': f"UPS load at {load_avg:.1f}% - consider rightsizing equipment",
                    'priority': 'Medium'
                })
            
            pq = analysis.get('power_quality', {})
            v_imbalance = pq.get('voltage_imbalance', 0)
            if v_imbalance > 2:
                recommendations.append({
                    'category': 'Power Quality',
                    'message': f"Voltage imbalance at {v_imbalance:.2f}% - investigate phase balancing",
                    'priority': 'Medium'
                })
        
        # Data quality recommendations
        for dataset, quality in self.data_quality.items():
            if quality['completeness'] < 95:
                recommendations.append({
                    'category': 'Data Quality',
                    'message': f"Data completeness at {quality['completeness']:.1f}% - investigate missing data",
                    'priority': 'Medium'
                })
        
        self.recommendations = recommendations
    
    def generate_pdf_report(self, filename=None):
        """Generate streamlined PDF report with system overview only"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"PowerAI_Report_{timestamp}.pdf"
        
        print(f"📄 Generating PDF report: {filename}")
        
        with PdfPages(filename) as pdf:
            self._create_system_overview(pdf)
        
        print(f"✅ PDF report generated: {filename}")
        return filename
    
    def _create_executive_summary(self, pdf):
        """Create executive summary page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Header
        ax.text(0.5, 0.95, "Power AI System Analysis", 
                ha='center', va='top', fontsize=24, fontweight='bold', color=self.colors['primary'])
        ax.text(0.5, 0.92, "Executive Summary Report", 
                ha='center', va='top', fontsize=16, color=self.colors['secondary'])
        ax.text(0.5, 0.89, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", 
                ha='center', va='top', fontsize=12, style='italic')
        
        # System overview
        y_pos = 0.82
        ax.text(0.05, y_pos, "📊 SYSTEM OVERVIEW", fontsize=14, fontweight='bold')
        y_pos -= 0.05
        
        total_rows = sum(info['total_rows'] for info in self.datasets.values())
        total_datasets = len(self.datasets)
        
        ax.text(0.05, y_pos, f"• Datasets Analyzed: {total_datasets}", fontsize=11)
        y_pos -= 0.03
        ax.text(0.05, y_pos, f"• Total Data Points: {total_rows:,}", fontsize=11)
        y_pos -= 0.03
        ax.text(0.05, y_pos, f"• Components: UPS, Energy Meters (2), PDU Channels (8)", fontsize=11)
        y_pos -= 0.03
        
        # Key findings
        y_pos -= 0.05
        ax.text(0.05, y_pos, "⚡ KEY FINDINGS", fontsize=14, fontweight='bold')
        y_pos -= 0.05
        
        # Calculate key metrics
        avg_loads = []
        for analysis in self.power_analysis.values():
            ups_perf = analysis.get('ups_performance', {})
            if 'load_avg' in ups_perf:
                avg_loads.append(ups_perf['load_avg'])
        
        if avg_loads:
            overall_load = np.mean(avg_loads)
            ax.text(0.05, y_pos, f"• Average UPS Load: {overall_load:.1f}%", fontsize=11)
            y_pos -= 0.03
        
        total_completeness = np.mean([quality['completeness'] for quality in self.data_quality.values()])
        ax.text(0.05, y_pos, f"• Data Quality Score: {total_completeness:.1f}%", fontsize=11)
        y_pos -= 0.03
        
        # Critical alerts
        critical_recs = [r for r in self.recommendations if r['priority'] == 'High']
        if critical_recs:
            y_pos -= 0.05
            ax.text(0.05, y_pos, "🚨 CRITICAL ALERTS", fontsize=14, fontweight='bold', color=self.colors['danger'])
            y_pos -= 0.05
            
            for rec in critical_recs[:3]:
                ax.text(0.05, y_pos, f"• {rec['message']}", fontsize=11, color=self.colors['danger'])
                y_pos -= 0.04
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_system_overview(self, pdf):
        """Create beautiful executive dashboard page"""
        # Create figure with modern proportions
        fig = plt.figure(figsize=(11, 8.5), facecolor='white')  # Landscape for better layout
        fig.patch.set_facecolor('white')
        
        # Modern header with clean typography
        fig.text(0.02, 0.95, "POWER AI", fontsize=32, fontweight='bold', color='#0f172a')
        fig.text(0.02, 0.90, "System Performance Dashboard", fontsize=16, color='#475569')
        fig.text(0.98, 0.95, datetime.now().strftime('%B %d, %Y'), 
                ha='right', fontsize=12, color='#64748b')
        fig.text(0.98, 0.92, datetime.now().strftime('%H:%M'), 
                ha='right', fontsize=12, color='#64748b')
        
        # Elegant separator line
        fig.add_artist(plt.Line2D([0.02, 0.98], [0.88, 0.88], color='#e2e8f0', linewidth=1))
        
        # Calculate key metrics
        total_datasets = len(self.datasets)
        total_rows = sum(info['total_rows'] for info in self.datasets.values())
        
        # Get UPS load data
        ups_loads = []
        dataset_names = []
        for name, analysis in self.power_analysis.items():
            ups_perf = analysis.get('ups_performance', {})
            if 'load_avg' in ups_perf:
                ups_loads.append(ups_perf['load_avg'])
                dataset_names.append(name.split('_')[1] if '_' in name else name)
        
        # Get power quality data
        power_quality_data = []
        for name, analysis in self.power_analysis.items():
            pq = analysis.get('power_quality', {})
            if 'voltage_imbalance' in pq:
                power_quality_data.append(pq['voltage_imbalance'])
        
        # Create clean layout with proper spacing
        gs = fig.add_gridspec(3, 4, 
                             height_ratios=[1, 2, 1.5], 
                             width_ratios=[1, 1, 1, 1],
                             hspace=0.3, wspace=0.2,
                             left=0.05, right=0.95, top=0.82, bottom=0.05)
        
        # KPI Cards at top
        self._create_executive_kpis(fig, gs, total_datasets, total_rows, ups_loads, power_quality_data)
        
        # Main visualization section
        self._create_main_charts(fig, gs, ups_loads, dataset_names, power_quality_data)
        
        # Status overview at bottom
        self._create_status_overview(fig, gs)
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
        plt.close()
    
    def _create_executive_kpis(self, fig, gs, total_datasets, total_rows, ups_loads, power_quality_data):
        """Create executive KPI cards"""
        # Calculate KPIs
        avg_ups_load = np.mean(ups_loads) if ups_loads else 0
        max_ups_load = max(ups_loads) if ups_loads else 0
        avg_power_quality = np.mean(power_quality_data) if power_quality_data else 0
        
        # Data quality score
        data_quality_score = np.mean([quality['completeness'] for quality in self.data_quality.values()])
        
        kpis = [
            {"title": "ACTIVE DATASETS", "value": f"{total_datasets}", "unit": "", "color": "#0369a1", "status": "good"},
            {"title": "DATA VOLUME", "value": f"{total_rows/1000000:.1f}", "unit": "M Points", "color": "#0369a1", "status": "good"},
            {"title": "UPS LOAD AVG", "value": f"{avg_ups_load:.0f}", "unit": "%", "color": "#dc2626" if avg_ups_load > 80 else "#059669", "status": "critical" if avg_ups_load > 80 else "good"},
            {"title": "MAX UPS LOAD", "value": f"{max_ups_load:.0f}", "unit": "%", "color": "#dc2626" if max_ups_load > 90 else "#f59e0b" if max_ups_load > 80 else "#059669", "status": "critical" if max_ups_load > 90 else "warning" if max_ups_load > 80 else "good"}
        ]
        
        for i, kpi in enumerate(kpis):
            ax = fig.add_subplot(gs[0, i])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Modern card with subtle shadow effect
            rect = Rectangle((0.02, 0.1), 0.96, 0.8, 
                           facecolor='white', edgecolor='#e5e7eb', 
                           linewidth=1)
            ax.add_patch(rect)
            
            # Status indicator bar
            status_colors = {"good": "#10b981", "warning": "#f59e0b", "critical": "#ef4444"}
            status_rect = Rectangle((0.02, 0.1), 0.96, 0.05, 
                                  facecolor=status_colors.get(kpi['status'], '#6b7280'), 
                                  edgecolor='none')
            ax.add_patch(status_rect)
            
            # KPI content
            ax.text(0.5, 0.75, kpi['value'], fontsize=28, fontweight='bold', 
                   ha='center', va='center', color=kpi['color'])
            ax.text(0.5, 0.55, kpi['unit'], fontsize=12, 
                   ha='center', va='center', color='#6b7280')
            ax.text(0.5, 0.3, kpi['title'], fontsize=9, fontweight='bold',
                   ha='center', va='center', color='#374151')
    
    def _create_main_charts(self, fig, gs, ups_loads, dataset_names, power_quality_data):
        """Create main visualization charts"""
        # UPS Performance Chart (left side)
        ax1 = fig.add_subplot(gs[1, :2])
        if ups_loads and dataset_names:
            # Create modern bar chart
            bars = ax1.bar(range(len(ups_loads)), ups_loads, 
                          color=['#ef4444' if load > 90 else '#f59e0b' if load > 80 else '#10b981' for load in ups_loads],
                          alpha=0.8, width=0.6)
            
            # Add reference lines
            ax1.axhline(y=80, color='#f59e0b', linestyle='--', alpha=0.6, linewidth=2)
            ax1.axhline(y=90, color='#ef4444', linestyle='--', alpha=0.6, linewidth=2)
            
            # Styling
            ax1.set_title('UPS Load Distribution', fontsize=16, fontweight='bold', color='#1f2937', pad=20)
            ax1.set_ylabel('Load (%)', fontsize=12, color='#374151')
            ax1.set_xticks(range(len(dataset_names)))
            ax1.set_xticklabels([name[:8] for name in dataset_names], rotation=0, fontsize=10)
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.2, axis='y')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('#d1d5db')
            ax1.spines['bottom'].set_color('#d1d5db')
            
            # Add value labels on bars
            for bar, load in zip(bars, ups_loads):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{load:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'UPS Data\nNot Available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14, color='#6b7280')
            ax1.set_title('UPS Load Distribution', fontsize=16, fontweight='bold', color='#1f2937', pad=20)
            ax1.axis('off')
        
        # System Health Overview (right side)
        ax2 = fig.add_subplot(gs[1, 2:])
        self._create_health_gauge(ax2, ups_loads, power_quality_data)
    
    def _create_health_gauge(self, ax, ups_loads, power_quality_data):
        """Create modern health gauge"""
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        ax.set_title('System Health Score', fontsize=16, fontweight='bold', color='#1f2937', pad=20)
        
        # Calculate overall health score
        health_score = self._calculate_overall_health(ups_loads, power_quality_data)
        
        # Draw gauge background
        circle = plt.Circle((0, 0), 1, fill=False, edgecolor='#e5e7eb', linewidth=8)
        ax.add_patch(circle)
        
        # Draw health arc
        theta = np.linspace(0, 2 * np.pi * (health_score / 100), 100)
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Color based on health score
        if health_score >= 80:
            color = '#10b981'  # Green
            status = 'EXCELLENT'
        elif health_score >= 60:
            color = '#f59e0b'  # Yellow
            status = 'GOOD'
        else:
            color = '#ef4444'  # Red
            status = 'NEEDS ATTENTION'
        
        # Draw the arc
        for i in range(len(theta)-1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, linewidth=8, alpha=0.8)
        
        # Add score text
        ax.text(0, 0.1, f'{health_score:.0f}', fontsize=48, fontweight='bold', 
               ha='center', va='center', color=color)
        ax.text(0, -0.2, status, fontsize=12, fontweight='bold',
               ha='center', va='center', color='#374151')
        ax.text(0, -0.35, 'OVERALL HEALTH', fontsize=10,
               ha='center', va='center', color='#6b7280')
    
    def _calculate_overall_health(self, ups_loads, power_quality_data):
        """Calculate overall system health score"""
        scores = []
        
        # UPS health (weight: 40%)
        if ups_loads:
            avg_load = np.mean(ups_loads)
            max_load = max(ups_loads)
            if max_load > 95:
                ups_score = 20
            elif max_load > 90:
                ups_score = 40
            elif max_load > 80:
                ups_score = 70
            elif avg_load < 20:
                ups_score = 80  # Underutilized but stable
            else:
                ups_score = 95
            scores.append(ups_score * 0.4)
        
        # Power quality health (weight: 30%)
        if power_quality_data:
            avg_imbalance = np.mean(power_quality_data)
            if avg_imbalance > 3:
                pq_score = 30
            elif avg_imbalance > 2:
                pq_score = 60
            elif avg_imbalance > 1:
                pq_score = 80
            else:
                pq_score = 95
            scores.append(pq_score * 0.3)
        
        # Data quality health (weight: 30%)
        if self.data_quality:
            avg_completeness = np.mean([quality['completeness'] for quality in self.data_quality.values()])
            data_score = min(avg_completeness, 100)
            scores.append(data_score * 0.3)
        
        return sum(scores) if scores else 75  # Default score if no data
    
    def _create_status_overview(self, fig, gs):
        """Create status overview section"""
        ax = fig.add_subplot(gs[2, :])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Section title
        ax.text(0.02, 0.9, 'SYSTEM STATUS OVERVIEW', fontsize=14, fontweight='bold', color='#1f2937')
        
        # Create status items
        status_items = []
        
        # Check UPS status
        ups_loads = []
        for analysis in self.power_analysis.values():
            ups_perf = analysis.get('ups_performance', {})
            if 'load_avg' in ups_perf:
                ups_loads.append(ups_perf['load_avg'])
        
        if ups_loads:
            max_load = max(ups_loads)
            if max_load > 90:
                status_items.append({"icon": "🔴", "text": f"UPS Critical Load: {max_load:.0f}% - Immediate attention required"})
            elif max_load > 80:
                status_items.append({"icon": "🟡", "text": f"UPS High Load: {max_load:.0f}% - Monitor closely"})
            else:
                status_items.append({"icon": "🟢", "text": f"UPS Optimal Load: {max_load:.0f}% - System operating normally"})
        
        # Check data quality
        if self.data_quality:
            avg_completeness = np.mean([quality['completeness'] for quality in self.data_quality.values()])
            if avg_completeness < 90:
                status_items.append({"icon": "🟡", "text": f"Data Quality: {avg_completeness:.1f}% - Some missing data detected"})
            else:
                status_items.append({"icon": "🟢", "text": f"Data Quality: {avg_completeness:.1f}% - High quality data"})
        
        # Check power quality
        power_quality_issues = False
        for analysis in self.power_analysis.values():
            pq = analysis.get('power_quality', {})
            if 'voltage_imbalance' in pq and pq['voltage_imbalance'] > 2:
                power_quality_issues = True
                break
        
        if power_quality_issues:
            status_items.append({"icon": "🟡", "text": "Power Quality: Voltage imbalance detected - Review electrical balance"})
        else:
            status_items.append({"icon": "🟢", "text": "Power Quality: Stable voltage levels across all phases"})
        
        # Add dataset info
        total_datasets = len(self.datasets)
        total_rows = sum(info['total_rows'] for info in self.datasets.values())
        status_items.append({"icon": "📊", "text": f"Data Sources: {total_datasets} datasets • {total_rows:,} total records analyzed"})
        
        # Display status items
        y_pos = 0.7
        for item in status_items[:4]:  # Limit to 4 items for clean layout
            ax.text(0.02, y_pos, item['icon'], fontsize=14, va='center')
            ax.text(0.06, y_pos, item['text'], fontsize=11, va='center', color='#374151')
            y_pos -= 0.15
    
    def _plot_dataset_comparison(self, ax):
        """Plot dataset comparison"""
        if not self.datasets:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Dataset Comparison')
            return
        
        names = [name.split('_')[1][:10] if '_' in name else name[:10] for name in self.datasets.keys()]
        sizes = [info['total_rows'] / 1000 for info in self.datasets.values()]
        durations = [info['time_range']['duration_hours'] for info in self.datasets.values()]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, sizes, width, label='Data Points (K)', color=self.colors['primary'], alpha=0.8)
        ax.bar(x + width/2, durations, width, label='Duration (Hours)', color=self.colors['secondary'], alpha=0.8)
        
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Count / Hours')
        ax.set_title('Dataset Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_ups_loads(self, ax):
        """Plot UPS load distribution"""
        loads = []
        names = []
        
        for name, analysis in self.power_analysis.items():
            ups_perf = analysis.get('ups_performance', {})
            if 'load_avg' in ups_perf:
                loads.append(ups_perf['load_avg'])
                names.append(name.split('_')[1][:8] if '_' in name else name[:8])
        
        if loads:
            colors = [self.colors['success'] if load < 80 else 
                     self.colors['warning'] if load < 90 else 
                     self.colors['danger'] for load in loads]
            
            bars = ax.bar(names, loads, color=colors, alpha=0.8)
            ax.set_ylabel('Load (%)')
            ax.set_title('UPS Load Distribution')
            ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Warning (80%)')
            ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Critical (90%)')
            ax.legend()
            
            # Add value labels
            for bar, load in zip(bars, loads):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{load:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'UPS Load Data\nNot Available', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('UPS Load Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_power_quality(self, ax):
        """Plot power quality metrics"""
        voltage_imbalances = []
        names = []
        
        for name, analysis in self.power_analysis.items():
            pq = analysis.get('power_quality', {})
            if 'voltage_imbalance' in pq:
                voltage_imbalances.append(pq['voltage_imbalance'])
                names.append(name.split('_')[1][:8] if '_' in name else name[:8])
        
        if voltage_imbalances:
            colors = [self.colors['success'] if imb < 1 else 
                     self.colors['warning'] if imb < 2 else 
                     self.colors['danger'] for imb in voltage_imbalances]
            
            bars = ax.bar(names, voltage_imbalances, color=colors, alpha=0.8)
            ax.set_ylabel('Imbalance (%)')
            ax.set_title('Voltage Imbalance')
            ax.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='Warning (1%)')
            ax.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Critical (2%)')
            ax.legend()
            
            # Add value labels
            for bar, imb in zip(bars, voltage_imbalances):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{imb:.2f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Power Quality Data\nNot Available', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Voltage Imbalance')
        ax.grid(True, alpha=0.3)
    
    def _create_stats_cards(self, fig, gs):
        """Create beautiful statistics cards at the top"""
        # Calculate key statistics
        total_datasets = len(self.datasets)
        total_rows = sum(info['total_rows'] for info in self.datasets.values())
        
        # Calculate average load
        avg_loads = []
        for analysis in self.power_analysis.values():
            ups_perf = analysis.get('ups_performance', {})
            if 'load_avg' in ups_perf:
                avg_loads.append(ups_perf['load_avg'])
        overall_load = np.mean(avg_loads) if avg_loads else 0
        
        # Calculate data quality
        total_completeness = np.mean([quality['completeness'] for quality in self.data_quality.values()])
        
        # Create cards
        cards = [
            {"title": "Datasets", "value": f"{total_datasets}", "icon": "📊", "color": "#3b82f6"},
            {"title": "Data Points", "value": f"{total_rows/1000:.0f}K", "icon": "📈", "color": "#059669"},
            {"title": "Avg UPS Load", "value": f"{overall_load:.1f}%", "icon": "🔋", "color": "#dc2626" if overall_load > 80 else "#059669"},
            {"title": "Data Quality", "value": f"{total_completeness:.1f}%", "icon": "✅", "color": "#7c3aed"}
        ]
        
        for i, card in enumerate(cards):
            ax = fig.add_subplot(gs[0, i])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Card background
            rect = Rectangle((0.05, 0.1), 0.9, 0.8, 
                           facecolor='white', edgecolor=card['color'], 
                           linewidth=2, alpha=0.1)
            ax.add_patch(rect)
            
            # Icon and value
            ax.text(0.2, 0.7, card['icon'], fontsize=24, ha='center', va='center')
            ax.text(0.2, 0.4, card['value'], fontsize=16, fontweight='bold', 
                   ha='center', va='center', color=card['color'])
            ax.text(0.2, 0.25, card['title'], fontsize=10, ha='center', va='center', 
                   color='#374151')
    
    def _plot_dataset_comparison_beautiful(self, ax):
        """Create beautiful dataset comparison chart"""
        if not self.datasets:
            ax.text(0.5, 0.5, '📊 No Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='#6b7280')
            ax.set_title('Dataset Overview', fontweight='bold', pad=20, color='#1f2937')
            ax.axis('off')
            return
        
        names = [name.split('_')[1][:8] if '_' in name else name[:8] for name in self.datasets.keys()]
        sizes = [info['total_rows'] / 1000 for info in self.datasets.values()]
        durations = [info['time_range']['duration_hours'] / 24 for info in self.datasets.values()]  # Convert to days
        
        x = np.arange(len(names))
        width = 0.35
        
        # Beautiful gradient colors
        colors1 = ['#3b82f6', '#1d4ed8', '#1e40af'][:len(names)]
        colors2 = ['#10b981', '#059669', '#047857'][:len(names)]
        
        bars1 = ax.bar(x - width/2, sizes, width, label='Data (K rows)', color=colors1, alpha=0.8, edgecolor='white', linewidth=1)
        bars2 = ax.bar(x + width/2, durations, width, label='Duration (days)', color=colors2, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Styling
        ax.set_xlabel('Datasets', fontweight='bold', color='#374151')
        ax.set_ylabel('Count / Duration', fontweight='bold', color='#374151')
        ax.set_title('📊 Dataset Overview', fontweight='bold', pad=15, color='#1f2937')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0, color='#4b5563')
        
        # Beautiful legend
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=9)
        
        # Grid and spines
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.02,
                   f'{height:.0f}K', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    def _plot_ups_loads_beautiful(self, ax):
        """Create beautiful UPS load chart"""
        loads = []
        names = []
        
        for name, analysis in self.power_analysis.items():
            ups_perf = analysis.get('ups_performance', {})
            if 'load_avg' in ups_perf:
                loads.append(ups_perf['load_avg'])
                names.append(name.split('_')[1][:8] if '_' in name else name[:8])
        
        if not loads:
            ax.text(0.5, 0.5, '🔋 UPS Load Data\nNot Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='#6b7280')
            ax.set_title('🔋 UPS Load Status', fontweight='bold', pad=15, color='#1f2937')
            ax.axis('off')
            return
        
        # Color coding based on load levels
        colors = []
        for load in loads:
            if load < 60: colors.append('#10b981')      # Green - Good
            elif load < 80: colors.append('#f59e0b')    # Yellow - Warning  
            elif load < 90: colors.append('#f97316')    # Orange - High
            else: colors.append('#ef4444')              # Red - Critical
        
        bars = ax.bar(names, loads, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Threshold lines
        ax.axhline(y=80, color='#f59e0b', linestyle='--', alpha=0.6, linewidth=2)
        ax.axhline(y=90, color='#ef4444', linestyle='--', alpha=0.6, linewidth=2)
        
        # Add percentage labels on bars
        for bar, load in zip(bars, loads):
            height = bar.get_height()
            color = '#ffffff' if load > 70 else '#374151'
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{load:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=11, color=color)
        
        # Styling
        ax.set_ylabel('Load Percentage (%)', fontweight='bold', color='#374151')
        ax.set_title('🔋 UPS Load Status', fontweight='bold', pad=15, color='#1f2937')
        ax.set_ylim(0, 100)
        
        # Grid and spines
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        
        # Add legend for thresholds
        ax.text(0.02, 0.98, '— 80% Warning\n— 90% Critical', transform=ax.transAxes, 
               fontsize=8, va='top', color='#6b7280', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor='white', alpha=0.8, edgecolor='#e5e7eb'))
    
    def _plot_power_quality_beautiful(self, ax):
        """Create beautiful power quality chart"""
        voltage_imbalances = []
        names = []
        
        for name, analysis in self.power_analysis.items():
            pq = analysis.get('power_quality', {})
            if 'voltage_imbalance' in pq:
                voltage_imbalances.append(pq['voltage_imbalance'])
                names.append(name.split('_')[1][:8] if '_' in name else name[:8])
        
        if not voltage_imbalances:
            ax.text(0.5, 0.5, '⚡ Power Quality Data\nNot Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='#6b7280')
            ax.set_title('⚡ Power Quality', fontweight='bold', pad=15, color='#1f2937')
            ax.axis('off')
            return
        
        # Color coding for voltage imbalance
        colors = []
        for imb in voltage_imbalances:
            if imb < 1: colors.append('#10b981')        # Green - Excellent
            elif imb < 2: colors.append('#f59e0b')      # Yellow - Acceptable
            elif imb < 3: colors.append('#f97316')      # Orange - Poor
            else: colors.append('#ef4444')              # Red - Critical
        
        bars = ax.bar(names, voltage_imbalances, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Threshold lines
        ax.axhline(y=1, color='#f59e0b', linestyle='--', alpha=0.6, linewidth=2)
        ax.axhline(y=2, color='#ef4444', linestyle='--', alpha=0.6, linewidth=2)
        
        # Add value labels
        for bar, imb in zip(bars, voltage_imbalances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(voltage_imbalances)*0.05,
                   f'{imb:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Styling
        ax.set_ylabel('Voltage Imbalance (%)', fontweight='bold', color='#374151')
        ax.set_title('⚡ Power Quality', fontweight='bold', pad=15, color='#1f2937')
        
        # Grid and spines
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        
        # Add legend for quality levels
        ax.text(0.02, 0.98, '— 1% Good\n— 2% Acceptable', transform=ax.transAxes, 
               fontsize=8, va='top', color='#6b7280', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor='white', alpha=0.8, edgecolor='#e5e7eb'))
    
    def _create_health_dashboard_beautiful(self, ax):
        """Create beautiful system health indicators"""
        # Calculate health scores
        ups_health = self._calculate_ups_health()
        pq_health = self._calculate_power_quality_health()
        data_health = self._calculate_data_quality_health()
        
        health_metrics = [
            {"name": "UPS Health", "score": ups_health, "icon": "🔋"},
            {"name": "Power Quality", "score": pq_health, "icon": "⚡"},
            {"name": "Data Quality", "score": data_health, "icon": "📊"}
        ]
        
        # Create circular health indicators
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        for i, metric in enumerate(health_metrics):
            x_center = i + 0.5
            y_center = 0.6
            radius = 0.15
            
            # Background circle
            circle_bg = plt.Circle((x_center, y_center), radius, 
                                 color='#f3f4f6', alpha=0.5)
            ax.add_patch(circle_bg)
            
            # Health score circle
            score_normalized = metric['score'] / 100
            color = self._get_health_color(metric['score'])
            
            # Create progress arc
            theta = np.linspace(0, 2 * np.pi * score_normalized, 100)
            x_arc = x_center + (radius - 0.02) * np.cos(theta - np.pi/2)
            y_arc = y_center + (radius - 0.02) * np.sin(theta - np.pi/2)
            
            ax.plot(x_arc, y_arc, color=color, linewidth=6, alpha=0.8)
            
            # Icon and score
            ax.text(x_center, y_center + 0.03, metric['icon'], 
                   ha='center', va='center', fontsize=16)
            ax.text(x_center, y_center - 0.05, f"{metric['score']:.0f}%", 
                   ha='center', va='center', fontweight='bold', fontsize=10, color=color)
            
            # Label
            ax.text(x_center, y_center - 0.25, metric['name'], 
                   ha='center', va='center', fontsize=9, color='#374151')
        
        ax.set_title('🏥 System Health Overview', fontweight='bold', pad=15, color='#1f2937')
    
    def _create_bottom_summary(self, fig, gs):
        """Create bottom summary section"""
        # Create summary text area
        summary_text = f"""
🎯 SYSTEM STATUS: Overall system performance is {'EXCELLENT' if self._get_overall_health() >= 90 else 'GOOD' if self._get_overall_health() >= 75 else 'NEEDS ATTENTION'}
📊 DATA COVERAGE: {len(self.datasets)} datasets spanning {sum(info['time_range']['duration_hours'] for info in self.datasets.values())/24:.1f} days
⚡ POWER SYSTEMS: {'All systems operating within normal parameters' if self._get_overall_health() >= 80 else 'Some systems require monitoring'}
        """.strip()
        
        fig.text(0.5, 0.06, summary_text, ha='center', va='bottom', fontsize=10, 
                color='#374151', bbox=dict(boxstyle="round,pad=0.8", 
                facecolor='#f8fafc', alpha=0.8, edgecolor='#e2e8f0'))
    
    def _get_overall_health(self):
        """Calculate overall system health score"""
        ups_health = self._calculate_ups_health()
        pq_health = self._calculate_power_quality_health()
        data_health = self._calculate_data_quality_health()
        return (ups_health + pq_health + data_health) / 3
    
    def _create_health_dashboard(self, ax):
        """Create system health dashboard"""
        ax.axis('off')
        
        # Calculate health scores
        ups_score = self._calculate_ups_health()
        pq_score = self._calculate_power_quality_health()
        data_score = self._calculate_data_quality_health()
        overall_score = np.mean([ups_score, pq_score, data_score])
        
        # Health indicators
        indicators = [
            ('UPS Performance', ups_score, self.colors['primary']),
            ('Power Quality', pq_score, self.colors['success']),
            ('Data Quality', data_score, self.colors['info']),
            ('Overall Health', overall_score, self._get_health_color(overall_score))
        ]
        
        # Create health bars
        y_positions = [0.8, 0.6, 0.4, 0.15]
        bar_height = 0.1
        
        for i, (name, score, color) in enumerate(indicators):
            y_pos = y_positions[i]
            
            # Background bar
            rect_bg = Rectangle((0.1, y_pos), 0.6, bar_height, facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect_bg)
            
            # Score bar
            rect_score = Rectangle((0.1, y_pos), 0.6 * (score / 100), bar_height, facecolor=color, alpha=0.8)
            ax.add_patch(rect_score)
            
            # Labels
            ax.text(0.05, y_pos + bar_height/2, name, ha='right', va='center', fontweight='bold')
            ax.text(0.75, y_pos + bar_height/2, f'{score:.0f}%', ha='left', va='center', fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('System Health Dashboard', fontsize=12, fontweight='bold', pad=20)
    
    def _create_power_analysis(self, pdf):
        """Create detailed power analysis page"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("Detailed Power System Analysis", fontsize=16, fontweight='bold')
        
        # Create time series plots and detailed analysis
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        
        # Plot actual data if available
        if self.datasets:
            # Get first dataset for plotting
            first_dataset = list(self.datasets.values())[0]['data']
            
            # UPS Load over time
            ax1 = fig.add_subplot(gs[0, :])
            if 'ups_load' in first_dataset.columns:
                sample_data = first_dataset.tail(1000)  # Last 1000 points
                ax1.plot(sample_data.index, sample_data['ups_load'], color=self.colors['primary'], linewidth=1)
                ax1.set_title('UPS Load Over Time (Sample)')
                ax1.set_ylabel('Load (%)')
                ax1.grid(True, alpha=0.3)
            
            # Voltage trends
            ax2 = fig.add_subplot(gs[1, 0])
            voltage_cols = [col for col in first_dataset.columns if 'ups_v' in col and 'out' in col][:3]
            if voltage_cols:
                sample_data = first_dataset.tail(500)
                for i, col in enumerate(voltage_cols):
                    ax2.plot(sample_data.index, sample_data[col], label=f'Phase {chr(65+i)}', alpha=0.8)
                ax2.set_title('Three-Phase Voltage')
                ax2.set_ylabel('Voltage (V)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Power distribution
            ax3 = fig.add_subplot(gs[1, 1])
            power_cols = [col for col in first_dataset.columns if 'ups_p' in col and col.endswith(('a', 'b', 'c'))][:3]
            if power_cols:
                power_data = [first_dataset[col].mean() for col in power_cols]
                phase_names = ['Phase A', 'Phase B', 'Phase C']
                ax3.pie(power_data, labels=phase_names, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Power Distribution by Phase')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_data_quality_report(self, pdf):
        """Create data quality assessment page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle("Data Quality Assessment", fontsize=16, fontweight='bold')
        
        # Data completeness
        if self.data_quality:
            datasets = list(self.data_quality.keys())
            completeness = [quality['completeness'] for quality in self.data_quality.values()]
            
            bars = ax1.bar(range(len(datasets)), completeness, color=self.colors['info'], alpha=0.8)
            ax1.set_ylabel('Completeness (%)')
            ax1.set_title('Data Completeness by Dataset')
            ax1.set_xticks(range(len(datasets)))
            ax1.set_xticklabels([d.split('_')[1][:8] if '_' in d else d[:8] for d in datasets], rotation=45)
            ax1.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Target (95%)')
            ax1.legend()
            
            # Add value labels
            for bar, comp in zip(bars, completeness):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{comp:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Column distribution
        if self.data_quality:
            ups_cols = [quality['ups_columns'] for quality in self.data_quality.values()]
            met_cols = [quality['meter_columns'] for quality in self.data_quality.values()]
            pdu_cols = [quality['pdu_columns'] for quality in self.data_quality.values()]
            
            x = np.arange(len(datasets))
            width = 0.25
            
            ax2.bar(x - width, ups_cols, width, label='UPS', color=self.colors['primary'], alpha=0.8)
            ax2.bar(x, met_cols, width, label='Meters', color=self.colors['success'], alpha=0.8)
            ax2.bar(x + width, pdu_cols, width, label='PDU', color=self.colors['warning'], alpha=0.8)
            
            ax2.set_ylabel('Column Count')
            ax2.set_title('Data Column Distribution')
            ax2.set_xticks(x)
            ax2.set_xticklabels([d.split('_')[1][:8] if '_' in d else d[:8] for d in datasets], rotation=45)
            ax2.legend()
        
        # Add summary statistics
        ax3.axis('off')
        ax3.text(0.1, 0.9, 'Data Quality Summary', fontsize=14, fontweight='bold', transform=ax3.transAxes)
        
        if self.data_quality:
            total_rows = sum(quality['total_rows'] for quality in self.data_quality.values())
            avg_completeness = np.mean([quality['completeness'] for quality in self.data_quality.values()])
            total_missing = sum(quality['missing_values'] for quality in self.data_quality.values())
            
            y_pos = 0.7
            metrics = [
                f"Total Rows Analyzed: {total_rows:,}",
                f"Average Completeness: {avg_completeness:.1f}%",
                f"Total Missing Values: {total_missing:,}",
                f"Datasets Processed: {len(self.data_quality)}"
            ]
            
            for metric in metrics:
                ax3.text(0.1, y_pos, f"• {metric}", fontsize=12, transform=ax3.transAxes)
                y_pos -= 0.15
        
        # Quality score visualization
        if self.data_quality:
            scores = [quality['completeness'] for quality in self.data_quality.values()]
            ax4.hist(scores, bins=10, color=self.colors['info'], alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Completeness (%)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Data Quality Distribution')
            ax4.axvline(x=95, color='red', linestyle='--', alpha=0.7, label='Target (95%)')
            ax4.legend()
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_recommendations(self, pdf):
        """Create recommendations page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        fig.suptitle("Recommendations & Action Items", fontsize=16, fontweight='bold')
        
        y_pos = 0.9
        
        # Group by priority
        priorities = ['High', 'Medium', 'Low']
        priority_colors = [self.colors['danger'], self.colors['warning'], self.colors['info']]
        
        for priority, color in zip(priorities, priority_colors):
            priority_recs = [r for r in self.recommendations if r['priority'] == priority]
            if priority_recs:
                ax.text(0.05, y_pos, f"{priority} Priority Items", fontsize=14, fontweight='bold', color=color)
                y_pos -= 0.05
                
                for i, rec in enumerate(priority_recs[:5], 1):
                    ax.text(0.05, y_pos, f"{i}. {rec['message']}", fontsize=11)
                    y_pos -= 0.06
                
                y_pos -= 0.03
        
        # Add general recommendations if none specific
        if not any(self.recommendations):
            general_recs = [
                "Implement continuous monitoring of power quality metrics",
                "Establish regular maintenance schedules based on load patterns", 
                "Consider predictive analytics for proactive system management",
                "Develop alerting systems for critical parameter thresholds",
                "Regular review of system performance against baseline metrics"
            ]
            
            ax.text(0.05, y_pos, "General Recommendations", fontsize=14, fontweight='bold', color=self.colors['info'])
            y_pos -= 0.05
            
            for i, rec in enumerate(general_recs, 1):
                ax.text(0.05, y_pos, f"{i}. {rec}", fontsize=11)
                y_pos -= 0.06
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # Helper methods
    def _calculate_ups_health(self):
        """Calculate UPS health score"""
        loads = []
        for analysis in self.power_analysis.values():
            ups_perf = analysis.get('ups_performance', {})
            if 'load_avg' in ups_perf:
                loads.append(ups_perf['load_avg'])
        
        if not loads:
            return 75
        
        avg_load = np.mean(loads)
        if 40 <= avg_load <= 80:
            return 95
        elif 20 <= avg_load < 40 or 80 < avg_load <= 90:
            return 80
        else:
            return 60
    
    def _calculate_power_quality_health(self):
        """Calculate power quality health score"""
        imbalances = []
        for analysis in self.power_analysis.values():
            pq = analysis.get('power_quality', {})
            if 'voltage_imbalance' in pq:
                imbalances.append(pq['voltage_imbalance'])
        
        if not imbalances:
            return 80
        
        avg_imbalance = np.mean(imbalances)
        if avg_imbalance < 1:
            return 95
        elif avg_imbalance < 2:
            return 80
        else:
            return 60
    
    def _calculate_data_quality_health(self):
        """Calculate data quality health score"""
        if not self.data_quality:
            return 85
        
        completeness_scores = [quality['completeness'] for quality in self.data_quality.values()]
        return np.mean(completeness_scores)
    
    def _get_health_color(self, score):
        """Get color based on health score"""
        if score >= 90:
            return self.colors['success']
        elif score >= 75:
            return self.colors['warning']
        else:
            return self.colors['danger']


def generate_comprehensive_report():
    """Generate a comprehensive report"""
    print("🚀 Generating Power AI Comprehensive PDF Report...")
    
    generator = PowerAIPDFReportGenerator()
    generator.load_and_analyze_data(sample_size=30000)
    report_path = generator.generate_pdf_report()
    
    print(f"📄 Report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_comprehensive_report() 