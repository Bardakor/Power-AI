#!/usr/bin/env python3
"""Interactive Power Data Visualizations - Advanced interactive charts in 300 lines"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import plotly.offline as pyo

class InteractivePowerViz:
    def __init__(self, data_dir="outputs/csv_data", output_dir="outputs/interactive_viz"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, sample_size=20000):
        datasets = {}
        for dataset_dir in self.data_dir.glob("*"):
            if dataset_dir.is_dir():
                csv_file = dataset_dir / "leituras.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file, nrows=sample_size)
                    df['datetime'] = pd.to_datetime(df['data_hora'])
                    df = df.set_index('datetime').sort_index()
                    datasets[dataset_dir.name] = df
        return datasets
    
    def create_time_series_slider(self, datasets):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          subplot_titles=('UPS Load (%)', 'Voltage (V)', 'Power (kW)'),
                          vertical_spacing=0.05)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (name, df) in enumerate(datasets.items()):
            color = colors[i % len(colors)]
            short_name = name.split('_')[1][:6]
            
            fig.add_trace(go.Scatter(x=df.index, y=df['ups_load'], name=f'{short_name} Load',
                                   line=dict(color=color), opacity=0.8), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ups_va_out'], name=f'{short_name} Voltage',
                                   line=dict(color=color), opacity=0.8, showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ups_pa'], name=f'{short_name} Power',
                                   line=dict(color=color), opacity=0.8, showlegend=False), row=3, col=1)
        
        fig.update_layout(height=800, title="⚡ Interactive Power Time Series",
                         xaxis3=dict(rangeslider=dict(visible=True), type="date"))
        return fig
    
    def create_3d_power_analysis(self, datasets):
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (name, df) in enumerate(datasets.items()):
            sample_df = df.sample(min(5000, len(df)))
            color = colors[i % len(colors)]
            short_name = name.split('_')[1][:6]
            
            fig.add_trace(go.Scatter3d(
                x=sample_df['ups_load'], y=sample_df['ups_va_out'], z=sample_df['ups_pa'],
                mode='markers', name=short_name, opacity=0.6,
                marker=dict(size=3, color=color),
                text=[f'Time: {t}<br>Load: {l:.1f}%<br>Voltage: {v:.1f}V<br>Power: {p:.1f}kW'
                     for t, l, v, p in zip(sample_df.index, sample_df['ups_load'],
                                         sample_df['ups_va_out'], sample_df['ups_pa'])],
                hovertemplate='%{text}<extra></extra>'))
        
        fig.update_layout(scene=dict(xaxis_title='UPS Load (%)', yaxis_title='Voltage (V)', zaxis_title='Power (kW)'),
                         title="🌐 3D Power Analysis", height=700)
        return fig
    
    def create_real_time_gauges(self, datasets):
        latest_data = {}
        for name, df in datasets.items():
            latest_data[name] = df.iloc[-1] if not df.empty else None
        
        fig = make_subplots(rows=2, cols=len(datasets), specs=[[{'type': 'indicator'}]*len(datasets),
                                                              [{'type': 'indicator'}]*len(datasets)],
                          subplot_titles=[f"{name.split('_')[1][:6]} Load" for name in datasets.keys()] +
                                       [f"{name.split('_')[1][:6]} Voltage" for name in datasets.keys()])
        
        for i, (name, data) in enumerate(latest_data.items(), 1):
            if data is not None:
                fig.add_trace(go.Indicator(mode="gauge+number+delta", value=data['ups_load'],
                                         domain={'x': [0, 1], 'y': [0, 1]},
                                         title={'text': "Load %"},
                                         delta={'reference': 50},
                                         gauge={'axis': {'range': [None, 100]},
                                               'bar': {'color': "darkblue"},
                                               'steps': [{'range': [0, 50], 'color': "lightgray"},
                                                       {'range': [50, 80], 'color': "gray"}],
                                               'threshold': {'line': {'color': "red", 'width': 4},
                                                           'thickness': 0.75, 'value': 90}}),
                             row=1, col=i)
                
                fig.add_trace(go.Indicator(mode="gauge+number", value=data['ups_va_out'],
                                         title={'text': "Voltage"},
                                         gauge={'axis': {'range': [200, 250]},
                                               'bar': {'color': "green"},
                                               'steps': [{'range': [200, 220], 'color': "lightgray"},
                                                       {'range': [240, 250], 'color': "gray"}]}),
                             row=2, col=i)
        
        fig.update_layout(height=600, title="⚡ Real-time Power Monitoring Gauges")
        return fig
    
    def create_anomaly_detection_plot(self, datasets):
        try:
            import sys
            sys.path.append('.')
            from tools.ml_engine import PowerAIPredictor
            predictor = PowerAIPredictor()
        except ImportError as e:
            print(f"ML Engine not available for anomaly detection: {e}")
            return go.Figure().add_annotation(text="ML Engine Required", showarrow=False, 
                                            x=0.5, y=0.5, xref="paper", yref="paper")
        
        fig = make_subplots(rows=len(datasets), cols=1, shared_xaxes=True,
                          subplot_titles=[name.split('_')[1][:10] for name in datasets.keys()])
        
        for i, (name, df) in enumerate(datasets.items(), 1):
            df_features = predictor.engineer_features(df.copy())
            df_anomalies = predictor.detect_anomalies(df_features)
            
            normal_data = df_anomalies[~df_anomalies['is_anomaly']]
            anomaly_data = df_anomalies[df_anomalies['is_anomaly']]
            
            fig.add_trace(go.Scatter(x=normal_data.index, y=normal_data['ups_load'],
                                   mode='markers', name='Normal', marker=dict(color='blue', size=2),
                                   opacity=0.6), row=i, col=1)
            
            if not anomaly_data.empty:
                fig.add_trace(go.Scatter(x=anomaly_data.index, y=anomaly_data['ups_load'],
                                       mode='markers', name='Anomaly', marker=dict(color='red', size=4),
                                       opacity=0.8), row=i, col=1)
        
        fig.update_layout(height=400*len(datasets), title="🚨 ML-Powered Anomaly Detection")
        return fig

def main():
    viz = InteractivePowerViz()
    print("🚀 Interactive Visualizations Complete!")

if __name__ == "__main__":
    main()
