#!/usr/bin/env python3
"""Power AI Dashboard - Complete Dash frontend in 500 lines"""
import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

class PowerAIDashboard:
    def __init__(self, data_dir="outputs/csv_data"):
        self.data_dir = Path(data_dir)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
        self.datasets = self.load_data()
        self.setup_layout()
        self.setup_callbacks()
        
    def load_data(self, sample_size=30000):
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
    
    def setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("⚡ Power AI Dashboard", className="text-center mb-4",
                           style={'color': '#2E86AB', 'fontWeight': 'bold'}),
                    html.P("Real-time power monitoring with ML predictions", 
                          className="text-center text-muted mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("🎛️ Controls", className="mb-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Dataset:"),
                                    dcc.Dropdown(
                                        id='dataset-dropdown',
                                        options=[{'label': name.split('_')[1][:15], 'value': name} 
                                               for name in self.datasets.keys()],
                                        value=list(self.datasets.keys())[0] if self.datasets else None,
                                        clearable=False
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Time Range:"),
                                    dcc.Dropdown(
                                        id='time-range',
                                        options=[
                                            {'label': 'Last 24 Hours', 'value': '24H'},
                                            {'label': 'Last 7 Days', 'value': '7D'},
                                            {'label': 'Last 30 Days', 'value': '30D'},
                                            {'label': 'All Data', 'value': 'ALL'}
                                        ],
                                        value='7D',
                                        clearable=False
                                    )
                                ], width=6)
                            ]),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("🤖 Run ML Analysis", id="ml-button", color="primary", className="me-2"),
                                    dbc.Button("📊 Generate Report", id="report-button", color="success", className="me-2"),
                                    dbc.Button("🔄 Refresh Data", id="refresh-button", color="info")
                                ])
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("📊 Key Metrics", className="mb-0")),
                        dbc.CardBody(id="metrics-cards")
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Tabs([
                dbc.Tab(label="📈 Real-time Monitoring", tab_id="monitoring"),
                dbc.Tab(label="🤖 ML Predictions", tab_id="predictions"),
                dbc.Tab(label="⚡ Power Quality", tab_id="quality"),
                dbc.Tab(label="📅 Historical Analysis", tab_id="historical"),
                dbc.Tab(label="🚨 Anomalies", tab_id="anomalies")
            ], id="tabs", active_tab="monitoring"),
            
            html.Div(id="tab-content", className="mt-4"),
            
            dcc.Interval(id='interval-component', interval=30*1000, n_intervals=0),
            dcc.Store(id='ml-results-store'),
            dcc.Store(id='dataset-store')
            
        ], fluid=True)
    
    def setup_callbacks(self):
        @self.app.callback(
            Output('dataset-store', 'data'),
            Input('dataset-dropdown', 'value'),
            Input('time-range', 'value')
        )
        def update_dataset_store(dataset_name, time_range):
            if not dataset_name or dataset_name not in self.datasets:
                return {}
            
            df = self.datasets[dataset_name].copy()
            
            if time_range != 'ALL':
                hours = {'24H': 24, '7D': 168, '30D': 720}[time_range]
                cutoff = df.index[-1] - timedelta(hours=hours)
                df = df[df.index >= cutoff]
            
            return df.to_json(date_format='iso', orient='split')
        
        @self.app.callback(
            Output('metrics-cards', 'children'),
            Input('dataset-store', 'data')
        )
        def update_metrics(dataset_json):
            if not dataset_json:
                return html.P("No data available")
            
            df = pd.read_json(dataset_json, orient='split')
            df.index = pd.to_datetime(df.index)
            
            current_load = df['ups_load'].iloc[-1] if not df.empty else 0
            avg_voltage = df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].mean().mean()
            total_power = df[['ups_pa', 'ups_pb', 'ups_pc']].sum(axis=1).iloc[-1] if not df.empty else 0
            efficiency = (total_power / (avg_voltage * df[['ups_ia_out', 'ups_ib_out', 'ups_ic_out']].mean().mean() + 1)) * 100
            
            return dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{current_load:.1f}%", className="text-primary"),
                            html.P("Current UPS Load", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{avg_voltage:.0f}V", className="text-success"),
                            html.P("Average Voltage", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{total_power:.1f}kW", className="text-warning"),
                            html.P("Total Power", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{efficiency:.1f}%", className="text-info"),
                            html.P("System Efficiency", className="mb-0")
                        ])
                    ])
                ], width=3)
            ])
        
        @self.app.callback(
            Output('ml-results-store', 'data'),
            Input('ml-button', 'n_clicks'),
            State('dataset-store', 'data'),
            prevent_initial_call=True
        )
        def run_ml_analysis(n_clicks, dataset_json):
            if not dataset_json:
                return {}
            
            try:
                import sys
                sys.path.append('.')
                from tools.advanced_ml_engine import AdvancedPowerAIPredictor
                
                df = pd.read_json(dataset_json, orient='split')
                df.index = pd.to_datetime(df.index)
                
                # Initialize advanced predictor
                predictor = AdvancedPowerAIPredictor()
                
                # Feature engineering
                df = predictor.engineer_features(df)
                
                # Train models
                results = {}
                targets = ['ups_total_power', 'met1_total_power', 'ups_load']
                
                for target in targets:
                    if target in df.columns:
                        model_result = predictor.train_advanced_models(df, target)
                        results[target] = model_result
                        
                        # Future predictions
                        future_pred = predictor.predict_future_advanced(df, target, hours_ahead=24)
                        if future_pred is not None:
                            results[f'{target}_future'] = future_pred.to_json(date_format='iso', orient='split')
                
                # Anomaly detection
                df = predictor.detect_advanced_anomalies(df)
                anomalies = {
                    'total': int(df['is_any_anomaly'].sum()),
                    'percentage': float(df['is_any_anomaly'].mean() * 100),
                    'data': df[df['is_any_anomaly']].index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                }
                results['anomalies'] = anomalies
                
                # System insights
                insights = predictor.generate_insights(df, results)
                results['insights'] = insights
                
                return results
                
            except Exception as e:
                print(f"ML Analysis Error: {e}")
                return {'error': str(e)}
        
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('tabs', 'active_tab'),
            Input('dataset-store', 'data'),
            Input('ml-results-store', 'data')
        )
        def update_tab_content(active_tab, dataset_json, ml_results):
            if not dataset_json:
                return html.P("No data available")
            
            df = pd.read_json(dataset_json, orient='split')
            df.index = pd.to_datetime(df.index)
            
            if active_tab == "monitoring":
                return self.create_monitoring_tab(df)
            elif active_tab == "predictions":
                return self.create_predictions_tab(df, ml_results)
            elif active_tab == "quality":
                return self.create_quality_tab(df)
            elif active_tab == "historical":
                return self.create_historical_tab(df)
            elif active_tab == "anomalies":
                return self.create_anomalies_tab(df, ml_results)
            
            return html.P("Tab content not available")
    
    def create_monitoring_tab(self, df):
        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(x=df.index, y=df['ups_load'], 
                                    name='UPS Load', line=dict(color='#2E86AB')))
        fig_load.update_layout(title="UPS Load Over Time", xaxis_title="Time", yaxis_title="Load (%)")
        
        fig_voltage = go.Figure()
        fig_voltage.add_trace(go.Scatter(x=df.index, y=df['ups_va_out'], name='Phase A'))
        fig_voltage.add_trace(go.Scatter(x=df.index, y=df['ups_vb_out'], name='Phase B'))
        fig_voltage.add_trace(go.Scatter(x=df.index, y=df['ups_vc_out'], name='Phase C'))
        fig_voltage.update_layout(title="Voltage by Phase", xaxis_title="Time", yaxis_title="Voltage (V)")
        
        return dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_load)], width=6),
            dbc.Col([dcc.Graph(figure=fig_voltage)], width=6)
        ])
    
    def create_predictions_tab(self, df, ml_results):
        if not ml_results or 'error' in ml_results:
            return dbc.Alert("⚡ Run ML Analysis to see advanced predictions", color="info")
        
        # Model performance cards
        performance_cards = []
        targets = ['ups_total_power', 'met1_total_power', 'ups_load']
        target_names = {'ups_total_power': 'UPS Power', 'met1_total_power': 'Meter 1 Power', 'ups_load': 'UPS Load'}
        
        for target in targets:
            if target in ml_results:
                result = ml_results[target]
                performance_cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(f"🎯 {target_names.get(target, target)}"),
                            dbc.CardBody([
                                html.H4(f"R² = {result.get('r2', 0):.3f}", className="text-success"),
                                html.P(f"MAE: {result.get('mae', 0):.1f}"),
                                html.P(f"Model: {result.get('model_name', 'XGBoost')}", className="text-muted")
                            ])
                        ])
                    ], width=4)
                )
        
        content = [
            dbc.Row(performance_cards, className="mb-4")
        ]
        
        # System insights
        if 'insights' in ml_results:
            insights = ml_results['insights']
            recommendations = insights.get('recommendations', [])
            
            if recommendations:
                content.append(
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("💡 System Recommendations"),
                                dbc.CardBody([
                                    html.Ul([html.Li(rec) for rec in recommendations[:5]])
                                ])
                            ])
                        ], width=12)
                    ], className="mb-4")
                )
        
        # Future predictions plots
        prediction_plots = []
        for target in targets:
            future_key = f'{target}_future'
            if future_key in ml_results:
                try:
                    future_df = pd.read_json(ml_results[future_key], orient='split')
                    future_df.index = pd.to_datetime(future_df.index)
                    
                    fig_forecast = go.Figure()
                    
                    # Historical data (last 100 points)
                    if target in df.columns:
                        historical_data = df[target].dropna().tail(100)
                        fig_forecast.add_trace(go.Scatter(
                            x=historical_data.index, 
                            y=historical_data.values,
                            name='Historical', 
                            line=dict(color='blue', width=2)
                        ))
                    
                    # Future predictions
                    pred_col = f'predicted_{target}'
                    if pred_col in future_df.columns:
                        fig_forecast.add_trace(go.Scatter(
                            x=future_df.index, 
                            y=future_df[pred_col],
                            name='24h Forecast', 
                            line=dict(color='red', dash='dash', width=3)
                        ))
                    
                    fig_forecast.update_layout(
                        title=f"24-Hour {target_names.get(target, target)} Forecast",
                        xaxis_title="Time", 
                        yaxis_title="Value",
                        template="plotly_white",
                        height=400
                    )
                    
                    prediction_plots.append(
                        dbc.Col([dcc.Graph(figure=fig_forecast)], width=6)
                    )
                except Exception as e:
                    print(f"Error creating plot for {target}: {e}")
        
        if prediction_plots:
            # Arrange plots in rows of 2
            for i in range(0, len(prediction_plots), 2):
                row_plots = prediction_plots[i:i+2]
                content.append(dbc.Row(row_plots, className="mb-4"))
        
        # Anomaly summary
        if 'anomalies' in ml_results:
            anomalies = ml_results['anomalies']
            content.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🚨 Anomaly Detection Summary"),
                            dbc.CardBody([
                                html.H4(f"{anomalies.get('total', 0)} anomalies", className="text-warning"),
                                html.P(f"{anomalies.get('percentage', 0):.2f}% of data points"),
                                html.P("Check Anomalies tab for detailed analysis", className="text-muted")
                            ])
                        ])
                    ], width=12)
                ], className="mb-4")
            )
        
        return html.Div(content)
    
    def create_quality_tab(self, df):
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=df['ups_load'], y=df['ups_va_out'], 
                                       mode='markers', name='Load vs Voltage'))
        fig_scatter.update_layout(title="Power Quality Analysis", 
                                xaxis_title="UPS Load (%)", yaxis_title="Voltage (V)")
        
        power_factor = df[['ups_pa', 'ups_pb', 'ups_pc']].sum(axis=1) / \
                      (df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].mean(axis=1) * 
                       df[['ups_ia_out', 'ups_ib_out', 'ups_ic_out']].mean(axis=1) + 1e-6)
        
        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(x=df.index, y=power_factor, name='Power Factor'))
        fig_pf.update_layout(title="Power Factor Over Time", xaxis_title="Time", yaxis_title="Power Factor")
        
        return dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_scatter)], width=6),
            dbc.Col([dcc.Graph(figure=fig_pf)], width=6)
        ])
    
    def create_historical_tab(self, df):
        # Select only numeric columns for resampling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]
        
        df_hourly = df_numeric.resample('H').mean()
        df_daily = df_numeric.resample('D').mean()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df_daily.index, y=df_daily['ups_load'], 
                                     name='Daily Average Load'))
        fig_trend.update_layout(title="Long-term Load Trends", xaxis_title="Date", yaxis_title="Load (%)")
        
        hourly_avg = df_hourly.groupby(df_hourly.index.hour).mean()
        fig_pattern = go.Figure()
        fig_pattern.add_trace(go.Bar(x=hourly_avg.index, y=hourly_avg['ups_load'], 
                                   name='Average Load by Hour'))
        fig_pattern.update_layout(title="Daily Load Pattern", xaxis_title="Hour", yaxis_title="Load (%)")
        
        return dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_trend)], width=6),
            dbc.Col([dcc.Graph(figure=fig_pattern)], width=6)
        ])
    
    def create_anomalies_tab(self, df, ml_results):
        if not ml_results or 'error' in ml_results:
            return dbc.Alert("⚡ Run ML Analysis to detect anomalies with advanced methods", color="info")
        
        # Get anomaly information from advanced ML results
        anomalies_info = ml_results.get('anomalies', {})
        anomaly_count = anomalies_info.get('total', 0)
        anomaly_percentage = anomalies_info.get('percentage', 0)
        
        content = [
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🚨 Advanced Anomaly Detection Results"),
                        dbc.CardBody([
                            html.H3(f"{anomaly_count} anomalies found", className="text-warning"),
                            html.P(f"{anomaly_percentage:.2f}% of total data points"),
                            html.P("Using statistical Z-score and Isolation Forest methods", className="text-muted")
                        ])
                    ])
                ], width=12)
            ], className="mb-4")
        ]
        
        # Create anomaly visualization if we have the data
        try:
            # Try to use stored ML predictor results or recreate them
            import sys
            sys.path.append('.')
            from tools.advanced_ml_engine import AdvancedPowerAIPredictor
            
            predictor = AdvancedPowerAIPredictor()
            df_copy = df.copy()
            df_features = predictor.engineer_features(df_copy)
            df_anomalies = predictor.detect_advanced_anomalies(df_features)
            
            # Create visualization
            normal_data = df_anomalies[~df_anomalies['is_any_anomaly']]
            anomaly_data = df_anomalies[df_anomalies['is_any_anomaly']]
            
            # Power anomalies plot
            fig_power = go.Figure()
            
            if 'ups_total_power' in df_anomalies.columns:
                fig_power.add_trace(go.Scatter(
                    x=normal_data.index, 
                    y=normal_data['ups_total_power'],
                    mode='markers', 
                    name='Normal', 
                    opacity=0.6,
                    marker=dict(color='blue', size=4)
                ))
                
                if not anomaly_data.empty:
                    fig_power.add_trace(go.Scatter(
                        x=anomaly_data.index, 
                        y=anomaly_data['ups_total_power'],
                        mode='markers', 
                        name='Anomaly', 
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                
                fig_power.update_layout(
                    title="Power Consumption Anomalies",
                    xaxis_title="Time", 
                    yaxis_title="Total Power (W)",
                    template="plotly_white",
                    height=400
                )
            
            # Load anomalies plot
            fig_load = go.Figure()
            fig_load.add_trace(go.Scatter(
                x=normal_data.index, 
                y=normal_data['ups_load'],
                mode='markers', 
                name='Normal', 
                opacity=0.6,
                marker=dict(color='green', size=4)
            ))
            
            if not anomaly_data.empty:
                fig_load.add_trace(go.Scatter(
                    x=anomaly_data.index, 
                    y=anomaly_data['ups_load'],
                    mode='markers', 
                    name='Anomaly', 
                    marker=dict(color='red', size=8, symbol='x')
                ))
            
            fig_load.update_layout(
                title="UPS Load Anomalies",
                xaxis_title="Time", 
                yaxis_title="UPS Load (%)",
                template="plotly_white",
                height=400
            )
            
            # Anomaly score distribution
            fig_scores = go.Figure()
            fig_scores.add_trace(go.Histogram(
                x=df_anomalies['anomaly_score'],
                nbinsx=50,
                name='Anomaly Scores',
                marker_color='orange'
            ))
            fig_scores.update_layout(
                title="Anomaly Score Distribution",
                xaxis_title="Anomaly Score", 
                yaxis_title="Frequency",
                template="plotly_white",
                height=400
            )
            
            content.extend([
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=fig_power)], width=6),
                    dbc.Col([dcc.Graph(figure=fig_load)], width=6)
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=fig_scores)], width=12)
                ], className="mb-4")
            ])
            
            # Anomaly timeline
            if not anomaly_data.empty:
                daily_anomalies = anomaly_data.resample('D').size()
                
                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Bar(
                    x=daily_anomalies.index,
                    y=daily_anomalies.values,
                    name='Daily Anomalies',
                    marker_color='red'
                ))
                fig_timeline.update_layout(
                    title="Anomaly Timeline (Daily Count)",
                    xaxis_title="Date", 
                    yaxis_title="Number of Anomalies",
                    template="plotly_white",
                    height=400
                )
                
                content.append(
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=fig_timeline)], width=12)
                    ], className="mb-4")
                )
            
        except Exception as e:
            print(f"Error creating anomaly visualizations: {e}")
            content.append(
                dbc.Alert(f"Could not create detailed visualizations: {str(e)}", color="warning")
            )
        
        return html.Div(content)
    
    def run_server(self, debug=True, port=8050):
        self.app.run(debug=debug, port=port)

def main():
    dashboard = PowerAIDashboard()
    print("🚀 Starting Power AI Dashboard...")
    print("📱 Access dashboard at: http://localhost:8050")
    dashboard.run_server(debug=False)

if __name__ == "__main__":
    main()
