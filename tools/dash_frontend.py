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
from io import StringIO
import base64

class PowerAIDashboard:
    def __init__(self, data_dir="outputs/csv_data"):
        self.data_dir = Path(data_dir)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
        self.datasets = self.load_data()
        self.setup_layout()
        self.setup_callbacks()
        
    def load_data(self, sample_size=30000):
        datasets = {}
        if self.data_dir.exists():
            for dataset_dir in self.data_dir.glob("*"):
                if dataset_dir.is_dir():
                    csv_file = dataset_dir / "leituras.csv"
                    if csv_file.exists():
                        try:
                            df = pd.read_csv(csv_file, nrows=sample_size)
                            if 'data_hora' in df.columns:
                                df['datetime'] = pd.to_datetime(df['data_hora'])
                                df = df.set_index('datetime').sort_index()
                            datasets[dataset_dir.name] = df
                        except Exception as e:
                            print(f"Error loading {csv_file}: {e}")
        return datasets
    
    def setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🔬 Power AI MLOps Dashboard", className="text-center mb-4",
                           style={'color': '#2E86AB', 'fontWeight': 'bold'}),
                    html.P("Real-time power monitoring with optimized ML predictions and correlation analysis", 
                          className="text-center text-muted mb-4")
                ])
            ]),
            
            # File Upload Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("📁 Upload Your Power Data", className="mb-0")),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    '🎯 Drag and Drop or ',
                                    html.A('Select CSV Files', style={'color': '#007bff', 'textDecoration': 'underline'})
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '100px',
                                    'lineHeight': '100px',
                                    'borderWidth': '2px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '10px',
                                    'borderColor': '#007bff',
                                    'textAlign': 'center',
                                    'backgroundColor': '#f8f9fa',
                                    'cursor': 'pointer',
                                    'transition': 'all 0.3s ease'
                                },
                                multiple=True
                            ),
                            html.Div(id='upload-status', className="mt-3"),
                            html.Div(id='uploaded-files', className="mt-2")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
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
                                    dbc.Button("🔬 Run MLOps Analysis", id="ml-button", color="primary", className="me-2"),
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
            dcc.Store(id='dataset-store'),
            dcc.Store(id='uploaded-datasets-store', data={}),
            dcc.Download(id="download-pdf")
            
        ], fluid=True)
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('uploaded-datasets-store', 'data'),
             Output('upload-status', 'children'),
             Output('uploaded-files', 'children'),
             Output('dataset-dropdown', 'options')],
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('uploaded-datasets-store', 'data'),
            prevent_initial_call=True
        )
        def handle_file_upload(contents, filenames, current_datasets):
            if not contents:
                return current_datasets, "", "", [{'label': name.split('_')[1][:15], 'value': name} for name in self.datasets.keys()]
            
            new_datasets = current_datasets.copy()
            uploaded_files_info = []
            status_messages = []
            
            for content, filename in zip(contents, filenames):
                try:
                    # Decode the uploaded file
                    content_type, content_string = content.split(',')
                    decoded = base64.b64decode(content_string)
                    
                    # Parse CSV
                    df = pd.read_csv(StringIO(decoded.decode('utf-8')))
                    
                    # Process datetime if present
                    if 'data_hora' in df.columns:
                        df['datetime'] = pd.to_datetime(df['data_hora'])
                        df = df.set_index('datetime').sort_index()
                    elif 'datetime' in df.columns:
                        df = df.set_index('datetime').sort_index()
                    else:
                        # Create a simple range index if no datetime available
                        df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
                    
                    # Store the dataset
                    dataset_name = filename.replace('.csv', '')
                    new_datasets[dataset_name] = df.to_json(date_format='iso', orient='split')
                    
                    # Add to main datasets for immediate use
                    self.datasets[dataset_name] = df
                    
                    uploaded_files_info.append(
                        dbc.Alert([
                            html.I(className="fas fa-check-circle me-2"),
                            f"✅ {filename} - {len(df)} rows, {len(df.columns)} columns"
                        ], color="success", className="mb-2")
                    )
                    
                    status_messages.append(f"Successfully uploaded {filename}")
                    
                except Exception as e:
                    uploaded_files_info.append(
                        dbc.Alert([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            f"❌ Error processing {filename}: {str(e)}"
                        ], color="danger", className="mb-2")
                    )
                    status_messages.append(f"Error with {filename}")
            
            # Update dropdown options to include uploaded files
            all_options = [{'label': name.split('_')[1][:15] if '_' in name else name[:15], 'value': name} 
                          for name in list(self.datasets.keys())]
            
            status = dbc.Alert(
                f"📤 Processed {len(contents)} file(s): {', '.join(status_messages)}", 
                color="info"
            ) if status_messages else ""
            
            return new_datasets, status, uploaded_files_info, all_options

        @self.app.callback(
            Output('dataset-store', 'data'),
            [Input('dataset-dropdown', 'value'),
             Input('time-range', 'value'),
             Input('uploaded-datasets-store', 'data')]
        )
        def update_dataset_store(dataset_name, time_range, uploaded_datasets):
            if not dataset_name:
                return {}
            
            # Try to get from uploaded datasets first, then local datasets
            df = None
            if dataset_name in uploaded_datasets:
                df = pd.read_json(StringIO(uploaded_datasets[dataset_name]), orient='split')
                df.index = pd.to_datetime(df.index)
            elif dataset_name in self.datasets:
                df = self.datasets[dataset_name].copy()
            else:
                return {}
            
            if time_range != 'ALL' and not df.empty:
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
            
            df = pd.read_json(StringIO(dataset_json), orient='split')
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
                from tools.mlops_advanced_engine import MLOpsAdvancedEngine
                import joblib
                
                df = pd.read_json(StringIO(dataset_json), orient='split')
                df.index = pd.to_datetime(df.index)
                
                # Initialize MLOps engine
                engine = MLOpsAdvancedEngine()
                
                # Filter to numeric columns only (like the standalone MLOps engine)
                df = df.select_dtypes(include=[np.number])
                
                # Feature engineering with correlation analysis
                df_original_features = len(df.columns)
                df = engine.engineer_features(df)
                df_final_features = len(df.columns)
                
                # Train optimized models
                results = {}
                targets = ['ups_total_power', 'met1_total_power', 'ups_load']
                
                for target in targets:
                    if target in df.columns:
                        # Train optimized model
                        model_result = engine.train_optimized_model(df, target)
                        
                        # Add additional metadata for dashboard
                        model_result.update({
                            'model_name': 'Optimized XGBoost (MLOps)',
                            'features_original': df_original_features,
                            'features_final': df_final_features,
                            'features_removed': df_original_features - df_final_features
                        })
                        
                        results[target] = model_result
                        
                        # Future predictions with optimized model
                        future_pred = engine.predict_future_optimized(df, target, hours_ahead=24)
                        if future_pred is not None:
                            results[f'{target}_future'] = future_pred.to_json(date_format='iso', orient='split')
                
                # Advanced anomaly detection
                df = engine.detect_advanced_anomalies(df)
                anomalies = {
                    'total': int(df['is_any_anomaly'].sum()),
                    'percentage': float(df['is_any_anomaly'].mean() * 100),
                    'statistical': int(df['is_statistical_anomaly'].sum()),
                    'ml_based': int(df['is_ml_anomaly'].sum()),
                    'iqr_based': int(df['is_iqr_anomaly'].sum()),
                    'data': df[df['is_any_anomaly']].index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'scores': df[df['is_any_anomaly']]['anomaly_score'].tolist()
                }
                results['anomalies'] = anomalies
                
                # Add correlation analysis results
                if hasattr(engine, 'correlation_analysis') and engine.correlation_analysis:
                    results['correlation_analysis'] = engine.correlation_analysis
                
                # Enhanced system insights for MLOps
                insights = self.generate_mlops_insights(df, results, engine)
                results['insights'] = insights
                
                return results
                
            except Exception as e:
                print(f"MLOps Analysis Error: {e}")
                import traceback
                traceback.print_exc()
                return {'error': str(e)}
        
        @self.app.callback(
            [Output('download-pdf', 'data'),
             Output('upload-status', 'children', allow_duplicate=True)],
            Input('report-button', 'n_clicks'),
            State('dataset-store', 'data'),
            prevent_initial_call=True
        )
        def generate_pdf_report(n_clicks, dataset_json):
            """Generate comprehensive PDF report and trigger download"""
            if not dataset_json:
                return None, dbc.Alert("⚠️ No data selected for report generation", color="warning")
            
            try:
                print("📄 Starting PDF report generation...")
                
                # Import the PDF generator
                import sys
                sys.path.append('.')
                from tools.pdf_report_generator import PowerAIPDFReportGenerator
                
                # Initialize generator and create report
                generator = PowerAIPDFReportGenerator()
                generator.load_and_analyze_data(sample_size=30000)
                report_path = generator.generate_pdf_report()
                
                # Read the PDF file for download
                import os
                from datetime import datetime
                
                if os.path.exists(report_path):
                    # Generate a user-friendly filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_filename = f"PowerAI_Report_{timestamp}.pdf"
                    
                    # Trigger download
                    download_data = dcc.send_file(str(report_path), filename=download_filename)
                    
                    status_message = dbc.Alert([
                        html.I(className="fas fa-download me-2"),
                        f"✅ PDF Report Generated & Downloaded! ",
                        html.Br(),
                        html.Small(f"File: {download_filename}", className="text-muted")
                    ], color="success")
                    
                    return download_data, status_message
                else:
                    return None, dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "❌ PDF file not found after generation"
                    ], color="danger")
                
            except Exception as e:
                print(f"PDF Report Generation Error: {e}")
                import traceback
                traceback.print_exc()
                return None, dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"❌ Error generating PDF report: {str(e)}"
                ], color="danger")
        
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('tabs', 'active_tab'),
            Input('dataset-store', 'data'),
            Input('ml-results-store', 'data')
        )
        def update_tab_content(active_tab, dataset_json, ml_results):
            if not dataset_json:
                return html.P("No data available")
            
            df = pd.read_json(StringIO(dataset_json), orient='split')
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
    
    def generate_mlops_insights(self, df, results, engine):
        """Generate MLOps-specific insights for the dashboard"""
        insights = {
            'recommendations': [],
            'feature_analysis': {},
            'model_performance': {},
            'correlation_summary': {}
        }
        
        try:
            # Feature analysis
            for target in ['ups_total_power', 'met1_total_power', 'ups_load']:
                if target in results:
                    model_metrics = results[target]
                    cv_r2 = model_metrics.get('cv_r2_mean', 0)
                    mae = model_metrics.get('mae', 0)
                    features_used = len(model_metrics.get('features_used', []))
                    
                    insights['model_performance'][target] = {
                        'cv_r2': cv_r2,
                        'mae': mae,
                        'features_used': features_used,
                        'stability': 'High' if model_metrics.get('cv_r2_std', 1) < 0.1 else 'Medium'
                    }
                    
                    # Performance recommendations
                    if cv_r2 > 0.8:
                        insights['recommendations'].append(f"✅ {target}: Excellent model performance (CV R² = {cv_r2:.3f})")
                    elif cv_r2 > 0.6:
                        insights['recommendations'].append(f"⚠️ {target}: Good performance, consider feature tuning (CV R² = {cv_r2:.3f})")
                    else:
                        insights['recommendations'].append(f"🔧 {target}: Performance needs improvement (CV R² = {cv_r2:.3f})")
            
            # Correlation analysis insights
            if 'correlation_analysis' in results:
                corr_analysis = results['correlation_analysis']
                insights['correlation_summary'] = {
                    'features_dropped': corr_analysis.get('features_dropped', 0),
                    'features_remaining': corr_analysis.get('features_remaining', 0),
                    'correlation_pairs': len(corr_analysis.get('high_corr_pairs', []))
                }
                
                if corr_analysis.get('features_dropped', 0) > 0:
                    insights['recommendations'].append(
                        f"🧹 Removed {corr_analysis['features_dropped']} highly correlated features for better model performance"
                    )
            
            # Anomaly insights
            if 'anomalies' in results:
                anomaly_pct = results['anomalies'].get('percentage', 0)
                if anomaly_pct > 10:
                    insights['recommendations'].append(f"🚨 High anomaly rate ({anomaly_pct:.1f}%) - investigate system issues")
                elif anomaly_pct > 5:
                    insights['recommendations'].append(f"⚠️ Moderate anomaly rate ({anomaly_pct:.1f}%) - monitor closely")
                else:
                    insights['recommendations'].append(f"✅ Normal anomaly rate ({anomaly_pct:.1f}%) - system operating well")
            
            # Feature importance insights
            for target in ['ups_total_power', 'ups_load']:
                if target in results and 'feature_importance' in results[target]:
                    top_features = results[target]['feature_importance'][:3]
                    if top_features:
                        top_feature_names = [feat['feature'] for feat in top_features]
                        insights['recommendations'].append(
                            f"🎯 {target}: Key drivers are {', '.join(top_feature_names[:2])}"
                        )
            
        except Exception as e:
            print(f"Error generating MLOps insights: {e}")
            insights['recommendations'].append("⚠️ Error generating insights - check system logs")
        
        return insights
    
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
            return dbc.Alert("🔬 Run MLOps Analysis to see optimized predictions with correlation analysis", color="info")
        
        # MLOps Model performance cards with enhanced metrics
        performance_cards = []
        targets = ['ups_total_power', 'met1_total_power', 'ups_load']
        target_names = {'ups_total_power': 'UPS Power', 'met1_total_power': 'Meter 1 Power', 'ups_load': 'UPS Load'}
        
        for target in targets:
            if target in ml_results:
                result = ml_results[target]
                cv_r2 = result.get('cv_r2_mean', 0)
                cv_std = result.get('cv_r2_std', 0)
                features_used = len(result.get('features_used', []))
                
                # Color coding based on CV performance
                if cv_r2 > 0.8:
                    card_color = "success"
                    performance_icon = "🏆"
                elif cv_r2 > 0.6:
                    card_color = "warning"
                    performance_icon = "⚠️"
                else:
                    card_color = "danger"
                    performance_icon = "🔧"
                
                performance_cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(f"{performance_icon} {target_names.get(target, target)}"),
                            dbc.CardBody([
                                html.H4(f"CV R² = {cv_r2:.3f} ± {cv_std:.3f}", className=f"text-{card_color}"),
                                html.P(f"MAE: {result.get('mae', 0):.1f}"),
                                html.P(f"Features: {features_used}", className="text-muted"),
                                html.P(f"Model: {result.get('model_name', 'Optimized XGBoost')}", className="text-muted"),
                                html.Small(f"MAPE: {result.get('mape', 0):.2f}%", className="text-info")
                            ])
                        ], color=card_color, outline=True)
                    ], width=4)
                )
        
        content = [
            dbc.Row(performance_cards, className="mb-4")
        ]
        
        # MLOps Correlation Analysis Summary
        if 'correlation_analysis' in ml_results:
            corr_analysis = ml_results['correlation_analysis']
            content.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🔗 MLOps Feature Correlation Analysis"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5(f"{corr_analysis.get('total_features', 0)}", className="text-primary"),
                                        html.P("Original Features", className="mb-0")
                                    ], width=3),
                                    dbc.Col([
                                        html.H5(f"{corr_analysis.get('features_dropped', 0)}", className="text-danger"),
                                        html.P("Features Removed", className="mb-0")
                                    ], width=3),
                                    dbc.Col([
                                        html.H5(f"{corr_analysis.get('features_remaining', 0)}", className="text-success"),
                                        html.P("Features Kept", className="mb-0")
                                    ], width=3),
                                    dbc.Col([
                                        html.H5(f"{len(corr_analysis.get('high_corr_pairs', []))}", className="text-warning"),
                                        html.P("Correlated Pairs", className="mb-0")
                                    ], width=3)
                                ])
                            ])
                        ])
                    ], width=12)
                ], className="mb-4")
            )
        
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
                    future_df = pd.read_json(StringIO(ml_results[future_key]), orient='split')
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
            return dbc.Alert("🔬 Run MLOps Analysis to detect anomalies with multi-method approach", color="info")
        
        # Get anomaly information from MLOps results
        anomalies_info = ml_results.get('anomalies', {})
        anomaly_count = anomalies_info.get('total', 0)
        anomaly_percentage = anomalies_info.get('percentage', 0)
        statistical_count = anomalies_info.get('statistical', 0)
        ml_count = anomalies_info.get('ml_based', 0)
        iqr_count = anomalies_info.get('iqr_based', 0)
        
        content = [
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔬 MLOps Multi-Method Anomaly Detection"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H3(f"{anomaly_count}", className="text-warning"),
                                    html.P("Total Anomalies", className="mb-0")
                                ], width=3),
                                dbc.Col([
                                    html.H5(f"{statistical_count}", className="text-info"),
                                    html.P("Statistical (Z-Score)", className="mb-0")
                                ], width=3),
                                dbc.Col([
                                    html.H5(f"{ml_count}", className="text-success"),
                                    html.P("ML (Isolation Forest)", className="mb-0")
                                ], width=3),
                                dbc.Col([
                                    html.H5(f"{iqr_count}", className="text-secondary"),
                                    html.P("IQR-Based", className="mb-0")
                                ], width=3)
                            ]),
                            html.Hr(),
                            html.P(f"{anomaly_percentage:.2f}% of total data points flagged as anomalous", 
                                  className="text-center"),
                            html.Small("Using Z-Score (>3σ), Isolation Forest (5% contamination), and IQR (1.5×IQR) methods", 
                                     className="text-muted")
                        ])
                    ])
                ], width=12)
            ], className="mb-4")
        ]
        
        # Create anomaly visualization if we have the data
        try:
            # Try to use stored MLOps results or recreate them
            import sys
            import pandas as pd
            sys.path.append('.')
            from tools.mlops_advanced_engine import MLOpsAdvancedEngine
            
            engine = MLOpsAdvancedEngine()
            df_copy = df.copy()
            
            # Ensure datetime index is properly set
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                if 'data_hora' in df_copy.columns:
                    df_copy['datetime'] = pd.to_datetime(df_copy['data_hora'])
                    df_copy = df_copy.set_index('datetime').sort_index()
                elif 'datetime' in df_copy.columns:
                    df_copy = df_copy.set_index('datetime').sort_index()
                else:
                    # Create a simple range index if no datetime available
                    df_copy.index = pd.date_range(start='2024-01-01', periods=len(df_copy), freq='H')
            
            df_features = engine.engineer_features(df_copy)
            df_anomalies = engine.detect_advanced_anomalies(df_features)
            
            # Ensure anomaly columns exist
            if 'is_any_anomaly' not in df_anomalies.columns:
                df_anomalies['is_any_anomaly'] = False
            if 'anomaly_score' not in df_anomalies.columns:
                df_anomalies['anomaly_score'] = 0.0
            
            # Create visualization
            normal_data = df_anomalies[~df_anomalies['is_any_anomaly']]
            anomaly_data = df_anomalies[df_anomalies['is_any_anomaly']]
            
            # Power anomalies plot
            fig_power = go.Figure()
            
            # Check for power column and ensure it's numeric
            power_col = None
            for col in ['ups_total_power', 'total_power', 'ups_pa', 'ups_pb', 'ups_pc']:
                if col in df_anomalies.columns:
                    # Ensure column is numeric
                    df_anomalies[col] = pd.to_numeric(df_anomalies[col], errors='coerce')
                    power_col = col
                    break
            
            if power_col and not normal_data.empty:
                fig_power.add_trace(go.Scatter(
                    x=normal_data.index.strftime('%Y-%m-%d %H:%M'), 
                    y=normal_data[power_col].fillna(0),
                    mode='markers', 
                    name='Normal', 
                    opacity=0.6,
                    marker=dict(color='blue', size=4)
                ))
                
                if not anomaly_data.empty:
                    fig_power.add_trace(go.Scatter(
                        x=anomaly_data.index.strftime('%Y-%m-%d %H:%M'), 
                        y=anomaly_data[power_col].fillna(0),
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
            
            # Check for load column and ensure it's numeric
            load_col = None
            for col in ['ups_load', 'load', 'ups_load_avg']:
                if col in df_anomalies.columns:
                    df_anomalies[col] = pd.to_numeric(df_anomalies[col], errors='coerce')
                    load_col = col
                    break
            
            if load_col and not normal_data.empty:
                fig_load.add_trace(go.Scatter(
                    x=normal_data.index.strftime('%Y-%m-%d %H:%M'), 
                    y=normal_data[load_col].fillna(0),
                    mode='markers', 
                    name='Normal', 
                    opacity=0.6,
                    marker=dict(color='green', size=4)
                ))
                
                if not anomaly_data.empty:
                    fig_load.add_trace(go.Scatter(
                        x=anomaly_data.index.strftime('%Y-%m-%d %H:%M'), 
                        y=anomaly_data[load_col].fillna(0),
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
            # Ensure anomaly_score is numeric and handle NaN values
            anomaly_scores = pd.to_numeric(df_anomalies['anomaly_score'], errors='coerce').fillna(0)
            fig_scores.add_trace(go.Histogram(
                x=anomaly_scores,
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
            
            # Only add graphs if we have valid data
            graphs_to_add = []
            if power_col:
                graphs_to_add.append(dbc.Col([dcc.Graph(figure=fig_power)], width=6))
            if load_col:
                graphs_to_add.append(dbc.Col([dcc.Graph(figure=fig_load)], width=6))
            
            if graphs_to_add:
                content.extend([
                    dbc.Row(graphs_to_add, className="mb-4"),
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=fig_scores)], width=12)
                    ], className="mb-4")
                ])
            
            # Anomaly timeline - ensure datetime index for resampling
            if not anomaly_data.empty and isinstance(anomaly_data.index, pd.DatetimeIndex):
                try:
                    daily_anomalies = anomaly_data.resample('D').size()
                    
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Bar(
                        x=daily_anomalies.index.strftime('%Y-%m-%d'),
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
                except Exception as timeline_error:
                    print(f"Error creating anomaly timeline: {timeline_error}")
            
        except Exception as e:
            print(f"Error creating anomaly visualizations: {e}")
            content.append(
                dbc.Alert(f"Could not create detailed visualizations: {str(e)}", color="warning")
            )
        
        return html.Div(content)
    
    def run_server(self, debug=True, port=8050, host='0.0.0.0'):
        self.app.run(debug=debug, port=port, host=host)

def main():
    import os
    dashboard = PowerAIDashboard()
    port = int(os.environ.get('PORT', 8050))
    print("🚀 Starting Power AI Dashboard...")
    print(f"📱 Access dashboard at: http://localhost:{port}")
    dashboard.run_server(debug=False, port=port, host='0.0.0.0')

if __name__ == "__main__":
    main()
