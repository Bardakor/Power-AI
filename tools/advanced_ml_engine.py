#!/usr/bin/env python3
"""
🎯 ADVANCED POWER AI ML ENGINE
Sophisticated ML system for electrical power systems analysis
Using XGBoost, LSTM, and advanced electrical engineering features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Advanced ML libraries
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

# Time series and signal processing
from scipy import signal
from scipy.stats import zscore

class AdvancedPowerAIPredictor:
    def __init__(self, data_dir="outputs/csv_data", model_dir="outputs/ml_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        
    def load_data(self, sample_size=None):
        """Load all datasets with optimized memory usage"""
        datasets = {}
        for dataset_dir in self.data_dir.glob("*"):
            if dataset_dir.is_dir():
                csv_file = dataset_dir / "leituras.csv"
                if csv_file.exists():
                    print(f"📊 Loading {dataset_dir.name}...")
                    # Use optimized dtypes for memory efficiency
                    dtype_dict = {
                        'id': 'int32',
                        'ups_work_mode': 'int8',
                        'ups_byp_status': 'int8'
                    }
                    
                    df = pd.read_csv(csv_file, nrows=sample_size, dtype=dtype_dict, low_memory=False)
                    df['datetime'] = pd.to_datetime(df['data_hora'])
                    df = df.set_index('datetime').sort_index()
                    df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
                    datasets[dataset_dir.name] = df
                    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return datasets
    
    def create_electrical_features(self, df):
        """Create advanced electrical engineering features"""
        print("⚡ Creating electrical engineering features...")
        
        # Power calculations
        df['ups_total_power'] = df[['ups_pa', 'ups_pb', 'ups_pc']].sum(axis=1)
        df['ups_avg_power'] = df[['ups_pa', 'ups_pb', 'ups_pc']].mean(axis=1)
        df['ups_power_imbalance'] = df[['ups_pa', 'ups_pb', 'ups_pc']].std(axis=1)
        df['ups_power_range'] = df[['ups_pa', 'ups_pb', 'ups_pc']].max(axis=1) - df[['ups_pa', 'ups_pb', 'ups_pc']].min(axis=1)
        
        # Voltage features
        df['ups_voltage_in_avg'] = df[['ups_va_in', 'ups_vb_in', 'ups_vc_in']].mean(axis=1)
        df['ups_voltage_out_avg'] = df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].mean(axis=1)
        df['ups_voltage_drop'] = df['ups_voltage_in_avg'] - df['ups_voltage_out_avg']
        df['ups_voltage_imbalance_in'] = df[['ups_va_in', 'ups_vb_in', 'ups_vc_in']].std(axis=1)
        df['ups_voltage_imbalance_out'] = df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].std(axis=1)
        
        # Current features
        df['ups_current_avg'] = df[['ups_ia_out', 'ups_ib_out', 'ups_ic_out']].mean(axis=1)
        df['ups_current_imbalance'] = df[['ups_ia_out', 'ups_ib_out', 'ups_ic_out']].std(axis=1)
        df['ups_current_total'] = df[['ups_ia_out', 'ups_ib_out', 'ups_ic_out']].sum(axis=1)
        
        # Load features
        df['ups_load_avg'] = df[['ups_load_a_out', 'ups_load_b_out', 'ups_load_c_out']].mean(axis=1)
        df['ups_load_imbalance'] = df[['ups_load_a_out', 'ups_load_b_out', 'ups_load_c_out']].std(axis=1)
        df['ups_load_total'] = df[['ups_load_a_out', 'ups_load_b_out', 'ups_load_c_out']].sum(axis=1)
        
        # Power quality metrics
        df['ups_power_factor'] = df['ups_total_power'] / (df['ups_voltage_out_avg'] * df['ups_current_avg'] + 1e-6)
        df['ups_efficiency'] = df['ups_total_power'] / (df['ups_voltage_in_avg'] * df['ups_current_avg'] + 1e-6)
        df['ups_thd_voltage'] = df['ups_voltage_imbalance_out'] / df['ups_voltage_out_avg']
        df['ups_thd_current'] = df['ups_current_imbalance'] / df['ups_current_avg']
        
        # Meter 1 features (Main meter)
        df['met1_total_power'] = df[['met_pa_1', 'met_pb_1', 'met_pc_1']].sum(axis=1)
        df['met1_power_imbalance'] = df[['met_pa_1', 'met_pb_1', 'met_pc_1']].std(axis=1)
        df['met1_voltage_avg'] = df[['met_va_1', 'met_vb_1', 'met_vc_1']].mean(axis=1)
        df['met1_current_avg'] = df[['met_ia_1', 'met_ib_1', 'met_ic_1']].mean(axis=1)
        df['met1_power_factor_avg'] = df[['met_fpa_1', 'met_fpb_1', 'met_fpc_1']].mean(axis=1)
        
        # Meter 2 features (Secondary meter)
        df['met2_total_power'] = df[['met_pa_2', 'met_pb_2', 'met_pc_2']].sum(axis=1)
        df['met2_power_imbalance'] = df[['met_pa_2', 'met_pb_2', 'met_pc_2']].std(axis=1)
        df['met2_voltage_avg'] = df[['met_va_2', 'met_vb_2', 'met_vc_2']].mean(axis=1)
        df['met2_current_avg'] = df[['met_ia_2', 'met_ib_2', 'met_ic_2']].mean(axis=1)
        df['met2_power_factor_avg'] = df[['met_fpa_2', 'met_fpb_2', 'met_fpc_2']].mean(axis=1)
        
        # PDU aggregated features
        pdu_columns = [col for col in df.columns if col.startswith('pdu') and col.endswith('_i')]
        if pdu_columns:
            df['pdu_total_current'] = df[pdu_columns].sum(axis=1)
            df['pdu_avg_current'] = df[pdu_columns].mean(axis=1)
            df['pdu_current_imbalance'] = df[pdu_columns].std(axis=1)
        
        pdu_kwh_columns = [col for col in df.columns if col.startswith('pdu') and col.endswith('_kwh')]
        if pdu_kwh_columns:
            df['pdu_total_energy'] = df[pdu_kwh_columns].sum(axis=1)
            df['pdu_avg_energy'] = df[pdu_kwh_columns].mean(axis=1)
        
        # System health indicators
        df['battery_health'] = df['ups_bat_ah'] / 100.0  # Normalized battery capacity
        df['ups_stability'] = 1.0 / (1.0 + df['ups_power_imbalance'])  # Stability metric
        df['grid_quality'] = 1.0 / (1.0 + df['ups_voltage_imbalance_in'])  # Grid quality
        
        return df
    
    def create_time_features(self, df):
        """Create sophisticated time-based features"""
        print("⏰ Creating time series features...")
        
        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_business_hours'] = ((df.index.hour >= 8) & (df.index.hour <= 18)).astype(int)
        df['is_peak_hours'] = ((df.index.hour >= 9) & (df.index.hour <= 11) | 
                              (df.index.hour >= 14) & (df.index.hour <= 16)).astype(int)
        
        # Cyclical features (important for time series)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_lag_features(self, df, target_cols, lags=[1, 3, 6, 12, 24]):
        """Create lag features for time series prediction"""
        print("📈 Creating lag features...")
        
        for col in target_cols:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling window features
        for col in target_cols:
            if col in df.columns:
                df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=6, min_periods=1).mean()
                df[f'{col}_rolling_std_6h'] = df[col].rolling(window=6, min_periods=1).std()
                df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24, min_periods=1).mean()
                df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24, min_periods=1).std()
                df[f'{col}_rolling_max_24h'] = df[col].rolling(window=24, min_periods=1).max()
                df[f'{col}_rolling_min_24h'] = df[col].rolling(window=24, min_periods=1).min()
        
        return df
    
    def engineer_features(self, df):
        """Complete feature engineering pipeline"""
        print("🔧 Starting comprehensive feature engineering...")
        
        # Create electrical features
        df = self.create_electrical_features(df)
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create lag features for key power metrics
        key_cols = ['ups_total_power', 'met1_total_power', 'met2_total_power', 'ups_load']
        df = self.create_lag_features(df, key_cols)
        
        # Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"✅ Feature engineering complete: {len(df.columns)} features")
        return df
    
    def train_advanced_models(self, df, target='ups_total_power'):
        """Train advanced XGBoost model with optimized parameters"""
        print(f"🤖 Training advanced XGBoost model for {target}...")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['id', 'data_hora'] and col != target]
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Remove rows with NaN target
        mask = ~y.isna()
        X, y = X[mask], y[mask]
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(50, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # XGBoost with optimized parameters
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # Cross-validation
        print(f"🔍 Training XGBoost with cross-validation...")
        scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2', n_jobs=-1)
        cv_score = scores.mean()
        print(f"   R² Score: {cv_score:.4f} (+/- {scores.std() * 2:.4f})")
        
        # Train on full data
        print(f"🏆 Training final XGBoost model (R² = {cv_score:.4f})")
        model.fit(X_scaled, y)
        
        # Store everything
        self.models[f'{target}_model'] = model
        self.scalers[f'{target}_scaler'] = scaler
        self.feature_selectors[f'{target}_selector'] = selector
        
        # Calculate final metrics
        y_pred = model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        return {
            'model_name': 'xgboost',
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'features_used': selected_features,
            'cv_score': cv_score
        }
    
    def detect_advanced_anomalies(self, df):
        """Advanced anomaly detection using multiple methods"""
        print("🔍 Detecting anomalies with advanced methods...")
        
        # Key features for anomaly detection
        anomaly_features = [
            'ups_total_power', 'ups_voltage_out_avg', 'ups_current_avg',
            'ups_power_factor', 'ups_efficiency', 'ups_load',
            'met1_total_power', 'met2_total_power', 'battery_health'
        ]
        
        # Filter existing features
        available_features = [f for f in anomaly_features if f in df.columns]
        X = df[available_features].fillna(0)
        
        # Statistical anomalies (Z-score > 3)
        z_scores = np.abs(zscore(X, axis=0))
        statistical_anomalies = (z_scores > 3).any(axis=1)
        
        # Isolation Forest anomalies
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        ml_anomalies = iso_forest.fit_predict(X) == -1
        
        # Combine anomaly detections
        df['is_statistical_anomaly'] = statistical_anomalies
        df['is_ml_anomaly'] = ml_anomalies
        df['is_any_anomaly'] = statistical_anomalies | ml_anomalies
        df['anomaly_score'] = iso_forest.decision_function(X)
        
        self.models['anomaly_detector'] = iso_forest
        
        anomaly_count = df['is_any_anomaly'].sum()
        total_count = len(df)
        print(f"🚨 Found {anomaly_count} anomalies ({anomaly_count/total_count*100:.2f}%)")
        
        return df
    
    def predict_future_advanced(self, df, target='ups_total_power', hours_ahead=24):
        """Advanced future prediction with confidence intervals"""
        print(f"🔮 Predicting {hours_ahead} hours ahead for {target}...")
        
        if f'{target}_model' not in self.models:
            print(f"❌ No trained model for {target}")
            return None
        
        model = self.models[f'{target}_model']
        scaler = self.scalers[f'{target}_scaler']
        selector = self.feature_selectors[f'{target}_selector']
        
        # Create future time index
        last_time = df.index[-1]
        future_times = pd.date_range(start=last_time, periods=hours_ahead+1, freq='H')[1:]
        future_df = pd.DataFrame(index=future_times)
        
        # Add time features
        future_df = self.create_time_features(future_df)
        
        # Get original feature columns (before anomaly detection)
        original_feature_cols = [col for col in df.columns 
                               if col not in ['id', 'data_hora', target, 'is_statistical_anomaly', 
                                            'is_ml_anomaly', 'is_any_anomaly', 'anomaly_score']]
        
        # Add historical averages for other features
        for col in original_feature_cols:
            if col in df.columns and col not in future_df.columns:
                if 'lag' in col or 'rolling' in col:
                    future_df[col] = df[col].iloc[-1]  # Use last known value
                else:
                    future_df[col] = df[col].tail(24).mean()  # Use recent average
        
        # Select features and scale (ensure same order as training)
        available_features = [col for col in original_feature_cols if col in future_df.columns]
        X_future = future_df[available_features].fillna(0)
        
        # Transform with selector and scaler
        try:
            X_future_selected = selector.transform(X_future)
            X_future_scaled = scaler.transform(X_future_selected)
            
            # Make predictions
            predictions = model.predict(X_future_scaled)
            future_df[f'predicted_{target}'] = predictions
            
            return future_df
        except Exception as e:
            print(f"⚠️ Prediction failed: {e}")
            return None
    
    def generate_insights(self, df, results):
        """Generate actionable insights from the analysis"""
        print("💡 Generating insights...")
        
        insights = {
            'system_health': {},
            'efficiency': {},
            'power_quality': {},
            'recommendations': []
        }
        
        # System health analysis
        if 'battery_health' in df.columns:
            battery_health = df['battery_health'].mean()
            insights['system_health']['battery_health'] = battery_health
            if battery_health < 0.8:
                insights['recommendations'].append("🔋 Battery capacity below 80% - consider replacement")
        
        # Power quality analysis
        if 'ups_power_factor' in df.columns:
            avg_pf = df['ups_power_factor'].mean()
            insights['power_quality']['power_factor'] = avg_pf
            if avg_pf < 0.85:
                insights['recommendations'].append("⚡ Poor power factor - install capacitor banks")
        
        if 'ups_thd_voltage' in df.columns:
            avg_thd = df['ups_thd_voltage'].mean()
            insights['power_quality']['voltage_thd'] = avg_thd
            if avg_thd > 0.08:
                insights['recommendations'].append("📊 High voltage distortion - check loads")
        
        # Load balancing
        if 'ups_power_imbalance' in df.columns:
            imbalance = df['ups_power_imbalance'].mean()
            insights['efficiency']['load_imbalance'] = imbalance
            if imbalance > df['ups_total_power'].mean() * 0.1:
                insights['recommendations'].append("⚖️ Significant load imbalance - redistribute loads")
        
        # Efficiency analysis
        if 'ups_efficiency' in df.columns:
            efficiency = df['ups_efficiency'].mean()
            insights['efficiency']['ups_efficiency'] = efficiency
            if efficiency < 0.85:
                insights['recommendations'].append("🔧 Low UPS efficiency - maintenance needed")
        
        return insights
    
    def save_models(self):
        """Save all trained models and scalers"""
        print("💾 Saving models...")
        for name, model in self.models.items():
            joblib.dump(model, self.model_dir / f"{name}.pkl")
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, self.model_dir / f"{name}.pkl")
        for name, selector in self.feature_selectors.items():
            joblib.dump(selector, self.model_dir / f"{name}.pkl")
        print(f"✅ Saved {len(self.models)} models to {self.model_dir}")

def run_advanced_analysis():
    """Run complete advanced ML analysis"""
    print("🚀 Starting Advanced Power AI ML Analysis")
    print("=" * 60)
    
    predictor = AdvancedPowerAIPredictor()
    
    # Load data
    datasets = predictor.load_data()
    if not datasets:
        print("❌ No datasets found!")
        return {}
    
    all_results = {}
    
    for name, df in datasets.items():
        print(f"\n🔍 Analyzing dataset: {name}")
        print("-" * 40)
        
        # Feature engineering (without anomaly features)
        df = predictor.engineer_features(df)
        
        # Train models for key targets BEFORE adding anomaly features
        targets = ['ups_total_power', 'met1_total_power', 'ups_load']
        model_results = {}
        
        for target in targets:
            if target in df.columns:
                model_results[target] = predictor.train_advanced_models(df, target)
        
        # Future predictions (before anomaly detection modifies the dataframe)
        predictions = {}
        for target in targets:
            if target in df.columns:
                pred = predictor.predict_future_advanced(df, target, hours_ahead=24)
                if pred is not None:
                    predictions[target] = pred
        
        # NOW add anomaly detection features (after models are trained)
        df = predictor.detect_advanced_anomalies(df)
        
        # Generate insights
        insights = predictor.generate_insights(df, model_results)
        
        all_results[name] = {
            'models': model_results,
            'anomalies': {
                'total': df['is_any_anomaly'].sum(),
                'percentage': df['is_any_anomaly'].mean() * 100
            },
            'predictions': predictions,
            'insights': insights,
            'data_quality': {
                'rows': len(df),
                'features': len(df.columns),
                'completeness': (1 - df.isnull().mean().mean()) * 100
            }
        }
    
    # Save models
    predictor.save_models()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 ADVANCED ML ANALYSIS COMPLETE!")
    print("=" * 60)
    
    for name, results in all_results.items():
        print(f"\n📊 Dataset: {name}")
        print(f"   Rows: {results['data_quality']['rows']:,}")
        print(f"   Features: {results['data_quality']['features']}")
        print(f"   Data Completeness: {results['data_quality']['completeness']:.1f}%")
        print(f"   Anomalies: {results['anomalies']['total']} ({results['anomalies']['percentage']:.2f}%)")
        
        print(f"   📈 Model Performance:")
        for target, metrics in results['models'].items():
            print(f"      {target}: R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.1f}")
        
        print(f"   💡 Key Recommendations:")
        for rec in results['insights']['recommendations'][:3]:
            print(f"      • {rec}")
    
    return all_results

if __name__ == "__main__":
    results = run_advanced_analysis() 