#!/usr/bin/env python3
"""Power AI ML Engine - Complete prediction system in 100 lines"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

class PowerAIPredictor:
    def __init__(self, data_dir="outputs/csv_data", model_dir="outputs/ml_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.scalers = {}
        
    def load_data(self, sample_size=50000):
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
    
    def engineer_features(self, df):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek >= 5
        df['total_power'] = df[['ups_pa', 'ups_pb', 'ups_pc']].sum(axis=1)
        df['power_imbalance'] = df[['ups_pa', 'ups_pb', 'ups_pc']].std(axis=1)
        df['voltage_avg'] = df[['ups_va_out', 'ups_vb_out', 'ups_vc_out']].mean(axis=1)
        df['current_avg'] = df[['ups_ia_out', 'ups_ib_out', 'ups_ic_out']].mean(axis=1)
        df['power_factor'] = df['total_power'] / (df['voltage_avg'] * df['current_avg'] + 1e-6)
        df['efficiency'] = df['ups_load'] / 100
        return df
        
    def train_consumption_model(self, df):
        features = ['hour', 'day_of_week', 'month', 'is_weekend', 'voltage_avg', 'current_avg', 'power_imbalance']
        X = df[features].fillna(0)
        y = df['total_power'].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        self.models['consumption'] = model
        self.scalers['consumption'] = scaler
        return {'mae': mae, 'r2': r2, 'predictions': predictions, 'y_test': y_test}
        
    def detect_anomalies(self, df):
        features = ['ups_load', 'voltage_avg', 'current_avg', 'power_factor', 'efficiency']
        X = df[features].fillna(0)
        model = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        anomalies = model.fit_predict(X)
        df['is_anomaly'] = anomalies == -1
        self.models['anomaly'] = model
        return df
        
    def predict_future(self, df, hours_ahead=24):
        if 'consumption' not in self.models:
            self.train_consumption_model(df)
        future_times = pd.date_range(start=df.index[-1], periods=hours_ahead+1, freq='H')[1:]
        future_df = pd.DataFrame(index=future_times)
        future_df['hour'] = future_df.index.hour
        future_df['day_of_week'] = future_df.index.dayofweek
        future_df['month'] = future_df.index.month
        future_df['is_weekend'] = future_df.index.dayofweek >= 5
        future_df['voltage_avg'] = df['voltage_avg'].mean()
        future_df['current_avg'] = df['current_avg'].mean()
        future_df['power_imbalance'] = df['power_imbalance'].mean()
        features = ['hour', 'day_of_week', 'month', 'is_weekend', 'voltage_avg', 'current_avg', 'power_imbalance']
        X_future = self.scalers['consumption'].transform(future_df[features])
        predictions = self.models['consumption'].predict(X_future)
        future_df['predicted_power'] = predictions
        return future_df
        
    def optimize_system(self, df):
        optimizations = []
        if df['power_factor'].mean() < 0.85:
            optimizations.append("Install power factor correction capacitors")
        if df['power_imbalance'].mean() > df['total_power'].mean() * 0.1:
            optimizations.append("Rebalance loads across phases")
        if df['ups_load'].mean() > 80:
            optimizations.append("Consider UPS capacity upgrade")
        if df['efficiency'].mean() < 0.8:
            optimizations.append("Implement energy efficiency measures")
        return optimizations
        
    def save_models(self):
        for name, model in self.models.items():
            joblib.dump(model, self.model_dir / f"{name}_model.pkl")
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, self.model_dir / f"{name}_scaler.pkl")
            
    def load_models(self):
        for model_file in self.model_dir.glob("*_model.pkl"):
            name = model_file.stem.replace("_model", "")
            self.models[name] = joblib.load(model_file)
        for scaler_file in self.model_dir.glob("*_scaler.pkl"):
            name = scaler_file.stem.replace("_scaler", "")
            self.scalers[name] = joblib.load(scaler_file)

def run_full_analysis():
    predictor = PowerAIPredictor()
    datasets = predictor.load_data()
    results = {}
    for name, df in datasets.items():
        df = predictor.engineer_features(df)
        consumption_results = predictor.train_consumption_model(df)
        anomaly_df = predictor.detect_anomalies(df)
        future_predictions = predictor.predict_future(df)
        optimizations = predictor.optimize_system(df)
        results[name] = {
            'consumption': consumption_results,
            'anomalies': anomaly_df['is_anomaly'].sum(),
            'future': future_predictions,
            'optimizations': optimizations
        }
    predictor.save_models()
    return results

if __name__ == "__main__":
    results = run_full_analysis()
    print("🤖 Power AI ML Engine Complete!")
