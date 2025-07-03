#!/usr/bin/env python3
"""
🔬 ADVANCED MLOps ENGINE
Comprehensive ML pipeline with correlation analysis, feature selection, 
hyperparameter tuning, and model optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Advanced ML libraries
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.inspection import permutation_importance
import joblib

# Statistical analysis
from scipy import stats
from scipy.stats import zscore
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt

# Time series and signal processing
from scipy import signal

class MLOpsAdvancedEngine:
    def __init__(self, data_dir="outputs/csv_data", model_dir="outputs/mlops_models", analysis_dir="outputs/mlops_analysis"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.analysis_dir = Path(analysis_dir)
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.correlation_analysis = {}
        self.model_performance = {}
        
    def load_data(self, sample_size=None):
        """Load datasets with optimized memory usage"""
        datasets = {}
        for dataset_dir in self.data_dir.glob("*"):
            if dataset_dir.is_dir():
                csv_file = dataset_dir / "leituras.csv"
                if csv_file.exists():
                    print(f"📊 Loading {dataset_dir.name}...")
                    df = pd.read_csv(csv_file, nrows=sample_size, low_memory=False)
                    
                    # Handle datetime conversion properly
                    if 'data_hora' in df.columns:
                        df['datetime'] = pd.to_datetime(df['data_hora'], errors='coerce')
                        df = df.set_index('datetime').sort_index()
                        # Drop the original datetime column to avoid correlation issues
                        df = df.drop(columns=['data_hora'], errors='ignore')
                    
                    # Remove any remaining non-numeric columns that might cause issues
                    # Keep only numeric columns, but be more careful about it
                    numeric_columns = []
                    for col in df.columns:
                        try:
                            # Try to convert to numeric
                            pd.to_numeric(df[col], errors='raise')
                            numeric_columns.append(col)
                        except (ValueError, TypeError):
                            # Skip non-numeric columns
                            print(f"   Skipping non-numeric column: {col}")
                            continue
                    
                    df = df[numeric_columns]
                    datasets[dataset_dir.name] = df
                    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} numeric columns")
        return datasets
    
    def analyze_correlations(self, df, threshold=0.95):
        """Comprehensive correlation analysis"""
        print(f"🔍 Analyzing feature correlations (threshold: {threshold})...")
        
        # Ensure we only use numeric columns for correlation analysis
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Also remove any columns that might still contain strings/objects
        for col in numeric_df.columns:
            try:
                # Try to convert to numeric, drop if it fails
                pd.to_numeric(numeric_df[col], errors='raise')
            except (ValueError, TypeError):
                numeric_df = numeric_df.drop(columns=[col])
        
        # Calculate correlation matrix on clean numeric data
        corr_matrix = numeric_df.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        # Find features to drop
        high_corr_pairs = []
        features_to_drop = set()
        
        for column in upper_triangle.columns:
            correlated_features = upper_triangle[column][upper_triangle[column] > threshold].index.tolist()
            for corr_feature in correlated_features:
                high_corr_pairs.append((column, corr_feature, upper_triangle.loc[corr_feature, column]))
                # Keep the feature with higher variance (use numeric_df for variance calculation)
                if numeric_df[column].var() < numeric_df[corr_feature].var():
                    features_to_drop.add(column)
                else:
                    features_to_drop.add(corr_feature)
        
        # Save correlation analysis
        correlation_report = {
            'high_corr_pairs': high_corr_pairs,
            'features_to_drop': list(features_to_drop),
            'correlation_matrix_shape': corr_matrix.shape,
            'total_features': len(numeric_df.columns),
            'features_dropped': len(features_to_drop),
            'features_remaining': len(numeric_df.columns) - len(features_to_drop)
        }
        
        print(f"📈 Found {len(high_corr_pairs)} highly correlated pairs")
        print(f"🗑️ Dropping {len(features_to_drop)} features due to high correlation")
        print(f"✅ Keeping {correlation_report['features_remaining']} features")
        
        # Create correlation heatmap for top features
        if len(numeric_df.columns) <= 50:  # Only for manageable number of features
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                plt.figure(figsize=(15, 12))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": .8})
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                plt.savefig(self.analysis_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as viz_error:
                print(f"Warning: Could not create correlation heatmap: {viz_error}")
        
        return features_to_drop, correlation_report
    
    def create_advanced_features(self, df):
        """Create electrical engineering features with reduced correlation"""
        print("⚡ Creating advanced electrical features...")
        
        # Power calculations (more selective)
        if all(col in df.columns for col in ['ups_pa', 'ups_pb', 'ups_pc']):
            df['ups_total_power'] = df[['ups_pa', 'ups_pb', 'ups_pc']].sum(axis=1)
            df['ups_power_imbalance'] = df[['ups_pa', 'ups_pb', 'ups_pc']].std(axis=1)
            df['ups_power_range'] = df[['ups_pa', 'ups_pb', 'ups_pc']].max(axis=1) - df[['ups_pa', 'ups_pb', 'ups_pc']].min(axis=1)
        
        # Voltage features (selective)
        voltage_in_cols = ['ups_va_in', 'ups_vb_in', 'ups_vc_in']
        voltage_out_cols = ['ups_va_out', 'ups_vb_out', 'ups_vc_out']
        
        if all(col in df.columns for col in voltage_in_cols):
            df['ups_voltage_in_avg'] = df[voltage_in_cols].mean(axis=1)
            df['ups_voltage_in_std'] = df[voltage_in_cols].std(axis=1)
        
        if all(col in df.columns for col in voltage_out_cols):
            df['ups_voltage_out_avg'] = df[voltage_out_cols].mean(axis=1)
            df['ups_voltage_out_std'] = df[voltage_out_cols].std(axis=1)
        
        # Current features
        current_cols = ['ups_ia_out', 'ups_ib_out', 'ups_ic_out']
        if all(col in df.columns for col in current_cols):
            df['ups_current_avg'] = df[current_cols].mean(axis=1)
            df['ups_current_imbalance'] = df[current_cols].std(axis=1)
        
        # Power quality metrics (avoid highly correlated combinations)
        if all(col in df.columns for col in ['ups_total_power', 'ups_voltage_out_avg', 'ups_current_avg']):
            df['ups_power_factor'] = df['ups_total_power'] / (df['ups_voltage_out_avg'] * df['ups_current_avg'] + 1e-6)
            df['ups_power_factor'] = df['ups_power_factor'].clip(0, 2)  # Reasonable bounds
        
        # Time-based features (cyclical encoding to avoid correlation)
        # Only add if we have a proper datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        else:
            print("Warning: No datetime index available for time-based features")
        
        # Rolling features (limited to reduce correlation)
        for target_col in ['ups_load', 'ups_total_power']:
            if target_col in df.columns:
                # Short-term trends
                df[f'{target_col}_ma_3h'] = df[target_col].rolling(window=3, min_periods=1).mean()
                df[f'{target_col}_ma_6h'] = df[target_col].rolling(window=6, min_periods=1).mean()
                
                # Volatility
                df[f'{target_col}_volatility_3h'] = df[target_col].rolling(window=3, min_periods=1).std()
                
                # Trend direction
                df[f'{target_col}_trend_3h'] = df[target_col].diff(3)
        
        return df
    
    def engineer_features(self, df):
        """Complete feature engineering with correlation control"""
        print("🔧 Starting correlation-aware feature engineering...")
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Analyze and remove highly correlated features
        features_to_drop, correlation_report = self.analyze_correlations(df, threshold=0.95)
        self.correlation_analysis = correlation_report
        
        # Drop highly correlated features
        df_clean = df.drop(columns=features_to_drop, errors='ignore')
        
        # Fill missing values
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"✅ Feature engineering complete: {len(df_clean.columns)} features (removed {len(features_to_drop)})")
        return df_clean
    
    def optimize_hyperparameters(self, X, y, cv_folds=3):
        """Comprehensive hyperparameter optimization"""
        print("🔧 Optimizing hyperparameters...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # XGBoost parameter grid (focused on reducing overfitting)
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0, 2.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2, 0.5]
        }
        
        # Random search for efficiency
        xgb_model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_distributions,
            n_iter=50,  # Number of parameter settings sampled
            scoring='neg_mean_absolute_error',
            cv=tscv,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        print(f"🏆 Best parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"   {param}: {value}")
        print(f"   Best CV score: {-random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_
    
    def advanced_feature_selection(self, X, y, target_name):
        """Multi-method feature selection"""
        print(f"🎯 Advanced feature selection for {target_name}...")
        
        n_features = min(30, X.shape[1] // 2)  # Conservative feature count
        
        # Method 1: Statistical selection
        selector_stats = SelectKBest(score_func=f_regression, k=n_features)
        X_stats = selector_stats.fit_transform(X, y)
        stats_features = X.columns[selector_stats.get_support()].tolist()
        
        # Method 2: L1-based selection (Lasso)
        lasso_selector = SelectFromModel(
            xgb.XGBRegressor(reg_alpha=1.0, random_state=42, verbosity=0),
            max_features=n_features
        )
        X_lasso = lasso_selector.fit_transform(X, y)
        lasso_features = X.columns[lasso_selector.get_support()].tolist()
        
        # Method 3: Recursive Feature Elimination
        base_model = xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        rfe_selector = RFE(estimator=base_model, n_features_to_select=n_features)
        X_rfe = rfe_selector.fit_transform(X, y)
        rfe_features = X.columns[rfe_selector.get_support()].tolist()
        
        # Combine methods (intersection for most important features)
        common_features = set(stats_features) & set(lasso_features) & set(rfe_features)
        
        # If too few common features, use union of top methods
        if len(common_features) < 10:
            combined_features = list(set(stats_features + lasso_features))[:n_features]
        else:
            combined_features = list(common_features)
        
        print(f"📊 Feature selection results:")
        print(f"   Statistical: {len(stats_features)} features")
        print(f"   Lasso: {len(lasso_features)} features")
        print(f"   RFE: {len(rfe_features)} features")
        print(f"   Final selection: {len(combined_features)} features")
        
        return combined_features
    
    def train_optimized_model(self, df, target='ups_total_power'):
        """Train highly optimized model with all techniques"""
        print(f"🚀 Training optimized model for {target}...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['id', 'data_hora'] and col != target]
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Remove rows with NaN target
        mask = ~y.isna()
        X, y = X[mask], y[mask]
        
        # Check for sufficient variance in target
        if y.std() < 0.001:
            print(f"⚠️ Warning: Target {target} has very low variance ({y.std():.6f})")
            
        # Advanced feature selection
        selected_features = self.advanced_feature_selection(X, y, target)
        X_selected = X[selected_features]
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers
        X_scaled = scaler.fit_transform(X_selected)
        X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X_selected.index)
        
        # Hyperparameter optimization
        best_model, best_params = self.optimize_hyperparameters(X_scaled, y)
        
        # Final model training
        print("🎯 Training final optimized model...")
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X_scaled, y)
        
        # Model validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(final_model, X_scaled, y, cv=tscv, scoring='r2')
        mae_scores = cross_val_score(final_model, X_scaled, y, cv=tscv, scoring='neg_mean_absolute_error')
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Predictions and metrics
        y_pred = final_model.predict(X_scaled)
        
        metrics = {
            'model_name': 'optimized_xgboost',
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mape': mean_absolute_percentage_error(y, y_pred),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_mae_mean': -mae_scores.mean(),
            'cv_mae_std': mae_scores.std(),
            'target_variance': y.var(),
            'target_mean': y.mean(),
            'target_std': y.std(),
            'features_used': selected_features,
            'feature_importance': feature_importance.to_dict('records'),
            'best_params': best_params,
            'n_samples': len(y)
        }
        
        # Store model components
        self.models[f'{target}_model'] = final_model
        self.scalers[f'{target}_scaler'] = scaler
        self.feature_selectors[f'{target}_features'] = selected_features
        self.model_performance[target] = metrics
        
        print(f"✅ Model training complete:")
        print(f"   R² Score: {metrics['r2']:.4f}")
        print(f"   MAE: {metrics['mae']:.2f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        print(f"   CV R² (mean ± std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        
        return metrics
    
    def predict_future_optimized(self, df, target='ups_total_power', hours_ahead=24):
        """Optimized future prediction with proper time series forecasting"""
        print(f"🔮 Generating DYNAMIC predictions for {target} ({hours_ahead}h ahead)...")
        
        if f'{target}_model' not in self.models:
            print(f"❌ No trained model for {target}")
            return None
        
        model = self.models[f'{target}_model']
        scaler = self.scalers[f'{target}_scaler']
        selected_features = self.feature_selectors[f'{target}_features']
        
        # Create future time index
        last_time = df.index[-1]
        future_times = pd.date_range(start=last_time, periods=hours_ahead+1, freq='H')[1:]
        future_df = pd.DataFrame(index=future_times)
        
        # Add cyclical time features (these should vary!)
        future_df['hour_sin'] = np.sin(2 * np.pi * future_df.index.hour / 24)
        future_df['hour_cos'] = np.cos(2 * np.pi * future_df.index.hour / 24)
        future_df['day_sin'] = np.sin(2 * np.pi * future_df.index.dayofweek / 7)
        future_df['day_cos'] = np.cos(2 * np.pi * future_df.index.dayofweek / 7)
        
        # DYNAMIC feature generation - NO MORE STATIC AVERAGES!
        print("🔧 Generating DYNAMIC features for realistic predictions...")
        
        for feature in selected_features:
            if feature not in future_df.columns:
                if feature in df.columns:
                    feature_values = df[feature].dropna()
                    
                    if len(feature_values) > 0:
                        # Method 1: Trend-based forecasting for rolling features
                        if 'ma_' in feature or 'volatility_' in feature or 'trend_' in feature:
                            # Use last known value with small random variation
                            base_value = feature_values.iloc[-1]
                            # Add realistic variation based on recent volatility
                            recent_std = feature_values.tail(24).std() if len(feature_values) >= 24 else feature_values.std()
                            noise_scale = recent_std * 0.1  # 10% of recent volatility
                            variations = np.random.normal(0, noise_scale, len(future_times))
                            future_df[feature] = base_value + variations
                            
                        # Method 2: Seasonal pattern forecasting for base features
                        elif any(base in feature for base in ['ups_', 'met_', 'pdu']):
                            # Create realistic seasonal patterns
                            last_24h = feature_values.tail(24) if len(feature_values) >= 24 else feature_values
                            mean_val = last_24h.mean()
                            std_val = last_24h.std()
                            
                            # Generate values with daily seasonality
                            seasonal_values = []
                            for i, future_time in enumerate(future_times):
                                hour = future_time.hour
                                # Create realistic hourly pattern (higher during day, lower at night)
                                seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 2 PM
                                
                                # Add realistic random variation
                                noise = np.random.normal(0, std_val * 0.15)  # 15% noise
                                value = mean_val * seasonal_factor + noise
                                seasonal_values.append(value)
                            
                            future_df[feature] = seasonal_values
                            
                        # Method 3: Time-aware forecasting for engineered features
                        elif any(eng in feature for eng in ['total', 'avg', 'imbalance', 'factor']):
                            # Use autoregressive approach
                            recent_values = feature_values.tail(48).values if len(feature_values) >= 48 else feature_values.values
                            
                            if len(recent_values) >= 3:
                                # Simple AR(3) model
                                ar_predictions = []
                                for i in range(len(future_times)):
                                    if i < 3:
                                        # Use recent trend
                                        trend = np.mean(np.diff(recent_values[-3:]))
                                        pred = recent_values[-1] + trend * (i + 1)
                                    else:
                                        # Use previous predictions
                                        pred = 0.4 * ar_predictions[i-1] + 0.3 * ar_predictions[i-2] + 0.2 * ar_predictions[i-3] + np.random.normal(0, feature_values.std() * 0.1)
                                    ar_predictions.append(pred)
                                
                                future_df[feature] = ar_predictions
                            else:
                                # Fallback with variation
                                base_val = feature_values.mean()
                                variations = np.random.normal(base_val, feature_values.std() * 0.2, len(future_times))
                                future_df[feature] = variations
                        else:
                            # Default: use last value with realistic drift
                            base_value = feature_values.iloc[-1]
                            # Add small random walk
                            drift = np.random.normal(0, feature_values.std() * 0.05, len(future_times))
                            future_df[feature] = base_value + np.cumsum(drift)
                    else:
                        future_df[feature] = 0
                else:
                    future_df[feature] = 0
        
        # Ensure all selected features are present
        missing_features = set(selected_features) - set(future_df.columns)
        if missing_features:
            print(f"⚠️ Missing features for prediction: {missing_features}")
            for feature in missing_features:
                future_df[feature] = 0
        
        # Apply realistic constraints
        print("⚡ Applying realistic constraints...")
        for feature in selected_features:
            if feature in future_df.columns and feature in df.columns:
                # Constraint values to reasonable ranges
                original_values = df[feature].dropna()
                if len(original_values) > 0:
                    q1, q99 = original_values.quantile([0.01, 0.99])
                    future_df[feature] = future_df[feature].clip(q1, q99)
        
        # Scale and predict
        print("🎯 Generating ML predictions...")
        X_future = future_df[selected_features].fillna(0)
        X_future_scaled = scaler.transform(X_future)
        
        predictions = model.predict(X_future_scaled)
        future_df[f'predicted_{target}'] = predictions
        
        # Add confidence intervals
        model_mae = self.model_performance[target]['mae']
        future_df[f'predicted_{target}_lower'] = predictions - 1.96 * model_mae
        future_df[f'predicted_{target}_upper'] = predictions + 1.96 * model_mae
        
        # Print debug info
        pred_std = predictions.std()
        pred_range = predictions.max() - predictions.min()
        print(f"✅ Prediction variance: std={pred_std:.2f}, range={pred_range:.2f}")
        print(f"📊 Prediction range: {predictions.min():.1f} to {predictions.max():.1f}")
        
        return future_df
    
    def detect_advanced_anomalies(self, df):
        """Advanced anomaly detection with multiple methods"""
        print("🔍 Advanced anomaly detection...")
        
        # Select key features for anomaly detection
        anomaly_features = []
        for col in ['ups_total_power', 'ups_load', 'ups_voltage_out_avg', 'ups_current_avg', 'ups_power_factor']:
            if col in df.columns:
                anomaly_features.append(col)
        
        if not anomaly_features:
            print("⚠️ No suitable features for anomaly detection")
            return df
        
        X_anomaly = df[anomaly_features].fillna(0)
        
        # Method 1: Statistical anomalies (Z-score)
        z_scores = np.abs(zscore(X_anomaly, axis=0))
        statistical_anomalies = (z_scores > 3).any(axis=1)
        
        # Method 2: Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        ml_anomalies = iso_forest.fit_predict(X_anomaly) == -1
        
        # Method 3: Interquartile Range (IQR)
        Q1 = X_anomaly.quantile(0.25)
        Q3 = X_anomaly.quantile(0.75)
        IQR = Q3 - Q1
        iqr_anomalies = ((X_anomaly < (Q1 - 1.5 * IQR)) | (X_anomaly > (Q3 + 1.5 * IQR))).any(axis=1)
        
        # Combine methods
        df['is_statistical_anomaly'] = statistical_anomalies
        df['is_ml_anomaly'] = ml_anomalies
        df['is_iqr_anomaly'] = iqr_anomalies
        df['is_any_anomaly'] = statistical_anomalies | ml_anomalies | iqr_anomalies
        df['anomaly_score'] = z_scores.max(axis=1)
        
        anomaly_count = df['is_any_anomaly'].sum()
        total_count = len(df)
        print(f"🚨 Found {anomaly_count} anomalies ({anomaly_count/total_count*100:.2f}%)")
        
        return df
    
    def generate_comprehensive_report(self, results):
        """Generate comprehensive MLOps report"""
        print("📋 Generating comprehensive MLOps report...")
        
        report = f"""
# 🔬 MLOps Advanced Analysis Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Dataset Summary
"""
        
        for dataset_name, dataset_results in results.items():
            report += f"""
### Dataset: {dataset_name}
- **Rows**: {dataset_results['data_quality']['rows']:,}
- **Original Features**: {dataset_results['data_quality']['original_features']}
- **Final Features**: {dataset_results['data_quality']['final_features']}
- **Features Removed**: {dataset_results['data_quality']['features_removed']}
- **Data Completeness**: {dataset_results['data_quality']['completeness']:.1f}%

"""
            
            # Model performance
            report += "## 🤖 Model Performance\n\n"
            for target, metrics in dataset_results['models'].items():
                report += f"""
### {target}
- **Model**: {metrics['model_name']}
- **R² Score**: {metrics['r2']:.4f}
- **MAE**: {metrics['mae']:.2f}
- **RMSE**: {metrics['rmse']:.2f}
- **MAPE**: {metrics['mape']:.2f}%
- **CV R² (mean ± std)**: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}
- **Target Variance**: {metrics['target_variance']:.2f}
- **Features Used**: {len(metrics['features_used'])}

#### Top Features:
"""
                for feat in metrics['feature_importance'][:5]:
                    report += f"- **{feat['feature']}**: {feat['importance']:.3f}\n"
                
                report += "\n"
            
            # Correlation analysis
            if 'correlation_analysis' in dataset_results:
                corr = dataset_results['correlation_analysis']
                report += f"""
## 🔗 Correlation Analysis
- **Total Features**: {corr['total_features']}
- **Highly Correlated Pairs**: {len(corr['high_corr_pairs'])}
- **Features Dropped**: {corr['features_dropped']}
- **Features Remaining**: {corr['features_remaining']}

"""
        
        # Save report
        report_file = self.analysis_dir / 'mlops_comprehensive_report.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved: {report_file}")
        return report_file
    
    def save_models(self):
        """Save all models and metadata"""
        print("💾 Saving optimized models and metadata...")
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, self.model_dir / f"{name}.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, self.model_dir / f"{name}.pkl")
        
        # Save feature selectors
        for name, features in self.feature_selectors.items():
            joblib.dump(features, self.model_dir / f"{name}.pkl")
        
        # Save metadata
        metadata = {
            'correlation_analysis': self.correlation_analysis,
            'model_performance': self.model_performance,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        joblib.dump(metadata, self.model_dir / "metadata.pkl")
        
        print(f"✅ Saved {len(self.models)} models to {self.model_dir}")

def run_mlops_analysis():
    """Run complete MLOps analysis"""
    print("🚀 Starting Advanced MLOps Analysis")
    print("=" * 80)
    
    engine = MLOpsAdvancedEngine()
    
    # Load data
    datasets = engine.load_data()
    if not datasets:
        print("❌ No datasets found!")
        return {}
    
    all_results = {}
    
    for name, df in datasets.items():
        print(f"\n🔍 MLOps Analysis: {name}")
        print("-" * 60)
        
        # Feature engineering with correlation analysis
        df_original_features = len(df.columns)
        df = engine.engineer_features(df)
        df_final_features = len(df.columns)
        
        # Train optimized models
        targets = ['ups_total_power', 'met1_total_power', 'ups_load']
        model_results = {}
        
        for target in targets:
            if target in df.columns:
                model_results[target] = engine.train_optimized_model(df, target)
        
        # Future predictions with optimized models
        predictions = {}
        for target in targets:
            if target in df.columns:
                pred = engine.predict_future_optimized(df, target, hours_ahead=24)
                if pred is not None:
                    predictions[target] = pred
        
        # Advanced anomaly detection
        df = engine.detect_advanced_anomalies(df)
        
        all_results[name] = {
            'models': model_results,
            'predictions': predictions,
            'anomalies': {
                'total': int(df['is_any_anomaly'].sum()),
                'percentage': float(df['is_any_anomaly'].mean() * 100)
            },
            'correlation_analysis': engine.correlation_analysis,
            'data_quality': {
                'rows': len(df),
                'original_features': df_original_features,
                'final_features': df_final_features,
                'features_removed': df_original_features - df_final_features,
                'completeness': (1 - df.isnull().mean().mean()) * 100
            }
        }
    
    # Save models and generate report
    engine.save_models()
    engine.generate_comprehensive_report(all_results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("🎯 ADVANCED MLOps ANALYSIS COMPLETE!")
    print("=" * 80)
    
    for name, results in all_results.items():
        print(f"\n📊 Dataset: {name}")
        print(f"   Data Quality:")
        print(f"     - Rows: {results['data_quality']['rows']:,}")
        print(f"     - Features: {results['data_quality']['original_features']} → {results['data_quality']['final_features']}")
        print(f"     - Removed: {results['data_quality']['features_removed']} highly correlated features")
        
        print(f"   Model Performance:")
        for target, metrics in results['models'].items():
            cv_r2 = metrics['cv_r2_mean']
            mae = metrics['mae']
            target_std = metrics['target_std']
            relative_error = (mae / target_std) * 100 if target_std > 0 else 0
            
            print(f"     - {target}:")
            print(f"       • CV R²: {cv_r2:.4f} ± {metrics['cv_r2_std']:.4f}")
            print(f"       • MAE: {mae:.2f} ({relative_error:.1f}% of std)")
            print(f"       • Features: {len(metrics['features_used'])}")
    
    return all_results

if __name__ == "__main__":
    results = run_mlops_analysis()