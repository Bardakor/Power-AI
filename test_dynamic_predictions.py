#!/usr/bin/env python3
"""
🧪 MLOps TESTING: Dynamic Predictions vs Flat Predictions
Test the improved prediction system to ensure variance
"""

from tools.mlops_advanced_engine import MLOpsAdvancedEngine
import pandas as pd
import numpy as np

def test_dynamic_predictions():
    print('🧪 TESTING DYNAMIC PREDICTIONS...')
    print('=' * 50)
    
    engine = MLOpsAdvancedEngine()
    
    # Load a small sample for testing
    datasets = engine.load_data(sample_size=5000)
    if not datasets:
        print('❌ No datasets found')
        return
    
    name, df = list(datasets.items())[0]
    print(f'📊 Testing with {name}: {len(df)} rows')
    
    # Quick feature engineering and model training
    print('🔧 Feature engineering...')
    df = engine.engineer_features(df) 
    
    if 'ups_total_power' not in df.columns:
        print('⚠️ Target column not found')
        return
        
    print('🤖 Training optimized model...')
    metrics = engine.train_optimized_model(df, 'ups_total_power')
    print(f'✅ Model trained: R²={metrics["r2"]:.3f}')
    
    # Test future predictions
    print('🔮 Testing future predictions...')
    future_pred = engine.predict_future_optimized(df, 'ups_total_power', hours_ahead=12)
    
    if future_pred is None:
        print('❌ Prediction failed')
        return
        
    predictions = future_pred['predicted_ups_total_power']
    
    print('\n🎯 PREDICTION ANALYSIS:')
    print('=' * 30)
    print(f'Mean: {predictions.mean():.1f}')
    print(f'Std: {predictions.std():.1f}')
    print(f'Min: {predictions.min():.1f}')
    print(f'Max: {predictions.max():.1f}')
    print(f'Range: {predictions.max() - predictions.min():.1f}')
    print(f'Variance: {predictions.var():.1f}')
    
    # Check if predictions vary
    variance_test = predictions.std() > 1.0
    range_test = (predictions.max() - predictions.min()) > 10.0
    
    print('\n🔍 VARIANCE TESTS:')
    print('=' * 20)
    print(f'Standard Deviation > 1.0: {variance_test} ({"✅ PASS" if variance_test else "❌ FAIL"})')
    print(f'Range > 10.0: {range_test} ({"✅ PASS" if range_test else "❌ FAIL"})')
    
    if variance_test and range_test:
        print('\n🚀 SUCCESS: Predictions have REALISTIC VARIANCE!')
        print('🎉 Dynamic forecasting is working!')
    else:
        print('\n❌ STILL FLAT: Predictions lack sufficient variance')
        print('🔧 Need more debugging...')
    
    # Show first few predictions
    print('\n📊 First 6 predictions:')
    for i, (time, pred) in enumerate(zip(future_pred.index[:6], predictions[:6])):
        print(f'   {time.strftime("%H:%M")}: {pred:.1f}')

if __name__ == "__main__":
    test_dynamic_predictions() 