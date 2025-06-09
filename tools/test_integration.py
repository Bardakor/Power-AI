#!/usr/bin/env python3
"""
Integration test script for Power AI system
Tests all 5 specialized files and their interactions
"""
import sys
sys.path.append('.')

def test_ml_engine():
    """Test ML Engine functionality"""
    print("🤖 Testing ML Engine...")
    try:
        from tools.ml_engine import PowerAIPredictor
        predictor = PowerAIPredictor()
        datasets = predictor.load_data(sample_size=1000)
        
        if datasets:
            print(f"✅ ML Engine: Loaded {len(datasets)} datasets")
            
            # Test with first dataset
            first_dataset = list(datasets.values())[0]
            df_features = predictor.engineer_features(first_dataset.copy())
            consumption_results = predictor.train_consumption_model(df_features)
            print(f"✅ ML Engine: Model R² = {consumption_results['r2']:.3f}")
            
            anomaly_df = predictor.detect_anomalies(df_features)
            anomaly_count = anomaly_df['is_anomaly'].sum()
            print(f"✅ ML Engine: Found {anomaly_count} anomalies")
            
            future_predictions = predictor.predict_future(df_features, hours_ahead=12)
            print(f"✅ ML Engine: Generated {len(future_predictions)} future predictions")
            
            optimizations = predictor.optimize_system(df_features)
            print(f"✅ ML Engine: Generated {len(optimizations)} optimization recommendations")
            
            return True
        else:
            print("⚠️ ML Engine: No data available for testing")
            return False
            
    except Exception as e:
        print(f"❌ ML Engine failed: {e}")
        return False

def test_interactive_viz():
    """Test Interactive Visualizations"""
    print("\n🎨 Testing Interactive Visualizations...")
    try:
        from tools.interactive_viz import InteractivePowerViz
        viz = InteractivePowerViz()
        datasets = viz.load_data(sample_size=1000)
        
        if datasets:
            print(f"✅ Interactive Viz: Loaded {len(datasets)} datasets")
            
            # Test key visualizations
            fig1 = viz.create_time_series_slider(datasets)
            print("✅ Interactive Viz: Time series slider created")
            
            fig2 = viz.create_3d_power_analysis(datasets)
            print("✅ Interactive Viz: 3D power analysis created")
            
            fig3 = viz.create_real_time_gauges(datasets)
            print("✅ Interactive Viz: Real-time gauges created")
            
            fig4 = viz.create_anomaly_detection_plot(datasets)
            print("✅ Interactive Viz: Anomaly detection plot created")
            
            return True
        else:
            print("⚠️ Interactive Viz: No data available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Interactive Viz failed: {e}")
        return False

def test_dash_frontend():
    """Test Dash Frontend (without starting server)"""
    print("\n📱 Testing Dash Frontend...")
    try:
        from tools.dash_frontend import PowerAIDashboard
        dashboard = PowerAIDashboard()
        
        print(f"✅ Dash Frontend: Initialized with {len(dashboard.datasets)} datasets")
        
        if dashboard.datasets:
            # Test data processing
            first_dataset_name = list(dashboard.datasets.keys())[0]
            test_df = dashboard.datasets[first_dataset_name].head(100)
            
            # Test tab creation methods
            monitoring_tab = dashboard.create_monitoring_tab(test_df)
            print("✅ Dash Frontend: Monitoring tab created")
            
            quality_tab = dashboard.create_quality_tab(test_df)
            print("✅ Dash Frontend: Quality tab created")
            
            historical_tab = dashboard.create_historical_tab(test_df)
            print("✅ Dash Frontend: Historical tab created")
            
            return True
        else:
            print("⚠️ Dash Frontend: No data available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Dash Frontend failed: {e}")
        return False

def test_ml_visualizations():
    """Test ML Visualizations"""
    print("\n📊 Testing ML Visualizations...")
    try:
        from tools.ml_visualizations import MLVisualizationEngine
        ml_viz = MLVisualizationEngine()
        
        # Test loading data and running ML
        results = ml_viz.load_data_and_run_ml()
        
        if results:
            print(f"✅ ML Visualizations: Processed {len(results)} datasets")
            
            # Test visualization creation
            fig1 = ml_viz.create_prediction_accuracy_plot(results)
            print("✅ ML Visualizations: Prediction accuracy plot created")
            
            fig2 = ml_viz.create_feature_importance_plot(results)
            print("✅ ML Visualizations: Feature importance plot created")
            
            fig3 = ml_viz.create_forecasting_confidence_plot(results)
            print("✅ ML Visualizations: Forecasting confidence plot created")
            
            fig4 = ml_viz.create_anomaly_detection_analysis(results)
            print("✅ ML Visualizations: Anomaly detection analysis created")
            
            return True
        else:
            print("⚠️ ML Visualizations: No data available for testing")
            return False
            
    except Exception as e:
        print(f"❌ ML Visualizations failed: {e}")
        return False

def test_additional_utilities():
    """Test Additional Utilities"""
    print("\n🛠️ Testing Additional Utilities...")
    try:
        from tools.additional_utilities import PowerAIUtilities, DataStreamManager
        utilities = PowerAIUtilities()
        
        print(f"✅ Additional Utilities: Initialized with config")
        
        # Test configuration management
        original_threshold = utilities.config['alert_thresholds']['high_load']
        new_thresholds = utilities.config['alert_thresholds'].copy()
        new_thresholds['high_load'] = 95
        utilities.update_config(alert_thresholds=new_thresholds)
        print("✅ Additional Utilities: Configuration updated")
        
        # Test data stream manager
        stream_manager = DataStreamManager(buffer_size=100)
        test_data = {'timestamp': '2025-06-10T12:00:00', 'ups_load': 75.5}
        stream_manager.add_data_point(test_data)
        recent_data = stream_manager.get_recent_data(1)
        print(f"✅ Additional Utilities: Data stream manager working ({len(recent_data)} points)")
        
        # Test data quality check
        import pandas as pd
        import numpy as np
        test_df = pd.DataFrame({
            'ups_load': np.random.uniform(0, 100, 100),
            'ups_va_out': np.random.uniform(220, 240, 100),
            'ups_vb_out': np.random.uniform(220, 240, 100),
            'ups_vc_out': np.random.uniform(220, 240, 100)
        })
        quality_report = utilities.data_quality_check(test_df)
        print(f"✅ Additional Utilities: Data quality check completed ({quality_report['total_records']} records)")
        
        # Test alert system
        test_df.loc[0, 'ups_load'] = 95  # Trigger high load alert
        alerts = utilities.alert_system(test_df)
        print(f"✅ Additional Utilities: Alert system generated {len(alerts)} alerts")
        
        return True
            
    except Exception as e:
        print(f"❌ Additional Utilities failed: {e}")
        return False

def test_cross_integration():
    """Test cross-file integration"""
    print("\n🔗 Testing Cross-Integration...")
    try:
        # Test ML engine -> Interactive viz integration
        from tools.ml_engine import PowerAIPredictor
        from tools.interactive_viz import InteractivePowerViz
        
        predictor = PowerAIPredictor()
        viz = InteractivePowerViz()
        
        datasets = predictor.load_data(sample_size=500)
        if datasets:
            # Test anomaly detection integration
            first_dataset = list(datasets.values())[0]
            df_features = predictor.engineer_features(first_dataset.copy())
            anomaly_df = predictor.detect_anomalies(df_features)
            
            # Create anomaly visualization
            viz_datasets = {list(datasets.keys())[0]: anomaly_df}
            anomaly_plot = viz.create_anomaly_detection_plot(viz_datasets)
            print("✅ Cross-Integration: ML → Visualization integration working")
            
            # Test ML results export
            from tools.additional_utilities import PowerAIUtilities
            utilities = PowerAIUtilities()
            
            test_results = {
                'dataset1': {
                    'consumption': {'r2': 0.85, 'mae': 2.3},
                    'anomalies': 5,
                    'optimizations': ['Test optimization']
                }
            }
            
            # Test export functionality
            summary = utilities.generate_summary_report(test_results)
            print("✅ Cross-Integration: ML → Utilities integration working")
            
            return True
        else:
            print("⚠️ Cross-Integration: No data available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Cross-Integration failed: {e}")
        return False

def main():
    """Run comprehensive integration tests"""
    print("🚀 Power AI Integration Test Suite")
    print("=" * 50)
    
    test_results = {
        'ML Engine': test_ml_engine(),
        'Interactive Visualizations': test_interactive_viz(),
        'Dash Frontend': test_dash_frontend(),
        'ML Visualizations': test_ml_visualizations(),
        'Additional Utilities': test_additional_utilities(),
        'Cross-Integration': test_cross_integration()
    }
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"📊 OVERALL: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All systems operational! Import/export issues resolved.")
    else:
        print("⚠️ Some issues remain. Check failed tests above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
