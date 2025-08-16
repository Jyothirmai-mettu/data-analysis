#!/usr/bin/env python3
"""
Batch Prediction Script for Calories Prediction System
This script allows you to test multiple predictions and evaluate model performance.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_models():
    """Load the trained models and scaler"""
    try:
        # Load the best model and scaler
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✅ Models loaded successfully!")
        return model, scaler
    except FileNotFoundError:
        print("❌ Model files not found. Please run the web app first to train models.")
        return None, None

def make_batch_predictions(model, scaler, test_data):
    """Make predictions on test data"""
    try:
        # Prepare features (same order as training)
        features = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        
        # Extract features from test data
        X_test = test_data[features].values
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return None

def evaluate_predictions(y_true, y_pred):
    """Evaluate prediction performance"""
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print("\n📊 Prediction Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
    except Exception as e:
        print(f"❌ Error evaluating predictions: {e}")
        return None

def create_sample_test_data():
    """Create sample test data for demonstration"""
    print("\n🧪 Creating sample test data...")
    
    # Sample test cases
    test_cases = [
        {
            'Gender': 1, 'Age': 25, 'Height': 175, 'Weight': 70,
            'Duration': 30, 'Heart_Rate': 140, 'Body_Temp': 39.5
        },
        {
            'Gender': 0, 'Age': 30, 'Height': 160, 'Weight': 55,
            'Duration': 45, 'Heart_Rate': 150, 'Body_Temp': 40.0
        },
        {
            'Gender': 1, 'Age': 40, 'Height': 180, 'Weight': 80,
            'Duration': 20, 'Heart_Rate': 120, 'Body_Temp': 38.8
        },
        {
            'Gender': 0, 'Age': 35, 'Height': 165, 'Weight': 60,
            'Duration': 60, 'Heart_Rate': 160, 'Body_Temp': 40.2
        },
        {
            'Gender': 1, 'Age': 28, 'Height': 170, 'Weight': 65,
            'Duration': 15, 'Heart_Rate': 110, 'Body_Temp': 38.5
        }
    ]
    
    # Create DataFrame
    test_df = pd.DataFrame(test_cases)
    
    # Add some realistic calorie values (these would normally be unknown)
    # We'll use these to demonstrate the evaluation
    realistic_calories = [180, 220, 120, 280, 90]
    test_df['Actual_Calories'] = realistic_calories
    
    print("✅ Sample test data created!")
    return test_df

def test_with_real_data():
    """Test with a subset of real data"""
    try:
        print("\n🔍 Testing with real data subset...")
        
        # Load the main dataset
        df = pd.read_csv('X_train.csv')
        
        # Take a small subset for testing (last 100 records)
        test_subset = df.tail(100).copy()
        
        # Prepare features
        features = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        X_test = test_subset[features].values
        y_true = test_subset['Calories'].values
        
        return X_test, y_true, test_subset
        
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        return None, None, None

def main():
    """Main function"""
    print("🚀 Starting Batch Prediction Testing")
    print("=" * 50)
    
    # Load models
    model, scaler = load_models()
    if model is None or scaler is None:
        return
    
    # Test 1: Sample data
    print("\n" + "="*50)
    print("🧪 TEST 1: Sample Test Data")
    print("="*50)
    
    sample_data = create_sample_test_data()
    print("\nSample test cases:")
    print(sample_data[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']])
    
    # Make predictions
    predictions = make_batch_predictions(model, scaler, sample_data)
    if predictions is not None:
        sample_data['Predicted_Calories'] = predictions.round(2)
        
        print("\n📊 Sample Predictions:")
        print(sample_data[['Gender', 'Age', 'Duration', 'Heart_Rate', 'Actual_Calories', 'Predicted_Calories']])
        
        # Evaluate
        evaluate_predictions(sample_data['Actual_Calories'], sample_data['Predicted_Calories'])
    
    # Test 2: Real data subset
    print("\n" + "="*50)
    print("🔍 TEST 2: Real Data Subset")
    print("="*50)
    
    X_test, y_true, test_subset = test_with_real_data()
    if X_test is not None:
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        
        # Evaluate
        evaluate_predictions(y_true, predictions)
        
        # Show some examples
        print("\n📝 Sample Real Predictions:")
        sample_results = pd.DataFrame({
            'Actual': y_true[:10],
            'Predicted': predictions[:10].round(2),
            'Difference': (y_true[:10] - predictions[:10]).round(2)
        })
        print(sample_results)
    
    print("\n🎉 Batch prediction testing completed!")
    print("\n💡 Tips:")
    print("- The Random Forest model typically provides the best accuracy")
    print("- Predictions are more accurate when input values are within expected ranges")
    print("- Use the web interface for interactive predictions")

if __name__ == "__main__":
    main()
