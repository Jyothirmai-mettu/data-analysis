#!/usr/bin/env python3
"""
Data Analysis and Preprocessing Script for Calories Prediction
This script helps analyze your dataset and prepare it for machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load and analyze the dataset"""
    print("🔍 Loading and analyzing your dataset...")
    
    try:
        # Load the main dataset
        df = pd.read_csv('X_train.csv')
        print(f"✅ Successfully loaded dataset with {len(df)} records")
        
        # Display basic information
        print("\n📊 Dataset Overview:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for missing values
        print("\n🔍 Missing Values:")
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing_values[missing_values > 0])
        
        # Display data types
        print("\n📋 Data Types:")
        print(df.dtypes)
        
        # Display first few rows
        print("\n📝 First 5 rows:")
        print(df.head())
        
        # Basic statistics
        print("\n📈 Basic Statistics:")
        print(df.describe())
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def analyze_features(df):
    """Analyze individual features"""
    print("\n🔬 Feature Analysis:")
    
    # Gender analysis
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        print(f"\n👥 Gender Distribution:")
        print(f"Male (1): {gender_counts.get(1, 0)}")
        print(f"Female (0): {gender_counts.get(0, 0)}")
    
    # Age analysis
    if 'Age' in df.columns:
        print(f"\n📅 Age Statistics:")
        print(f"Min Age: {df['Age'].min()}")
        print(f"Max Age: {df['Age'].max()}")
        print(f"Mean Age: {df['Age'].mean():.2f}")
        print(f"Age Range: {df['Age'].max() - df['Age'].min()}")
    
    # Height and Weight analysis
    if 'Height' in df.columns and 'Weight' in df.columns:
        print(f"\n📏 Height & Weight:")
        print(f"Height Range: {df['Height'].min():.1f} - {df['Height'].max():.1f} cm")
        print(f"Weight Range: {df['Weight'].min():.1f} - {df['Weight'].max():.1f} kg")
        
        # Calculate BMI
        df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
        print(f"BMI Range: {df['BMI'].min():.1f} - {df['BMI'].max():.1f}")
    
    # Exercise parameters
    if 'Duration' in df.columns:
        print(f"\n⏱️ Exercise Duration:")
        print(f"Min Duration: {df['Duration'].min()} minutes")
        print(f"Max Duration: {df['Duration'].max()} minutes")
        print(f"Mean Duration: {df['Duration'].mean():.2f} minutes")
    
    if 'Heart_Rate' in df.columns:
        print(f"\n💓 Heart Rate:")
        print(f"Min HR: {df['Heart_Rate'].min()} BPM")
        print(f"Max HR: {df['Heart_Rate'].max()} BPM")
        print(f"Mean HR: {df['Heart_Rate'].mean():.2f} BPM")
    
    if 'Body_Temp' in df.columns:
        print(f"\n🌡️ Body Temperature:")
        print(f"Min Temp: {df['Body_Temp'].min()}°C")
        print(f"Max Temp: {df['Body_Temp'].max()}°C")
        print(f"Mean Temp: {df['Body_Temp'].mean():.2f}°C")
    
    # Target variable analysis
    if 'Calories' in df.columns:
        print(f"\n🔥 Calories Analysis:")
        print(f"Min Calories: {df['Calories'].min()}")
        print(f"Max Calories: {df['Calories'].max()}")
        print(f"Mean Calories: {df['Calories'].mean():.2f}")
        print(f"Std Calories: {df['Calories'].std():.2f}")

def check_data_quality(df):
    """Check data quality and identify potential issues"""
    print("\n🔍 Data Quality Check:")
    
    issues_found = []
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == 'User_ID':  # Skip ID columns
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            issues_found.append(f"Column '{col}' has {len(outliers)} outliers")
    
    # Check for unrealistic values
    if 'Age' in df.columns:
        unrealistic_age = df[(df['Age'] < 10) | (df['Age'] > 100)]
        if len(unrealistic_age) > 0:
            issues_found.append(f"Age column has {len(unrealistic_age)} unrealistic values")
    
    if 'Height' in df.columns:
        unrealistic_height = df[(df['Height'] < 100) | (df['Height'] > 250)]
        if len(unrealistic_height) > 0:
            issues_found.append(f"Height column has {len(unrealistic_height)} unrealistic values")
    
    if 'Weight' in df.columns:
        unrealistic_weight = df[(df['Weight'] < 30) | (df['Weight'] > 200)]
        if len(unrealistic_weight) > 0:
            issues_found.append(f"Weight column has {len(unrealistic_weight)} unrealistic values")
    
    if 'Heart_Rate' in df.columns:
        unrealistic_hr = df[(df['Heart_Rate'] < 40) | (df['Heart_Rate'] > 220)]
        if len(unrealistic_hr) > 0:
            issues_found.append(f"Heart Rate column has {len(unrealistic_hr)} unrealistic values")
    
    if 'Body_Temp' in df.columns:
        unrealistic_temp = df[(df['Body_Temp'] < 35) | (df['Body_Temp'] > 42)]
        if len(unrealistic_temp) > 0:
            issues_found.append(f"Body Temperature column has {len(unrealistic_temp)} unrealistic values")
    
    if issues_found:
        print("⚠️ Potential issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("✅ No major data quality issues found!")

def prepare_data_for_ml(df):
    """Prepare data for machine learning"""
    print("\n🤖 Preparing data for machine learning...")
    
    try:
        # Create a copy for ML preparation
        df_ml = df.copy()
        
        # Drop unnecessary columns
        if 'User_ID' in df_ml.columns:
            df_ml = df_ml.drop('User_ID', axis=1)
        
        # Handle Gender column
        if 'Gender' in df_ml.columns:
            if df_ml['Gender'].dtype == 'object':
                df_ml['Gender'] = df_ml['Gender'].map({'male': 1, 'female': 0})
        
        # Separate features and target
        if 'Calories' in df_ml.columns:
            X = df_ml.drop('Calories', axis=1)
            y = df_ml['Calories']
            
            print(f"✅ Features shape: {X.shape}")
            print(f"✅ Target shape: {y.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"✅ Training set: {X_train.shape}")
            print(f"✅ Test set: {X_test.shape}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            print("✅ Features scaled successfully!")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
        else:
            print("❌ 'Calories' column not found in dataset")
            return None, None, None, None, None
            
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return None, None, None, None, None

def main():
    """Main function to run the analysis"""
    print("🚀 Starting Data Analysis for Calories Prediction System")
    print("=" * 60)
    
    # Load and analyze data
    df = load_and_analyze_data()
    if df is None:
        return
    
    # Analyze features
    analyze_features(df)
    
    # Check data quality
    check_data_quality(df)
    
    # Prepare for ML
    ml_data = prepare_data_for_ml(df)
    
    if ml_data[0] is not None:
        print("\n🎉 Data preparation completed successfully!")
        print("✅ Your dataset is ready for machine learning!")
        print("\n📋 Next steps:")
        print("1. Run 'python app.py' to start the web application")
        print("2. Open http://localhost:5000 in your browser")
        print("3. Start making predictions!")
    else:
        print("\n❌ Data preparation failed. Please check your dataset format.")

if __name__ == "__main__":
    main()
