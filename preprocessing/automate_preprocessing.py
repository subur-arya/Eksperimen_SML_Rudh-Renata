"""
Automated Preprocessing Script for Telco Customer Churn
Author: [ISI_NAMA_ANDA]
Description: Automated data preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os


def load_data(filepath):
    """
    Load dataset from CSV file
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Data loaded: {df.shape}")
    return df


def clean_data(df):
    """
    Clean dataset by fixing data types and handling missing values
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n[1] Cleaning data...")
    df_clean = df.copy()
    
    # Convert TotalCharges to numeric
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Fill NaN values in TotalCharges with median
    nan_count = df_clean['TotalCharges'].isnull().sum()
    if nan_count > 0:
        print(f"  - Filling {nan_count} NaN values in TotalCharges...")
        df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
    
    # Drop customerID
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
        print("  - Dropped customerID column")
    
    print(f"✓ Data cleaned: {df_clean.shape}")
    return df_clean


def encode_features(df):
    """
    Encode categorical features
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        pd.DataFrame: Encoded dataframe
    """
    print("\n[2] Encoding features...")
    df_encoded = df.copy()
    
    # Encode target variable
    if 'Churn' in df_encoded.columns:
        df_encoded['Churn'] = (df_encoded['Churn'] == 'Yes').astype(int)
        print("  - Encoded target: Churn (Yes=1, No=0)")
    
    # Binary encoding for binary features
    binary_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_features:
        if col in df_encoded.columns:
            df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
    print(f"  - Binary encoded: {len(binary_features)} features")
    
    # Handle MultipleLines
    if 'MultipleLines' in df_encoded.columns:
        df_encoded['MultipleLines'] = df_encoded['MultipleLines'].replace('No phone service', 'No')
        df_encoded['MultipleLines'] = (df_encoded['MultipleLines'] == 'Yes').astype(int)
    
    # Handle internet service related features
    internet_features = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in internet_features:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].replace('No internet service', 'No')
            df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
    print(f"  - Internet features encoded: {len(internet_features)} features")
    
    # One-hot encoding for multi-class features
    multi_class_features = ['InternetService', 'Contract', 'PaymentMethod']
    existing_multiclass = [col for col in multi_class_features if col in df_encoded.columns]
    if existing_multiclass:
        df_encoded = pd.get_dummies(df_encoded, columns=existing_multiclass, drop_first=True)
        print(f"  - One-hot encoded: {existing_multiclass}")
    
    print(f"✓ Features encoded: {df_encoded.shape}")
    return df_encoded


def scale_features(X):
    """
    Scale numerical features using StandardScaler
    
    Args:
        X (pd.DataFrame): Features dataframe
        
    Returns:
        pd.DataFrame: Scaled features
        StandardScaler: Fitted scaler object
    """
    print("\n[3] Scaling features...")
    X_scaled = X.copy()
    
    # Define numerical features to scale
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    existing_numerical = [col for col in numerical_features if col in X_scaled.columns]
    
    if existing_numerical:
        scaler = StandardScaler()
        X_scaled[existing_numerical] = scaler.fit_transform(X_scaled[existing_numerical])
        print(f"  - Scaled features: {existing_numerical}")
    else:
        scaler = None
        print("  - No numerical features to scale")
    
    print(f"✓ Features scaled: {X_scaled.shape}")
    return X_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("\n[4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  - Train set: {X_train.shape}")
    print(f"  - Test set: {X_test.shape}")
    print(f"  - Train target distribution: {y_train.value_counts().to_dict()}")
    print(f"  - Test target distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir='./'):
    """
    Save preprocessed data to CSV files
    
    Args:
        X_train, X_test, y_train, y_test: Split datasets
        output_dir (str): Output directory path
    """
    print("\n[5] Saving preprocessed data...")
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine features and target
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save to CSV
    train_path = os.path.join(output_dir, 'telco_churn_train.csv')
    test_path = os.path.join(output_dir, 'telco_churn_test.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"  ✓ Saved: {train_path}")
    print(f"  ✓ Saved: {test_path}")


def preprocess_pipeline(input_file, output_dir='./'):
    """
    Complete preprocessing pipeline
    
    Args:
        input_file (str): Path to raw CSV file
        output_dir (str): Output directory for preprocessed data
    """
    print("="*80)
    print("AUTOMATED PREPROCESSING PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    df = load_data(input_file)
    
    # Step 2: Clean data
    df_clean = clean_data(df)
    
    # Step 3: Encode features
    df_encoded = encode_features(df_clean)
    
    # Step 4: Separate features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Step 5: Scale features
    X_scaled, scaler = scale_features(X)
    
    # Step 6: Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Step 7: Save preprocessed data
    save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETED!")
    print("="*80)
    print(f"\n✓ Final features: {X_scaled.shape[1]}")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Automate preprocessing for Telco Churn dataset')
    parser.add_argument('--input', type=str, default='Telco_Customer_Churn.csv',
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='./',
                       help='Output directory for preprocessed data')
    
    args = parser.parse_args()
    
    # Run preprocessing pipeline
    preprocess_pipeline(args.input, args.output)