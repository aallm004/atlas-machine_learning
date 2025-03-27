#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def load_data(file_path):
    """Load CSV file containing Bitcoin price data"""
    
    # Load data with column names
    column_names = ['timestamp' ,'open', 'high', 'low', 'close', 'btc_volume', 'usd_volume', 'weighted_price']
    df = pd.read_csv(file_path, names=column_names)

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('datetime', inplace=True)

    return df

def aggregate_hourly(df):
    """Aggregate data to hourly timeframe"""
    # Put to hourly data
    hourly_data = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'btc_volume': 'sum',
        'usd_volume': 'sum',
        'weighted_price': 'mean',
        'timestamp': 'first'
    })

    # Drop rows that aren't numbers
    hourly_data.dropna(inplace=True)

    return hourly_data

def create_features(df):
    """Create additional features that might help with prediction"""
    # Price changing features
    df['price_change'] = df['close'].pct_change()
    df['price_range'] = (df['high'] - df['low']) / df['low']

    # Add moving averages
    df['ma_6h'] = df['close'].rolling(window=6).mean()
    df['ma_12h'] = df['close'].rolling(window=12).mean()
    df['ma_24h'] = df['close'].rolling(window=24).mean()

    # Add volume features
    df['volume_change'] = df['btc_volume'].pct_change()

    # Drop rows with not number values
    df.dropna(inplace=True)

    return df

def create_sequence(df, look_back=24, forecast_horizon=1):
    """Create the sequences for time series prediction"""
    # Select features
    feature_columns = ['open', 'high', 'low', 'close', 'btc_volume', 'usd_volume',
                       'weighted_price', 'price_change', 'price_range',
                       'ma_6h', 'ma_12h', 'ma_24h', 'volume_change']
    
    # Normalize features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])

    # Create targete variable (Next hour's closing price)
    target_scaler = MinMaxScaler()
    target_data = target_scaler.fit_transform(df[['close']])

    X, y = [], []

    # Create sequences
    for i in range(len(scaled_data) - look_back - forecast_horizon + 1):
        X.append(scaled_data[i:(i + look_back)])
        y.append(target_data[i + look_back + forecast_horizon - 1])
    
    return np.array(X), np.array(y), scaler, target_scaler, feature_columns

def train_val_split(X, y, val_ratio=0.2):
    """Split data into training and validation sets"""

    # Use the last val_ratio portion of data for validation
    split_idx = int(len(X) * (1 - val_ratio))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val

def save_processed_data(X_train, y_train, X_val, y_val, scaler, target_scaler, feature_columns, output_dir='processed_data'):
    """Save processed data"""

    # Create directory
    os.makedirs(output_dir, exist_ok=True)

    # Save numpy arrays
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/X_val.npy', X_val)
    np.save(f'{output_dir}/y_val.npy', y_val)

    # Save scalers and feature names for later use
    import joblib
    joblib.dump(scaler, f'{output_dir}/feature_scaler.joblib')
    joblib.dump(target_scaler, f'{output_dir}/target_scaler.joblib')

    # Save feature names
    with open(f'{output_dir}/feature_columns.txt', 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    print(f"Processed data saved to {output_dir}:")
    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess Bitcoin price data for RNN training')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='processed_data', help='Directory to save processed data')
    parser.add_argument('--lookback', type=int, default=24, help='Number of past hours to use for prediction')
    parser.add_argument('--horizon', type=int, default=1, help='Number of future hours to predict')

    args = parser.parse_args()

    # lOad and preprocess data
    print(f"Loading data from {args.input}...")
    raw_data = load_data(args.input)
    print(f"Raw data shape: {raw_data.shape}")

    # Aggregate to hourly data
    hourly_data = aggregate_hourly(raw_data)
    print(f"Hourly data shape: {hourly_data.shape}")

    # Create features
    featured_data = create_features(hourly_data)
    print(f"Featured data shape: {featured_data.shape}")

    # Create sequences
    X, y, scaler, target_scaler, feature_columns = create_sequence(
        featured_data,
        look_back=args.lookback,
        forecast_horizon=args.horizon
    )
    print(f"Sequences shape: X={X.shape}, y={y.shape}")

    # Split into training and val sets
    X_train, y_train, X_val, y_val = train_val_split(X, y)

    # Save processed data
    save_processed_data(
        X_train, y_train, X_val, y_val,
        scaler, target_scaler, feature_columns,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
