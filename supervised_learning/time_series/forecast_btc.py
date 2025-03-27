#!/usr/bin/env python3
"""Module for forecasting Bitcoin proces using RNN"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import argparse

def load_data(data_dir='processed_data'):
    """Load preprocessed training and validation data"""

    # Load numpy arrays
    X_train = np.load(f'{data_dir}/X_train.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    X_val = np.load(f'{data_dir}/X_val.npy')
    y_val = np.load(f'{data_dir}/y_val.npy')

    # Load target scaler for inverse transformation
    target_scaler = joblib.load(f'{data_dir}/target_scaler.joblib')

    return X_train, y_train, X_val, y_val, target_scaler

def create_model(input_shape):
    """Create a simple RNN model for Bitcoin price forecasting"""
    model = Sequential()

    # LSTM layer
    model.add(LSTM(units=50, return_sequences=True,
                   input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.2))

    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))

    # Third LSTM layer
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=1))

    # Compile mode
    model.compile(optimizer='adam', loss='mse')

    return model

def main():
    """Main f unction for running the BTC forecasting"""
    parser = argparse.ArgumentParser(description='Train Bitcoin price'
                                    'forecasting model')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help='Directory with processed data')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save model')

    args = parser.parse_args()

    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)

    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train, y_train, X_val, y_val, target_scaler = load_data(args.data_dir)

    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")

    # Create model
    print("Creating model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_model(input_shape)

    # Model summary
    model.summary()

    # Create early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # Model eval
    print("Evaluating model...")
    loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss (MSE): {loss}")

    # Predictions
    predictions = model.predict(X_val)

    # Convert predictions back to original scale
    predictions_original = target_scaler.inverse_transform(predictions)
    y_val_original = target_scaler.inverse_transform(y_val)

    # Calculate RMSE(root mean squared error)
    rmse = np.sqrt(np.mean((predictions_original - y_val_original) ** 2))
    print(f"RMSE (in USD): ${rmse:.2f}")

    # Save model
    model_path = os.path.join(args.model_dir, 'btc_forecast_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if name == "main":
    main()
