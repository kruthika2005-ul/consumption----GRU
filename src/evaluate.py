# src/evaluate.py

import joblib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_loader import DataLoader
from src.preprocess import Preprocessor


def main():

    print("=" * 60)
    print("Electricity Load Forecasting Evaluation")
    print("=" * 60)

    # -----------------------------
    # Load Dataset
    # -----------------------------
    loader = DataLoader("data/Electricity_consumption_small.csv")
   # loader = DataLoader("data/Electricity_consumption.csv")
    df = loader.load_data()

    # Use same rows as training
    df = df.head(100000)

    # -----------------------------
    # Load Scaler
    # -----------------------------
    scaler = joblib.load("models/scaler.pkl")

    scaled_data = scaler.transform(df)

    # -----------------------------
    # Create Sequences
    # -----------------------------
    preprocessor = Preprocessor(sequence_length=24)

    X, y = preprocessor.create_sequences(
        scaled_data,
        target_column=0
    )

    # -----------------------------
    # Load Model
    # -----------------------------
    model = load_model("models/final_gru_model.keras")

    print("Model Loaded Successfully!")

    # -----------------------------
    # Prediction
    # -----------------------------
    predictions = model.predict(
        X,
        batch_size=256,
        verbose=0
    )

    predictions = predictions.flatten()

    # -----------------------------
    # Metrics
    # -----------------------------
    mae = mean_absolute_error(y, predictions)

    mse = mean_squared_error(y, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y, predictions)

    print("\n==============================")
    print("Evaluation Results")
    print("==============================")

    print(f"MAE  : {mae:.6f}")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(14,6))

    plt.plot(
        y[:300],
        label="Actual"
    )

    plt.plot(
        predictions[:300],
        label="Predicted"
    )

    plt.title("Actual vs Predicted Electricity Load")

    plt.xlabel("Time")

    plt.ylabel("Scaled Power")

    plt.legend()

    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()