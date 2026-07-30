# src/predict.py

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

from src.data_loader import DataLoader
from src.preprocess import Preprocessor


def main():

    print("=" * 50)
    print("Electricity Load Prediction")
    print("=" * 50)

    # -------------------------
    # Load Dataset
    # -------------------------
    loader = DataLoader("data/Electricity_consumption_small.csv")
    # loader = DataLoader("data/Electricity_consumption.csv")
    df = loader.load_data()

    # -------------------------
    # Load Scaler
    # -------------------------
    scaler = joblib.load("models/scaler.pkl")

    # Scale data
    scaled_data = scaler.transform(df)

    # -------------------------
    # Create Sequences
    # -------------------------
    preprocessor = Preprocessor(sequence_length=24)

    X, y = preprocessor.create_sequences(
        scaled_data,
        target_column=0
    )

    # -------------------------
    # Load Model
    # -------------------------
    model = load_model("models/final_gru_model.keras")

    print("\nModel Loaded Successfully!")

    # -------------------------
    # Prediction
    # -------------------------
    predictions = model.predict(X, verbose=0)

    predictions = predictions.flatten()

    # -------------------------
    # Evaluation
    # -------------------------
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    print("\n==============================")
    print("Prediction Results")
    print("==============================")

    print(f"MAE  : {mae:.5f}")
    print(f"RMSE : {rmse:.5f}")

    # -------------------------
    # Display Predictions
    # -------------------------
    print("\nFirst 20 Predictions\n")

    for i in range(20):
        print(
            f"Actual : {y[i]:.4f}    "
            f"Predicted : {predictions[i]:.4f}"
        )


if __name__ == "__main__":
    main()