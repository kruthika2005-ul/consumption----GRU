# src/preprocess.py

import os
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import DataLoader


class Preprocessor:

    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()

    def fit_transform(self, df):
        """
        Scale the dataset and save the scaler.
        """

        # Scale the data
        scaled_data = self.scaler.fit_transform(df)

        # Create models folder if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Save scaler
        joblib.dump(self.scaler, os.path.join("models", "scaler.pkl"))

        return scaled_data

    def transform(self, df):
        """
        Transform new data using the fitted scaler.
        """

        return self.scaler.transform(df)

    def create_sequences(self, data, target_column=0):
        """
        Create sequences for GRU.

        Parameters
        ----------
        data : numpy array
            Scaled dataset

        target_column : int
            Column index of target

        Returns
        -------
        X, y
        """

        X = []
        y = []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, target_column])

        return np.array(X), np.array(y)


if __name__ == "__main__":

    # Change this filename if your CSV has a different name
    loader = DataLoader("data/Electricity_consumption.csv")

    df = loader.load_data()

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)
    print(df.head())

    preprocessor = Preprocessor(sequence_length=24)

    scaled_data = preprocessor.fit_transform(df)

    X, y = preprocessor.create_sequences(scaled_data)

    print("\nPreprocessing Completed Successfully")
    print("=" * 50)

    print("Scaled Data Shape :", scaled_data.shape)
    print("Input Shape       :", X.shape)
    print("Target Shape      :", y.shape)

    print("\nScaler saved to:")
    print("models/scaler.pkl")