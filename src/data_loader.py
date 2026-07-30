# src/data_loader.py

import pandas as pd


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Load and preprocess the electricity consumption dataset.
        """

        # Load dataset
        df = pd.read_csv(
            self.file_path,
            sep=",",
            na_values=["?"],
            low_memory=False
        )

        # Combine Date and Time columns
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            format="%d/%m/%Y %H:%M:%S"
        )

        # Set Datetime as index
        df.set_index("Datetime", inplace=True)

        # Drop original columns
        df.drop(columns=["Date", "Time"], inplace=True)

        # Convert all columns to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Remove missing values
        df.dropna(inplace=True)

        # Sort by datetime
        df.sort_index(inplace=True)

        return df


if __name__ == "__main__":

    loader = DataLoader("data/Electricity_consumption_small.csv")
    #loader = DataLoader("data/Electricity_consumption.csv")
    data = loader.load_data()

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)
    print(data.head())

    print("\nDataset Shape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nMissing Values:")
    print(data.isnull().sum())