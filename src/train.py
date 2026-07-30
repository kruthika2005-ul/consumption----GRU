# src/train.py

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data_loader import DataLoader
from src.preprocess import Preprocessor
from src.model import GRUModel


def main():

    print("=" * 60)
    print("Electricity Load Forecasting using GRU")
    print("=" * 60)

    # Load Dataset
    loader = DataLoader("data/Electricity_consumption_small.csv")
    #loader = DataLoader("data/Electricity_consumption.csv")
    df = loader.load_data()

    # --------------------------------------------------
    # Use only first 100000 rows (FAST TRAINING)
    # --------------------------------------------------
    df = df.head(1000)

    print("Dataset Shape:", df.shape)

    # Preprocess
    preprocessor = Preprocessor(sequence_length=24)

    scaled = preprocessor.fit_transform(df)

    X, y = preprocessor.create_sequences(scaled)

    print("X Shape:", X.shape)
    print("y Shape:", y.shape)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    print("Training Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))

    # Build Model
    builder = GRUModel(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    model = builder.build_model()

    os.makedirs("models", exist_ok=True)

    checkpoint = ModelCheckpoint(
        "models/final_gru_model.keras",
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )

    early = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=256,
        callbacks=[checkpoint, early],
        verbose=1
    )

    # Save model
    model.save("models/final_gru_model.keras")

    print("\nModel Saved Successfully!")

    # Evaluate
    loss, mae = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("\n==============================")
    print("Evaluation")
    print("==============================")
    print("Loss :", loss)
    print("MAE  :", mae)


if __name__ == "__main__":
    main()