# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from src.data_loader import DataLoader
from src.preprocess import Preprocessor


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Electricity Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electricity Load Forecasting using GRU")

st.write(
    """
Predict future electricity load using a trained
GRU (Gated Recurrent Unit) Deep Learning model.
"""
)

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

@st.cache_resource
def load_files():

    model = load_model("models/final_gru_model.keras")

    scaler = joblib.load("models/scaler.pkl")

    loader = DataLoader("data/Electricity_consumption.csv")

    df = loader.load_data()

    return model, scaler, df


try:

    model, scaler, df = load_files()

except Exception as e:

    st.error(f"Error Loading Files\n\n{e}")

    st.stop()


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

st.subheader("Dataset")

st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)


# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

st.subheader("Forecast")

sequence_length = 24

scaled = scaler.transform(df)

preprocessor = Preprocessor(sequence_length)

X, y = preprocessor.create_sequences(scaled)

prediction = model.predict(X, verbose=0)

prediction = prediction.flatten()

st.success("Prediction Completed Successfully")


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y, prediction)

mse = mean_squared_error(y, prediction)

rmse = np.sqrt(mse)

r2 = r2_score(y, prediction)

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:.5f}")

col2.metric("MSE", f"{mse:.5f}")

col3.metric("RMSE", f"{rmse:.5f}")

col4.metric("R² Score", f"{r2:.4f}")


# -------------------------------------------------------
# Graph
# -------------------------------------------------------

st.subheader("Actual vs Predicted")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    y[:300],
    label="Actual"
)

ax.plot(
    prediction[:300],
    label="Predicted"
)

ax.set_xlabel("Time")

ax.set_ylabel("Scaled Load")

ax.legend()

ax.grid(True)

st.pyplot(fig)


# -------------------------------------------------------
# Sample Predictions
# -------------------------------------------------------

st.subheader("Sample Predictions")

results = pd.DataFrame({

    "Actual": y[:20],

    "Predicted": prediction[:20]

})

st.dataframe(results)


# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.markdown("---")

st.write(
    "Developed using Python, TensorFlow, GRU, Streamlit and Scikit-learn."
)