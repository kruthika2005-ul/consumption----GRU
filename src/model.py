# src/model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class GRUModel:

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self):
        """
        Build the GRU model.
        """

        model = Sequential()

        # First GRU Layer
        model.add(
            GRU(
                units=64,
                return_sequences=True,
                input_shape=self.input_shape
            )
        )
        model.add(Dropout(0.2))

        # Second GRU Layer
        model.add(
            GRU(
                units=32,
                return_sequences=False
            )
        )
        model.add(Dropout(0.2))

        # Fully Connected Layer
        model.add(Dense(16, activation="relu"))

        # Output Layer
        model.add(Dense(1))

        # Compile Model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        return model


if __name__ == "__main__":

    # Example input shape:
    # (sequence_length, number_of_features)
    input_shape = (24, 7)

    model_builder = GRUModel(input_shape)

    model = model_builder.build_model()

    print("=" * 50)
    print("GRU Model Summary")
    print("=" * 50)

    model.summary()