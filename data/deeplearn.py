# Header: Read the data, fit and predict

from tensorflow import keras
import explore
import numpy as np


def prepare_data():
    raw_data = explore.read_resampled(resolution=20)
    _, row_range, col_range = raw_data.shape

    # Flatten the 2D np array to 1D, to get the y_train
    y_train = raw_data.reshape(-1, 1)

    # TODO
    # try removing zeroes
    # change data ot log scale

    # Use divmod to get the X_train, i.e. latitude and longitude indices
    data_indices = np.divmod(np.arange(col_range * row_range), col_range)
    X_train = (np.vstack(data_indices)).transpose()
    return X_train, y_train, row_range, col_range


def get_model():
    """Create a Keras Dense Model"""
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[2]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1, activation="relu")
    ])
    # Train a Keras Dense Model
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

    return model


def train_model():
    # Load the data
    X_train, y_train, _, _ = prepare_data()

    model = get_model()

    # print(X_train, y_train)
    model.fit(X_train, y_train, epochs=20)

    # Save the Model
    model.save("my_keras_model.h5")


def apply_model():
    # Load the Model
    model = keras.models.load_model("my_keras_model.h5")

    X_train, y_train, row_range, col_range = prepare_data()

    y_pred = model.predict(X_train)

    explore.plot_data(y_pred.reshape(row_range, col_range))


if __name__ == "__main__":
    train_model()
    # apply_model()
