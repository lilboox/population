# Header: Read the data, fit and predict

from tensorflow import keras
import explore
import numpy as np


def prepare_data():
    """Prepare the data for training and prediction.
    TODO: train a model to predict ocean vs land, and use that in the new model.
    """

    raw_data = explore.read_resampled(resolution=5)
    # change negative values in data to 0
    # raw_data[raw_data < 0] = 0

    # change data to log scale
    raw_data = np.log(raw_data)

    # change the Nan and infinite values to 0
    raw_data[np.isnan(raw_data)] = -20
    # change infinite values to 0
    raw_data[np.isinf(raw_data)] = -20

    _, row_range, col_range = raw_data.shape

    # Flatten the 2D np array to 1D, to get the y_train
    y_train = raw_data.reshape(-1, 1)

    # Use divmod to get the X_train, i.e. latitude and longitude indices
    data_indices = np.divmod(np.arange(col_range * row_range), col_range)
    X_train = (np.vstack(data_indices)).transpose()
    return X_train, y_train, row_range, col_range


def get_model():
    """Create a Keras Dense Model"""
    model = keras.models.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1, activation="linear")
    ])
    # Train a Keras Dense Model
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

    return model


def train_model():
    # Load the data
    X_orig, y_orig, _, _ = prepare_data()

    model = get_model()

    X_train = np.tile(X_orig, (100, 1))  # copies of X_orig to train
    y_train = np.tile(y_orig, (100, 1))

    print("X_train", X_train.shape)
    print("y_train", y_train.shape)

    model.fit(X_train, y_train, epochs=5)

    # Save the Model
    model.save(f"{explore.ROOT_DIR}/my_keras_model.h5")
    return model


def apply_model(model=None):
    # Load the Model
    if model is None:
        model = keras.models.load_model(f"{explore.ROOT_DIR}/my_keras_model.h5")

    X_orig, y_orig, row_range, col_range = prepare_data()

    y_pred = model.predict(X_orig)

    explore.plot_two_data(
        y_orig.reshape(row_range, col_range),
        y_pred.reshape(row_range, col_range))


if __name__ == "__main__":
    model = train_model()
    apply_model(model)
