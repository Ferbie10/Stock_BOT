import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import keras
import datetime
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_lstm_model(hp, num_features):
    model = Sequential()

    # Input layer
    model.add(LSTM(units=hp.Int('input_units', min_value=30, max_value=200, step=10),
                   return_sequences=True,
                   input_shape=(hp.Int('sequence_length', min_value=10,
                                max_value=100, step=10), num_features),
                   kernel_regularizer=regularizers.l2(hp.Float('l2_reg_input', 1e-4, 1e-2, sampling='log'))))
    model.add(Dropout(hp.Float('input_dropout',
              min_value=0.0, max_value=0.5, step=0.1)))

    # Hidden layers
    for i in range(hp.Int('num_hidden_layers', 1, 4)):
        model.add(LSTM(units=hp.Int(f'hidden_units_{i}', min_value=30, max_value=200, step=10),
                       return_sequences=True,
                       kernel_regularizer=regularizers.l2(hp.Float(f'l2_reg_hidden_{i}', 1e-4, 1e-2, sampling='log'))))
        model.add(Dropout(
            hp.Float(f'hidden_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))

    # Output layer
    model.add(LSTM(units=hp.Int('output_units', min_value=30, max_value=200, step=10),
                   kernel_regularizer=regularizers.l2(hp.Float('l2_reg_output', 1e-4, 1e-2, sampling='log'))))
    model.add(Dropout(hp.Float('output_dropout',
              min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model


class LSTMModel:
    def __init__(self, df, close_idx, symbol, path):
        self.df = df
        self.close_idx = close_idx
        self.symbol = symbol
        self.path = path
        self.scaler = None
        self.scaled_data = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.train_ratio = 0.8
        # Set the num_features attribute here
        self.num_features = df.shape[1]

    def preprocess(self, seq_len):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        column_name = 'symbol'
        if column_name in self.df.columns:
            # Drop the column
            self.df = self.df.drop(column_name, axis=1)
            print("Column dropped.")
        else:
            print("Column not found.")
        self.scaled_data = self.scaler.fit_transform(self.df)
        num_training_samples = int(len(self.scaled_data) * self.train_ratio)

        self.x_train = [self.scaled_data[i - seq_len:i, :]
                        for i in range(seq_len, num_training_samples)]
        self.y_train = [self.scaled_data[i, self.close_idx]
                        for i in range(seq_len, num_training_samples)]

        self.x_train, self.y_train = np.array(
            self.x_train), np.array(self.y_train)

        self.x_test = [self.scaled_data[i - seq_len:i, :]
                       for i in range(num_training_samples, len(self.scaled_data))]
        self.y_test = [self.scaled_data[i, self.close_idx]
                       for i in range(num_training_samples, len(self.scaled_data))]

        self.x_test, self.y_test = np.array(self.x_test), np.array(self.y_test)

        self.x_train = np.reshape(
            self.x_train, (self.x_train.shape[0], self.x_train.shape[1], self.num_features))

        self.x_test = np.reshape(
            self.x_test, (self.x_test.shape[0], self.x_test.shape[1], self.num_features))

        return self.x_train, self.y_train, self.x_test, self.y_test

    def search_best_hyperparameters(self, x_train, y_train, epochs, max_trials):
        def lstm_hypermodel(hp): return build_lstm_model(hp, self.num_features)

        tuner = RandomSearch(lstm_hypermodel, objective='val_loss', max_trials=max_trials,
                             seed=42, executions_per_trial=2, directory=self.path)
        tuner.search(x_train, y_train, epochs=epochs,
                     validation_split=0.2, verbose=2)

        best_hp = tuner.get_best_hyperparameters()[0]
        # No need to save the sequence_length value in the best_hp object
        # best_hp.values['sequence_length'] = sequence_length
        return best_hp

    def fit_and_save_model(self, best_hp, x_train, y_train, epochs, model_path, tensorboard_callback):
        self.model = self.build_model(best_hp)

        # Get the best batch size from the search
        best_batch_size = best_hp.get('batch_size', 32)

        self.model.fit(x_train, y_train, epochs=epochs,
                       batch_size=best_batch_size, validation_split=0.2, verbose=2,
                       callbacks=[tensorboard_callback])  # Add the callback here

        self.model.save(model_path)

    def train_evaluate_and_predict(self, csv_cleaner, model_path, max_trials=20, epochs=5):
        # Temporarily preprocess the data with an arbitrary sequence length of 10.
        temp_seq_len = 10
        temp_x_train, temp_y_train, temp_x_test, temp_y_test = self.preprocess(
            temp_seq_len)

        best_hp = self.search_best_hyperparameters(
            temp_x_train, temp_y_train, epochs, max_trials)

        # Get the best sequence_length from the search
        best_sequence_length = best_hp.get('sequence_length', 10)

        # If the best sequence length is different from the temporary one, preprocess the data again
        if best_sequence_length != temp_seq_len:
            x_train, y_train, x_test, y_test = self.preprocess(
                best_sequence_length)
        else:
            x_train, y_train, x_test, y_test = temp_x_train, temp_y_train, temp_x_test, temp_y_test

        # Create a TensorBoard callback
        log_dir = os.path.join(
            "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.fit_and_save_model(best_hp, x_train, y_train,
                                epochs, model_path, tensorboard_callback)
        self.evaluate()

        prediction_days = [15, 30, 60, 90, 120]
        future_predictions = self.predict_future_close_price(
            csv_cleaner, prediction_days)
        return future_predictions

    @classmethod
    def load_model(cls, model_path, df, close_idx, symbol):
        model = keras.models.load_model(model_path)
        lstm_model_instance = cls(df, close_idx, symbol)
        lstm_model_instance.model = model
        return lstm_model_instance

    def evaluate(self):
        self.model = keras.models.load_model(
            f'{self.path}/{self.symbol}.h5')
        self.test_loss = self.model.evaluate(self.x_test, self.y_test)
        self.test_predictions = self.model.predict(self.x_test)

        dummy_array = np.zeros((self.y_test.shape[0], self.num_features))
        closing_price_col_idx = self.df.columns.get_loc('close')

        for i in range(self.test_predictions.shape[1]):
            dummy_array[:, closing_price_col_idx] = self.test_predictions[:, i]
            self.test_predictions[:, i] = self.scaler.inverse_transform(dummy_array)[
                :, closing_price_col_idx]

    def get_predictions(self):

        return self.test_predictions

    def predict_future_close_price(self, csv_cleaner, prediction_days):
        last_n_days_data = csv_cleaner.df[-self.num_time_steps:]
        last_n_days_data_numeric = last_n_days_data.select_dtypes(
            include=np.number)
        scaled_last_n_days_data = self.scaler.transform(
            last_n_days_data_numeric)
        x_input = np.array(scaled_last_n_days_data).reshape(
            1, self.num_time_steps, self.num_features)

        future_close_prices_scaled = self.model.predict(x_input)
        dummy_array = np.zeros((len(prediction_days), self.num_features))
        closing_price_col_idx = self.df.columns.get_loc('close')
        dummy_array[:, closing_price_col_idx] = future_close_prices_scaled

        future_close_prices = self.scaler.inverse_transform(
            dummy_array)[:, closing_price_col_idx]

        predictions = {}
        for days, future_close_price in zip(prediction_days, future_close_prices):
            most_recent_close_price = csv_cleaner.df.iloc[-1]['close']
            prediction = 1 if future_close_price > most_recent_close_price else 2
            predictions[days] = (prediction, future_close_price)

        return predictions
