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
        # Set the num_features attribute here
        self.num_features = df.shape[1]

    def preprocess(self, csv_cleaner, testing_seq_length):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        column_name = 'symbol'
        if column_name in self.df.columns:
            # Drop the column
            self.df = self.df.drop(column_name, axis=1)
            print("Column dropped.")
        else:
            print("Column not found.")
        self.scaled_data = self.scaler.fit_transform(self.df)
        num_training_samples = int(len(self.scaled_data) * 0.8)

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

    def search_best_hyperparameters(self):
        # Define the hyperparameter search space
        hp = kt.HyperParameters()
        hp.Int('sequence_length', min_value=10, max_value=100, step=10)
        hp.Int('num_lstm_layers', min_value=1, max_value=3, step=1)
        hp.Int('lstm_units', min_value=32, max_value=256, step=32)
        hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        hp.Int('batch_size', min_value=16, max_value=128, step=16)
        hp.Choice('optimizer', values=['adam', 'rmsprop'])

        # Create the tuner
        tuner = kt.RandomSearch(
            self.build_lstm_model,
            objective=kt.Objective("val_accuracy", direction="max"),
            hyperparameters=hp,
            max_trials=10,
            seed=42,
            overwrite=True
        )

        # Start the search
        tuner.search(self.x_train, self.y_train, epochs=self.epochs, validation_data=(self.x_val, self.y_val), verbose=1)

        # Get the best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        print(f"Best hyperparameters found: {best_hp}")

        return best_hp

    def search_best_hyperparameters(self, x_train, y_train, epochs, max_trials):
        if x_train is None or y_train is None:
            raise ValueError(
                "Invalid data: x_train and y_train must not be None")

        def lstm_hypermodel(hp): return build_lstm_model(hp, self.num_features)

        tuner = RandomSearch(lstm_hypermodel, objective='val_loss', max_trials=max_trials,
                             seed=42, executions_per_trial=2, directory=self.path)
        tuner.search(x_train, y_train, epochs=epochs,
                     validation_split=0.2, verbose=2)

        best_hp = tuner.get_best_hyperparameters()[0]
        return best_hp

    def fit_and_save_model(self, best_hp, x_train, y_train, epochs, model_path, tensorboard_callback):
        self.model = build_lstm_model(best_hp, self.num_features)

        # Get the best batch size from the search
        best_batch_size = best_hp.get('batch_size', 32)

        self.model.fit(x_train, y_train, epochs=epochs,
                       batch_size=best_batch_size, validation_split=0.2, verbose=2,
                       callbacks=[tensorboard_callback])  # Add the callback here

        self.model.save(model_path)

    def train_evaluate_and_predict(self, csv_cleaner, model_path):
        # Preprocess data with the default sequence length
        self.preprocess(csv_cleaner)

        # Search for the best hyperparameters
        best_hp = self.search_best_hyperparameters()

        # Build the LSTM model with the best hyperparameters and the best sequence length
        self.build_lstm_model(best_hp, csv_cleaner, best_hp['sequence_length'])

        # Train the model
        history = self.model.fit(self.x_train, self.y_train, validation_data=(
            self.x_val, self.y_val), epochs=self.epochs, batch_size=best_hp['batch_size'], verbose=1)

        # Evaluate the model
        evaluation = self.model.evaluate(
            self.x_test, self.y_test, batch_size=best_hp['batch_size'], verbose=1)
        print(f"Test loss: {evaluation[0]}, Test accuracy: {evaluation[1]}")

        # Save the model
        self.model.save(model_path)

        # Make predictions
        predictions = self.model.predict(self.x_test)

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
