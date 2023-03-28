import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import keras
import datetime
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import L2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def build_lstm_model(hp, num_time_steps, num_features):
    model = Sequential()
    model.add(LSTM(units=hp.Int('num_hidden_units', min_value=32, max_value=128, step=16),
                   return_sequences=True,
                   input_shape=(num_time_steps, num_features),
                   kernel_regularizer=L2(hp.Float('l2', 1e-5, 1e-3, sampling='log'))))
    model.add(LSTM(units=hp.Int('num_hidden_units', min_value=32, max_value=128, step=16),
                   return_sequences=True,
                   kernel_regularizer=L2(hp.Float('l2', 1e-5, 1e-3, sampling='log'))))
    model.add(LSTM(units=hp.Int('num_hidden_units', min_value=32, max_value=128, step=16),
                   kernel_regularizer=L2(hp.Float('l2', 1e-5, 1e-3, sampling='log'))))
    model.add(Dense(units=3))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


class LSTMModel:
    def __init__(self, cleaned_df, close_column_index, symbol, today_folder, train_test_split_ratio=0.8, num_time_steps=100, num_features=None, num_hidden_units=50):
        self.df = cleaned_df
        self.train_test_split_ratio = train_test_split_ratio
        self.num_time_steps = num_time_steps
        self.num_features = num_features if num_features else len(
            self.df.columns)

        self.num_hidden_units = num_hidden_units
        self.close_column_index = close_column_index
        self.symbol = symbol
        self.today_folder = today_folder

    def preprocess(self):
        if 'symbol' in self.df.columns:
            self.df.drop(columns=['symbol'], inplace=True)
        # Normalize the data using MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.df)

        # Split the data into training and testing sets
        num_training_samples = int(
            len(self.scaled_data) * self.train_test_split_ratio)
        self.x_train = []
        self.y_train = []

        for i in range(self.num_time_steps, num_training_samples):
            self.x_train.append(self.scaled_data[i - self.num_time_steps:i, :])
            # use the close price as the label
            self.y_train.append(self.scaled_data[i, self.close_column_index])
        self.x_train, self.y_train = np.array(
            self.x_train), np.array(self.y_train)

        self.x_test = []
        self.y_test = []

        for i in range(num_training_samples, len(self.scaled_data)):
            self.x_test.append(self.scaled_data[i - self.num_time_steps:i, :])
            # use the close price as the label
            self.y_test.append(self.scaled_data[i, self.close_column_index])
        self.x_test, self.y_test = np.array(self.x_test), np.array(self.y_test)

        # Reshape the data for use with an LSTM model
        self.x_train = np.reshape(
            self.x_train, (self.x_train.shape[0], self.x_train.shape[1], self.num_features))
        self.x_test = np.reshape(
            self.x_test, (self.x_test.shape[0], self.x_test.shape[1], self.num_features))

        return self.x_train, self.y_train, self.x_test, self.y_test
#unused for the time being

    @classmethod
    def load_model(cls, model_path, cleaned_df, close_column_index, symbol, today_folder):
        model = keras.models.load_model(model_path)
        lstm_model_instance = cls(
            cleaned_df, close_column_index, symbol, today_folder)
        lstm_model_instance.model = model
        return lstm_model_instance

    def evaluate(self):
        self.model = keras.models.load_model(
            f'{self.today_folder}/{self.symbol}.h5')
        with tf.device('/gpu:0'):
            self.test_loss = self.model.evaluate(self.x_test, self.y_test)
            self.test_predictions = self.model.predict(self.x_test)

            # Create a dummy array with the same shape as the test set
            dummy_array = np.zeros((self.y_test.shape[0], self.num_features))

            closing_price_column_index = self.df.columns.get_loc('close')
            for i in range(self.test_predictions.shape[1]):
                dummy_array[:, closing_price_column_index] = self.test_predictions[:, i]

                # Apply inverse_transform on the dummy array
                self.test_predictions[:, i] = self.scaler.inverse_transform(
                    dummy_array)[:, closing_price_column_index]

    def get_predictions(self):
        # Return the predictions on the test data
        return self.test_predictions

    def predict_future_close_price(self, csv_cleaner, prediction_days):
        last_n_days_data = csv_cleaner.df[-self.num_time_steps:]
        last_n_days_data_numeric = last_n_days_data.select_dtypes(
            include=np.number)
        scaled_last_n_days_data = self.scaler.transform(
            last_n_days_data_numeric)
        x_input = np.array(scaled_last_n_days_data).reshape(
            1, self.num_time_steps, self.num_features)

        with tf.device('/gpu:0'):
            future_close_prices_scaled = self.model.predict(x_input)

            # Create a dummy array with the same shape as the input for inverse_transform
            dummy_array = np.zeros((len(prediction_days), self.num_features))
            closing_price_column_index = self.df.columns.get_loc('close')
            dummy_array[:, closing_price_column_index] = future_close_prices_scaled

            # Apply inverse_transform on the dummy array
            future_close_prices = self.scaler.inverse_transform(
                dummy_array)[:, closing_price_column_index]

            predictions = {}
            for days, future_close_price in zip(prediction_days, future_close_prices):
                most_recent_close_price = csv_cleaner.df.iloc[-1]['close']
                prediction = 1 if future_close_price > most_recent_close_price else 2
                predictions[days] = (prediction, future_close_price)

        return predictions