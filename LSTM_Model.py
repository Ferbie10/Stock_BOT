import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import ssl
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class LSTMModel:
    def __init__(self, csv_file_path, train_test_split_ratio=0.8, num_time_steps=10, num_features=5, num_hidden_units=50, close_column_index=3):
        self.df = pd.read_csv(csv_file_path)
        self.train_test_split_ratio = train_test_split_ratio
        self.num_time_steps = num_time_steps
        self.num_features = num_features
        self.num_hidden_units = num_hidden_units
        self.close_column_index = close_column_index

    def preprocess(self):
        # Normalize the data using MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(
            self.df[['open', 'high', 'low', 'close', 'volume']])

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
        print(self.x_test)

        self.x_train = np.reshape(
            self.x_train, (self.x_train.shape[0], self.x_train.shape[1], self.num_features))
        self.x_test = np.reshape(
            self.x_test, (self.x_test.shape[0], self.x_test.shape[1], self.num_features))

    def build_model(self):
        # Build the LSTM model
        with tf.device('/gpu:0'):
            self.model = Sequential()
            self.model.add(LSTM(units=self.num_hidden_units, return_sequences=True, input_shape=(
                self.num_time_steps, self.num_features)))
            self.model.add(
                LSTM(units=self.num_hidden_units, return_sequences=True))
            self.model.add(LSTM(units=self.num_hidden_units))
            self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, num_epochs=100, batch_size=32):
        # Train the model
        with tf.device('/gpu:0'):
            self.model.fit(self.x_train, self.y_train,
                           epochs=num_epochs, batch_size=batch_size)

    def evaluate(self):
        # Evaluate the model
        with tf.device('/gpu:0'):
            self.test_loss = self.model.evaluate(self.x_test, self.y_test)
            self.test_predictions = self.model.predict(self.x_test)
            self.test_predictions = self.scaler.inverse_transform(
                self.test_predictions)

    def get_predictions(self):
        # Return the predictions on the test data
        return self.test_predictions
