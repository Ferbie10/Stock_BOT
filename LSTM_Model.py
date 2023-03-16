import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class LSTMModel:
    def __init__(self, cleaned_df, close_column_index,train_test_split_ratio=0.8, num_time_steps=10, num_features=22, num_hidden_units=50):
        self.df = cleaned_df
        self.train_test_split_ratio = train_test_split_ratio
        self.num_time_steps = num_time_steps
        self.num_features = num_features if num_features else len(self.df.columns)
        self.num_hidden_units = num_hidden_units
        self.close_column_index = close_column_index

    def preprocess(self):
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


    def predict_tomorrow(self, last_n_days_data):
        # Scale the input data using the previously fitted scaler
        scaled_data = self.scaler.transform(last_n_days_data)

        # Reshape the input data for the LSTM model
        input_data = np.reshape(
            scaled_data, (1, scaled_data.shape[0], scaled_data.shape[1]))

        # Make the prediction using the trained model
        prediction = self.model.predict(input_data)

        # Inverse transform the prediction to get the actual price
        actual_prediction = self.scaler.inverse_transform(prediction)
        
        return actual_prediction[0][0]

