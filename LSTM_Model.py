import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class LSTMModel:
    def __init__(self, dataframe, target_col, test_size=0.2, lookback=60, epochs=50, batch_size=32):
        self.data = dataframe
        self.target_col = target_col

        self.test_size = test_size
        self.lookback = int(lookback)
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()
        self.model = self.build_model()

    def prepare_data(self):
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)

        # Prepare the input and output sequences for the LSTM model
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i])
            # Assuming 'close' price is the first column
            y.append(scaled_data[i, self.target_col])
        X, y = np.array(X), np.array(y)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False)

        return X_train, X_test, y_train, y_test

    def build_model(self):
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), kernel_regularizer=l2(
            0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False, kernel_regularizer=l2(
            0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile the model with an optimizer and loss function
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def train(self):
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = ModelCheckpoint(
            filepath='best_model.h5', monitor='val_loss', save_best_only=True)

        # Train the model
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                       epochs=self.epochs, batch_size=self.batch_size, callbacks=[early_stopping, model_checkpoint])

    def predict(self, X, days=5):
        predictions = []

        for _ in range(days):
            prediction = self.model.predict(X)
            predictions.append(prediction[0])

            # Append the predicted value to the input sequence
            new_row = np.append(X[0][1:], prediction, axis=0)

            # Reshape the input sequence to match the required input shape
            X = np.reshape(new_row, (1, new_row.shape[0], new_row.shape[1]))

        return predictions
