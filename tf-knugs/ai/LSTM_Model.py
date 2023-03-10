import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from keras.callbacks import TensorBoard

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LSTMModel:
    def __init__(self, file_path, target_column=8, sequence_length=100, train_ratio=0.8, stride=10):
        self.file_path = file_path
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.stride = stride

        # Load data from csv file
        self.data = pd.read_csv(file_path)
        self.data = self.data.dropna()

        # Split data into training and testing sets
        self.split_index = int(len(self.data) * train_ratio)
        self.train_data = self.data[:self.split_index]
        self.test_data = self.data[self.split_index:]

        # Create a MinMaxScaler object and fit it on the training data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_data = self.scaler.fit_transform(self.train_data)

    def create_dataset(self, data):
        """
        Create input sequences for the LSTM model.

        :param data: 2D array of input data.
        :param sequence_length: number of time steps to include in each input sequence.
        :param stride: spacing between consecutive input sequences.
        :return: tuple containing input sequences and target values.
        """
        data = self.data.values  # Convert data to a numpy array
        dataX, dataY = [], []
        print(f"NUMBER OF SAMPLEs{len(self.data)}")
        n_samples = len(self.data) - self.sequence_length - 1
        print(f"NUMBER OF SAMPLEs{n_samples}")
        for i in range(0, n_samples, self.stride):
            # Extract the input sequence and target value
            seq = data[i:i+self.sequence_length]
            target = data[i+self.sequence_length][0]

            # Append the input sequence and target value to the output arrays
            dataX.append(seq.flatten())
            dataY.append(target)

        dataX = np.array(dataX)
        dataY = np.array(dataY)

        return dataX, dataY
    def train(self, epochs, batch_size=10):
        with tf.device('/gpu:0'):
            # Create training dataset
            trainX, trainY = self.create_dataset(
                self.train_data)

            # Reshape input data to be 3-dimensional in the format [samples, time steps, features]
            print(trainX.shape)

            trainX = np.reshape(
                trainX, (trainX.shape[0], self.sequence_length, self.data.shape[1]))

            # Build LSTM model
            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(
                self.sequence_length, self.train_data.shape[1])))
            self.model.add(Dense(1))
            self.model.compile(loss='mean_squared_error', optimizer='adam')

            # Define directory for TensorBoard logs
            log_dir = "/home/ferbie10/git/Stock_BOT-1/2005-01-20/logs/fit/"

            # Create TensorBoard callback
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Fit the model with TensorBoard callback
            self.model.fit(trainX, trainY, epochs=epochs,
                           batch_size=batch_size, verbose=2, callbacks=[tensorboard_callback])
            self.model.save('model.h5')

    def inverse_transform(self, data):
        # Check that data is a numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Check that data has two dimensions
        if len(data.shape) != 2:
            raise ValueError("Data must be a 2D array.")

        # Get the number of features
        n_features = self.scaler.data_range_.shape[0]

        # If data has only one feature, add a second dimension
        if data.shape[1] == 1:
            data = np.reshape(data, (data.shape[0], 1))

        # Rescale the data using the scaler's data range and min values
        rescaled_data = data * self.scaler.data_range_ + self.scaler.data_min_

        # Return the rescaled data
        return rescaled_data

    def evaluate(self):
        with tf.device('/gpu:0'):
            # Load the saved model
            path = '/home/ferbie10/git/Stock_BOT-1/model.h5'
            model = load_model(path)

            # Create testing dataset
            testX, testY = self.create_dataset(model)

            # Reshape input data to be 3-dimensional in the format [samples, time steps, features]
            n_samples = testX.shape[0]
            print(n_samples)
            
            n_features = self.train_data.shape[1]
            print(n_features)
            expected_shape = (n_samples, self.sequence_length, n_features)
            print(testX.shape)
            print(expected_shape)
            testX = np.reshape(testX, expected_shape)

            # Get predictions on the testing dataset using the loaded model
            testPredict = model.predict(testX)

            # Inverse transform the data to its original scale
            testPredict = self.inverse_transform(testPredict)
            testY = self.inverse_transform(testY.reshape(-1, 1))
            testPredict = testPredict.reshape(-1, n_features)
            # Calculate root mean squared error
            rmse = np.sqrt(mean_squared_error(testY, testPredict))
            print('Test RMSE: %.2f' % rmse)
