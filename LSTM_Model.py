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
                   input_shape=(hp.Int('sequence_length', min_value=10, max_value=100, step=10), num_features),
                   kernel_regularizer=regularizers.l2(hp.Float('l2_reg_input', 1e-4, 1e-2, sampling='log'))))
    model.add(Dropout(hp.Float('input_dropout', min_value=0.0, max_value=0.5, step=0.1)))

    # Hidden layers
    for i in range(hp.Int('num_hidden_layers', 1, 4)):
        model.add(LSTM(units=hp.Int(f'hidden_units_{i}', min_value=30, max_value=200, step=10),
                       return_sequences=True,
                       kernel_regularizer=regularizers.l2(hp.Float(f'l2_reg_hidden_{i}', 1e-4, 1e-2, sampling='log'))))
        model.add(Dropout(hp.Float(f'hidden_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))

    # Output layer
    model.add(LSTM(units=hp.Int('output_units', min_value=30, max_value=200, step=10),
                   kernel_regularizer=regularizers.l2(hp.Float('l2_reg_output', 1e-4, 1e-2, sampling='log'))))
    model.add(Dropout(hp.Float('output_dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='mean_squared_error')

    return model


class LSTMModel:
    def __init__(self, df, close_idx, symbol, today_folder, split_ratio=0.8, num_features=None, hidden_units=50):
        self.df = df
        self.split_ratio = split_ratio
        self.num_features = num_features if num_features else len(self.df.columns)
        self.hidden_units = hidden_units
        self.close_idx = close_idx
        self.symbol = symbol
        self.today_folder = today_folder

    def preprocess(self, seq_len):
        self.df.drop(columns=['symbol'], inplace=True, errors='ignore')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.df)
        num_training_samples = int(len(self.scaled_data) * self.split_ratio)
        
        self.x_train = [self.scaled_data[i - seq_len:i, :] for i in range(seq_len, num_training_samples)]
        self.y_train = [self.scaled_data[i, self.close_idx] for i in range(seq_len, num_training_samples)]
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

        self.x_test = [self.scaled_data[i - seq_len:i, :] for i in range(num_training_samples, len(self.scaled_data))]
        self.y_test = [self.scaled_data[i, self.close_idx] for i in range(num_training_samples, len(self.scaled_data))]
        self.x_test, self.y_test = np.array(self.x_test), np.array(self.y_test)

        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], self.num_features))
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], self.num_features))

        return self.x_train, self.y_train, self.x_test, self.y_test
    


    def train_evaluate_and_predict(self, csv_cleaner, model_path, seq_len=60, max_trials=20, epochs=100):
        x_train, y_train, x_test, y_test = self.preprocess(seq_len)
        lstm_hypermodel = lambda hp: build_lstm_model(hp, self.num_features)

        tuner = RandomSearch(lstm_hypermodel, objective='val_loss', max_trials=max_trials, seed=42, executions_per_trial=2, directory=self.today_folder)
        tuner.search(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=2)

        best_hp = tuner.get_best_hyperparameters()[0]
        self.model = tuner.hypermodel.build(best_hp)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=2)

        self.model.save(model_path)
        self.evaluate()

        prediction_days = [1, 2, 3, 4, 5]
        future_predictions = self.predict_future_close_price(csv_cleaner, prediction_days)

        return future_predictions

    @classmethod
    def load_model(cls, model_path, df, close_idx, symbol ):
        model = keras.models.load_model(model_path)
        lstm_model_instance = cls(df, close_idx, symbol)
        lstm_model_instance.model = model
        return lstm_model_instance

    def evaluate(self):
        self.model = keras.models.load_model(f'{self.today_folder}/{self.symbol}.h5')
        self.test_loss = self.model.evaluate(self.x_test, self.y_test)
        self.test_predictions = self.model.predict(self.x_test)

        dummy_array = np.zeros((self.y_test.shape[0], self.num_features))
        closing_price_col_idx = self.df.columns.get_loc('close')

        for i in range(self.test_predictions.shape[1]):
            dummy_array[:, closing_price_col_idx] = self.test_predictions[:, i]
            self.test_predictions[:, i] = self.scaler.inverse_transform(dummy_array)[:, closing_price_col_idx]

    def get_predictions(self):

        return self.test_predictions

    def predict_future_close_price(self, csv_cleaner, prediction_days):
        last_n_days_data = csv_cleaner.df[-self.num_time_steps:]
        last_n_days_data_numeric = last_n_days_data.select_dtypes(include=np.number)
        scaled_last_n_days_data = self.scaler.transform(last_n_days_data_numeric)
        x_input = np.array(scaled_last_n_days_data).reshape(1, self.num_time_steps, self.num_features)

        future_close_prices_scaled = self.model.predict(x_input)
        dummy_array = np.zeros((len(prediction_days), self.num_features))
        closing_price_col_idx = self.df.columns.get_loc('close')
        dummy_array[:, closing_price_col_idx] = future_close_prices_scaled

        future_close_prices = self.scaler.inverse_transform(dummy_array)[:, closing_price_col_idx]

        predictions = {}
        for days, future_close_price in zip(prediction_days, future_close_prices):
            most_recent_close_price = csv_cleaner.df.iloc[-1]['close']
            prediction = 1 if future_close_price > most_recent_close_price else 2
            predictions[days] = (prediction, future_close_price)

        return predictions

       
