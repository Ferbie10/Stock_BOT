import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
import ssl
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import dataPrep
import LSTM_Model
import datetime


class Get_Stock_History:
    def __init__(self, path, sp500, start_date):
        self.path = path
        self.sp500 = sp500
        self.start_date = start_date

    def normalize_stock_data(self, df):
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=np.number)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(numeric_df.values)
        normalized_df = pd.DataFrame(
            scaled_data, columns=numeric_df.columns, index=numeric_df.index)
        return normalized_df, scaler

    def download_stock_history(self, symbol):
        ticker = yf.Ticker(symbol)
        tik_history = ticker.history(period='15y', interval='1d')
        filename = os.path.join(self.path, f'{symbol}.csv')
        if not os.path.exists(filename):
            tik_history.to_csv(filename)
        return filename

    def preprocess_stock_data(self, filename, symbol):
        output_file_path = os.path.join(self.path, f'{symbol}_edited.csv')

        # Call CSV cleaner for the newly created file
        csv_cleaner = dataPrep.CSVCleaner(filename, output_file_path, symbol)
        csv_cleaner.clean()
        csv_cleaner.transform(output_file_path)
        normalized_df, scaler = self.normalize_stock_data(csv_cleaner.df)

        return normalized_df, csv_cleaner.df.columns.get_loc('close'), csv_cleaner

    def download_and_preprocess_data(self, symbol):
        filename = self.download_stock_history(symbol)
        normalized_df, close_column_index, csv_cleaner = self.preprocess_stock_data(
            filename, symbol)
        return normalized_df, close_column_index, csv_cleaner

    def process_existing_data(self, filename, symbol):
        normalized_df, close_column_index, csv_cleaner = self.preprocess_stock_data(
            filename, symbol)
        return normalized_df, close_column_index, csv_cleaner

    def load_processed_data(self, filepath):
        processed_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        symbol = os.path.basename(filepath).split('_')[0]
        close_column_index = processed_df.columns.get_loc('close')
        
        # Create a CSVCleaner object with the preprocessed data
        csv_cleaner = dataPrep.CSVCleaner(filepath, filepath, symbol)
        csv_cleaner.df = processed_df

        return processed_df, close_column_index, symbol, csv_cleaner

    def train_evaluate_and_predict(self, normalized_df, close_column_index, symbol, csv_cleaner, today_folder, model_filepath=None):
        lstm_model = LSTM_Model.LSTMModel(
            cleaned_df=normalized_df, close_column_index=close_column_index, symbol=symbol, today_folder=today_folder)

        if model_filepath:
            # Load the saved model
            lstm_model.load_model(model_filepath)
        else:
            # Build and train a new model
            lstm_model.preprocess()
            lstm_model.build_model()
            lstm_model.train()

        # Evaluate the model
        lstm_model.evaluate()

        # Use the predict_future_close_price function from the LSTMModel class
        prediction_1_day, future_close_price_1_day = lstm_model.predict_future_close_price(
            csv_cleaner, 1)
        prediction_5_day, future_close_price_5_day = lstm_model.predict_future_close_price(
            csv_cleaner, 5)
        prediction_20_day, future_close_price_20_day = lstm_model.predict_future_close_price(
            csv_cleaner, 20)

        # Add the prediction results to the DataFrame
        csv_cleaner.df.loc[pd.Timestamp.now(
        ), '1 day predict'] = prediction_1_day
        csv_cleaner.df.loc[pd.Timestamp.now(
        ), '5 day predict'] = prediction_5_day
        csv_cleaner.df.loc[pd.Timestamp.now(
        ), '20 day predict'] = prediction_20_day

        data_with_predictions = pd.read_csv(
            f'{symbol}_Predictions.csv', index_col=0, parse_dates=True)
        return data_with_predictions
