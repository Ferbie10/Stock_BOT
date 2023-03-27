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
    def __init__(self, path, stocks, start_date):
        self.path = path
        self.stocks = stocks
        self.start_date = start_date

    def normalize_stock_data(self, df):
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=np.number)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(numeric_df.values)
        normalized_df = pd.DataFrame(
            scaled_data, columns=numeric_df.columns, index=numeric_df.index)
        return normalized_df, scaler

    def download_stock_history(self, symbol, year, interval):
        ticker = yf.Ticker(symbol)
        tik_history = ticker.history(
            period=f'{year}y', interval=f'{interval}d')
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

    def download_and_preprocess_data(self, symbol, years, interval):
        filename = self.download_stock_history(symbol, years, interval)
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

        # Get the predictions
        prediction_days = [1, 5, 20]
        predictions = lstm_model.predict_future_close_price(
            csv_cleaner, prediction_days)

        # Transform predictions dictionary to a list of tuples
        predictions_list = [(pd.Timestamp.now(), days, pred[0], pred[1])
                            for days, pred in predictions.items()]

        # Create a new DataFrame for the predictions
        predictions_df = pd.DataFrame(predictions_list, columns=[
                                      'timestamp', 'prediction_days', 'prediction', 'future_close_price'])

        # Add the symbol to the predictions DataFrame
        predictions_df['symbol'] = symbol

        # Rearrange the columns in the desired order
        predictions_df = predictions_df[[
            'symbol', 'timestamp', 'prediction_days', 'prediction', 'future_close_price']]

        # Save the predictions DataFrame to a new CSV file
        predictions_df.to_csv(
            f'{today_folder}/{symbol}_Predictions_Separate.csv', index=False)
