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

    def normalize_data(self, df):
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
        normalized_df, scaler = self.normalize_data(csv_cleaner.df)

        return normalized_df, csv_cleaner.df.columns.get_loc('close'), csv_cleaner

    def train_and_evaluate_lstm_model(self, normalized_df, close_column_index, symbol):
        lstm_model = LSTM_Model.LSTMModel(
            cleaned_df=normalized_df, close_column_index=close_column_index, symbol=symbol)
        lstm_model.preprocess()
        lstm_model.build_model()
        lstm_model.train()
        lstm_model.evaluate()

        return lstm_model

    def compstockdata(self):
        for symbol in self.sp500:
            filename = self.download_stock_history(symbol)
            normalized_df, close_column_index, csv_cleaner = self.preprocess_stock_data(
                filename, symbol)
            lstm_model = self.train_and_evaluate_lstm_model(
                normalized_df, close_column_index, symbol)

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


def main():
    parent = '/root/home/git'
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # Initialize Date object to get start date
    today = datetime.date.today()
    years_past = 15
    start_year = today.year - years_past
    start_date = datetime.date(start_year, today.month, today.day)

    today_folder = os.path.join(parent, start_date.strftime('%Y-%m-%d'))
    print(today_folder)
    if not os.path.exists(today_folder):

        os.makedirs(today_folder)
    elif os.path.exists(today_folder):
        print(f"Path  {today_folder}")

    sp500 = ["aapl"]
    test = Get_Stock_History(today_folder, sp500, start_date)
    test.compstockdata()


main()
