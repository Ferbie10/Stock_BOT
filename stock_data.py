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

        return normalized_df, csv_cleaner.df.columns.get_loc('close')

    def train_and_evaluate_lstm_model(self, normalized_df, close_column_index):
        lstm_model = LSTM_Model.LSTMModel(
            cleaned_df=normalized_df, close_column_index=close_column_index)
        lstm_model.preprocess()
        lstm_model.build_model()
        lstm_model.train()
        lstm_model.evaluate()

        return lstm_model

    def predict_tomorrow_close_price(self, lstm_model, csv_cleaner, num_time_steps=10):
        last_n_days_data = csv_cleaner.df[-num_time_steps:]
        tomorrow_close_price = lstm_model.predict_tomorrow(last_n_days_data)
        return tomorrow_close_price

    def compstockdata(self):
        for symbol in self.sp500:
            filename = self.download_stock_history(symbol)
            normalized_df, close_column_index = self.preprocess_stock_data(filename, symbol)
            lstm_model = self.train_and_evaluate_lstm_model(normalized_df, close_column_index)
            tomorrow_close_price = self.predict_tomorrow_close_price(lstm_model, csv_cleaner)
            print("Predicted closing price for tomorrow:", tomorrow_close_price)



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