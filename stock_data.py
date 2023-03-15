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


class Get_SP500:
    def __init__(self, url):
        self.url = url
        self.sp500 = []

    def download(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        tables = pd.read_html(self.url)
        self.sp500 = tables[0]['Symbol'].tolist()
        return self.sp500


class Get_Stock_History:
    def __init__(self, path, sp500, start_date, end_date):
        self.path = path
        self.sp500 = sp500
        self.start_date = start_date
        self.end_date = end_date

    def compstockdata(self):

        for symbol in self.sp500:
            ticker = yf.Ticker(symbol)
            tik_history = ticker.history(
                period='1mo', interval='15m')
            filename = os.path.join(self.path, f'{symbol}.csv')
            if not os.path.exists(filename):
                tik_history.to_csv(filename)

            output_file_path = os.path.join(self.path, f'{symbol}_edited.csv')

            # Call CSV cleaner for the newly created file
            cleanCSV = dataPrep.CSVCleaner(filename, output_file_path)
            cleanCSV.clean()
            cleanCSV.transform(output_file_path)

            model = LSTM_Model.LSTMModel(output_file_path)
            model.preprocess()
            model.build_model()
            model.train()
            model.evaluate()


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
        print('no path')
        os.makedirs(today_folder)
    elif os.path.exists(today_folder):
        print(f"Path  {today_folder}")
    sp500 = ["aapl"]
    test = Get_Stock_History(today_folder, sp500, today_folder, start_date)
    test.compstockdata()


main()
