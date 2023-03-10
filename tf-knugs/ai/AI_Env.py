import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import ssl
import os



class Ticker:
    def __init__(self):
        self.url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    def download(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        tables = pd.read_html(self.url)
        self.sp500 = tables[0]
        return self.sp500

class Date:
    def __init__(self, assets_file_path):
        self.assets_df = pd.read_csv(assets_file_path)

    def get_start_date(self):
        start_date = self.assets_df.iloc[0, 0]
        start_datetime_object = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = start_datetime_object + timedelta(days=1)
        return start_datetime_object, end_date


def sp500_data(sp500, start_date, start, end_date):
    data = []
    directory = start
    parent = 'C:\\Users\\jjudi\\Documents\\AI_test\\'
    path = os.path.join(parent, directory)
    if not os.path.exists(path):
        os.mkdir(path)

    for symbol in sp500:
        ticker = yf.Ticker(symbol)
        tik_history = ticker.history(
            period='1mo', start=start_date, end=end_date, interval='1wk')
        filename = os.path.join(path, f'{symbol}.csv')
        tik_history.to_csv(filename)
        data.append(symbol)
    return data


def main():
    parent = 'C:/Users/jjudi/Documents/AI_test'
    assets_file_path = os.path.join(parent, 'assets.csv')
    
    get_dates = Date(assets_file_path)
    start_datetime_object, end_date = get_dates.get_start_date()
    init_cash = assets_df["Cash"]
    #sp500 = ticker()
    sp500 = ['aapl']
    tickers = sp500_data(sp500, start_datetime_object, start_date, end_date)
    print(tickers)


main()
