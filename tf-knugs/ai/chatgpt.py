import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import ssl
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import dataPrep
import LSTM_Model

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
                period='10y', start=self.start_date, end=self.end_date, interval='15m')
            filename = os.path.join(self.path, f'{symbol}.csv')
            if not os.path.exists(filename):
                tik_history.to_csv(filename)
            
            output_file_path = os.path.join(self.path,f'{symbol}_edited.csv')
            


            # Call CSV cleaner for the newly created file
            cleanCSV = dataPrep.CSVCleaner(filename,output_file_path)
            cleanCSV.clean()
            cleanCSV.transform(output_file_path)
            
            model = LSTM_Model.LSTMModel(output_file_path)
            model.preprocess()
            model.build_model()
            model.train()
            model.evaluate()

class Date:
    def __init__(self, assets_file_path):
        self.assets_df = pd.read_csv(assets_file_path)

    def get_start_date(self):
        start_date = self.assets_df.iloc[0, 0]
        start_datetime_object = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = start_datetime_object + timedelta(days=1)
        return start_datetime_object, end_date



def main():
    parent = 'C:\\Users\\jjudi\\Documents\\AI_test'
    assets_file_path = os.path.join(parent, 'assets.csv')
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # Initialize Date object to get start date
    get_dates = Date(assets_file_path)
    start_datetime_object, end_date = get_dates.get_start_date()
    path = os.path.join(parent, start_datetime_object.strftime('%Y-%m-%d'))
    if not os.path.exists(path):
        os.makedirs(path)
            

    # Download and clean stock data
    
    #Complist = Get_SP500(url)
    #sp500List = Complist.download()
    sp500List = ['aapl']
    sp500_list = Get_Stock_History(path, sp500List, start_datetime_object, end_date)
    sp500_list.compstockdata()

    

    # Train model and make predictions
    
    


    

if __name__ == '__main__':
    main()
