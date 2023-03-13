import dataPrep
import stock_data
import LSTM_Model
import datetime
import os


def main():
    parent = '/home/ferbie10/git/Stock_BOT-1'
    
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # Initialize Date object to get start date
    today = datetime.date.today()
    years_past = 5
    start_year = today.year - years_past
    start_date = datetime.date(start_year, today.month, today.day)

    path = os.path.join(parent, start_date.strftime('%Y-%m-%d'))
    if not os.path.exists(path):
        os.makedirs(path)
            

    # Download and clean stock data
    
    #Complist = Get_SP500(url)
    #sp500List = Complist.download()
    sp500List = ['aapl']
    sp500_list = Get_Stock_History(path, sp500List, start_datetime_object, end_date)
    sp500_list.compstockdata()

    

    # Train model and make predictions