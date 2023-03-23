import dataPrep
import stock_data
import LSTM_Model
import datetime
import os


def main(*args):
    parent = '/home/ferbie10/git/Stock_BOT-1'
    
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # Initialize Date object to get start date
    today = datetime.date.today()
    years_past = 5
    start_year = today.year - years_past
    start_date = datetime.date(start_year, today.month, today.day)

    today_folder = os.path.join(parent, start_date.strftime('%Y-%m-%d'))
    if not os.path.exists(today_folder):
        os.makedirs(today_folder)
    loop = 0
    while (loop ==0):
        user_options = input("Please enter 1 to get stock data\n Please enter 2, to clean data\n \n3 Enter Enter 3 to model data")
        if user_options ==1:
            indivdual_or_list = input("Enter 1 for indivdual stock or 2 for stock index:  ")
            if indivdual_or_list ==1:
                stock_list= input("Please enter the stock Symbol:  ")


            else:
                index_url = input("Please enter the url of the Index List:   ")
                Complist = Get_SP500(url)
                stock_list = Complist.download()
            stocks = Get_Stock_History(today_folder, stock_list, start_date, today)
            stocks.compstockdata()
        elif user_options ==2:
            pass
        elif user_options ==3:
            pass
        else:
            loop=1

    

