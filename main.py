import LSTM_Model
import datetime
import os
import keras
import stock_data
import pandas as pd
import dataPrep
import pandas_datareader as pdr


def date(year, parent):
    today = datetime.date.today()
    years_past = int(year)
    start_year = today.year - years_past
    start_date = datetime.date(start_year, today.month, today.day)
    today_folder = os.path.join(parent, start_date.strftime('%Y-%m-%d'))
    if not os.path.exists(today_folder):
        os.makedirs(today_folder)
    elif os.path.exists(today_folder):
        pass
    return today_folder


def stock_folder(symbol, path):

    filename = os.path.join(path, f'{symbol}')
    if not os.path.exists(filename):
        os.mkdir(filename)
    else:
        pass
    return filename


def main():
    parent = '/root/home/git/'

    loop = 0
    while loop == 0:
        # user_options = input("Enter:\n1 for a new model\n2 to load in a CSV\n3 to load processed CSV\n4 to load in a Model\n0 to end the program\n")
        user_options = '1'
        if user_options == '1':
            # indivdual_or_list = int(input("Enter 1 for individual stock or 2 for stock index:  "))
            indivdual_or_list = 1
            if indivdual_or_list == 1:
                # stock_list = input("Please enter the stock Symbol:  ")
                stock_list = 'aapl'
                # years = input("Enter the number of years: ")
                years = 1
                interval = 1
                # interval = input("Please enter the intervel: ")
                today_folder = date(years, parent)
                stockfolder = stock_folder(stock_list, today_folder)
                single_stock = stock_data.Get_Stock_History(
                    stockfolder, stock_list)
                normalized_df, close_column_index, csv_cleaner = single_stock.download_and_preprocess_data(
                    years, interval)
                single_stock.train_evaluate_and_predict(
                    normalized_df, close_column_index, csv_cleaner)
                loop = 1

            else:
                index_url = input(
                    "Please enter the Ticker symbol of the ETF List:   ")
                years = input("Enter the number of years: ")
                interval = input("Please enter the intervel: ")
                tickers = pdr.get_data_yahoo(index_url).index.tolist()
                today_folder = date(years)
                for symbol in tickers:
                    stockfolder = stock_folder(symbol, today_folder)
                    single_stock = stock_data.Get_Stock_History(
                        stockfolder, symbol)
                    normalized_df, close_column_index, csv_cleaner = single_stock.download_and_preprocess_data(
                        years, interval)
                    single_stock.train_evaluate_and_predict(
                        normalized_df, close_column_index, symbol, csv_cleaner, stockfolder)
            os.system('clear')

        elif user_options == '2':
            csv_path = input("Please enter the path of the CSV file: ")
            stocks = stock_data.Get_Stock_History(
                today_folder, None, start_date, today)
            stocks.load_and_preprocess_csv(csv_path)
            os.system('clear')
        elif user_options == '3':
            # processed_data_path = input("Please enter the path of the processed data CSV file: ")
            processed_data_path = '/root/home/git/2008-03-23/aapl_edited.csv'
            ticker = processed_data_path.split(
                "/")[-1].replace("_edited.csv", "")

            stocks = stock_data.Get_Stock_History(
                today_folder, ticker, start_date)
            processed_df, close_column_index, symbol, csv_cleaner = stocks.load_processed_data(
                processed_data_path)

            stocks.train_evaluate_and_predict(
                processed_df, close_column_index, symbol, csv_cleaner, today_folder)
            os.system('clear')
        elif user_options == '4':
            model_path = input(
                "Please enter the path of the saved model file: ")
            lstm_model = LSTM_Model.LSTMModel.load_model(
                model_path, cleaned_df, close_column_index, symbol, today_folder)
            lstm_model.evaluate()

            os.system('clear')
        else:
            loop = 1


main()
