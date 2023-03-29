import datetime
import os

import keras
import pandas as pd
import yfinance as yf
from stocksymbol import StockSymbol

import dataPrep
import LSTM_Model
import MacroFact
import stock_data
from useful_functions import *

api_key = '496f822c-c430-433d-960a-12ef11cdd5dc'
ss = StockSymbol(api_key)


def main():
    parent = '/root/home/git/Stocks'

    loop = 0
    while loop == 0:

        # user_options = input("Enter:\n1 for a new model\n2 to load in a CSV\n3 to load processed CSV\n4 to load in a Model\n0 to end the program\n")
        user_options = '0'
        if user_options == '0':
            get_or_clean = '0'
            if get_or_clean == '0':
                csv_file_path = '/root/home/git/Stocks/Fed/'
                date, symbol, filename, desired_path = split_string(
                    csv_file_path)
                start_date = '1913-01-01'
                path_of_csv = indicators(start_date, csv_file_path)
                fed_data = dataPrep.CSVCleaner(
                    path_of_csv, csv_file_path, symbol='Fed')
                output_file_path = output_path(desired_path, symbol)
                fed_data.clean_fed(output_file_path)
            else:
                pass

        elif user_options == '1':
            # indivdual_or_list = int(input("Enter 1 for individual stock or 2 for stock index:  "))
            indivdual_or_list = 1
            if indivdual_or_list == 1:

                # stock_list = input("Please enter the stock Symbol:  ")
                symbol = 'aapl'
                # years = input("Enter the number of years: ")
                years = 5
                interval = 1
                # interval = input("Please enter the intervel: ")
                today_folder, start_date = get_path_date(years, parent)
                stockfolder = stock_folder(symbol, today_folder)
                single_stock = stock_data.Get_Stock_History(
                    stockfolder, symbol, start_date)
                normalized_df, close_column_index, csv_cleaner = single_stock.download_and_preprocess_data(
                    years, interval)

                lstm_model = LSTM_Model.LSTMModel(
                    normalized_df, close_column_index, symbol, today_folder)
                model_path = model_save_path(stockfolder, symbol)
                lstm_model.train_evaluate_and_predict(csv_cleaner, model_path)

                loop = 1

            else:
                # index_url = input("Please enter the Ticker symbol of the ETF List:   ")
                index_url = 'SPX'
                # years = input("Enter the number of years: ")
                # interval = input("Please enter the intervel: ")
                years = 1
                interval = 1
                tickers = ss.get_symbol_list(
                    index=index_url, symbols_only=True)
                today_folder = date(years, parent)
                for symbol in tickers:
                    stockfolder = stock_folder(symbol, today_folder)
                    single_stock = stock_data.Get_Stock_History(
                        stockfolder, symbol, start_date)
                    normalized_df, close_column_index, csv_cleaner = single_stock.download_and_preprocess_data(
                        years, interval)
                    single_stock.train_evaluate_and_predict(
                        normalized_df, close_column_index, csv_cleaner)
            os.system('clear')

        elif user_options == '2':
            csv_path = input("Please enter the path of the CSV file: ")
            date, symbol, filename, desired_path = split_string(csv_path)
            stocks = stock_data.Get_Stock_History(desired_path, symbol, date)
            normalized_df, close_column_index, csv_cleaner = stocks.process_existing_data(
                csv_path)
            stocks.train_evaluate_and_predict(
                normalized_df, close_column_index, csv_cleaner)
            os.system('clear')
        elif user_options == '3':
            # processed_data_path = input("Please enter the path of the processed data CSV file: ")
            processed_data_path = '/root/home/git/2008-03-28/aapl/aapl_edited.csv'
            date, symbol, filename, desired_path = split_string(
                processed_data_path)
            stocks = stock_data.Get_Stock_History(desired_path, symbol, date)
            processed_df, close_column_index, csv_cleaner = stocks.load_processed_data(
                processed_data_path)
            lstm_model = LSTM_Model.LSTMModel(
                processed_df, close_column_index, symbol, desired_path)
            model_path = model_save_path(desired_path, symbol)

            lstm_model.train_evaluate_and_predict(csv_cleaner, model_path)

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
