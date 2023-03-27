import LSTM_Model
import datetime
import os
import keras
import stock_data
import pandas as pd
import dataPrep


def main():
    parent = '/root/home/git/'

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    today = datetime.date.today()
    years_past = 5
    start_year = today.year - years_past
    start_date = datetime.date(start_year, today.month, today.day)
    today_folder = os.path.join(parent, start_date.strftime('%Y-%m-%d'))
    if not os.path.exists(today_folder):
        os.makedirs(today_folder)
    elif os.path.exists(today_folder):
        pass
    loop = 0
    while loop == 0:
        user_options = input(
            "Enter:\n1 for a new model\n2 to load in a CSV\n3 to load processed CSV\n4 to load in a Model\n0 to end the program\n")
        if user_options == '1':
            indivdual_or_list = int(
                input("Enter 1 for individual stock or 2 for stock index:  "))
            if indivdual_or_list == 1:
                stock_list = input("Please enter the stock Symbol:  ")
                single_stock = stock_data.Get_Stock_History(
                    today_folder, stock_list, start_date)
                normalized_df, close_column_index, csv_cleaner = single_stock.download_and_preprocess_data(
                    stock_list)
                single_stock.train_evaluate_and_predict(
                    normalized_df, close_column_index, stock_list, csv_cleaner, today_folder)

            else:
                index_url = input("Please enter the URL of the Index List:   ")
                Complist = dataPrep.Get_SP500(url)
                stock_list = Complist.download()
                stocks = dataPrep.Get_Stock_History(
                    today_folder, stock_list, start_date, today)
                stocks.compstockdata()
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
