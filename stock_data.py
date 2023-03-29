import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
import ssl
import os
from sklearn.preprocessing import MinMaxScaler
import dataPrep
from useful_functions import *


class Get_Stock_History:
    def __init__(self, path, symbol, start_date):
        self.path = path
        self.symbol = symbol
        self.start_date = start_date

    def normalize_stock_data(self, df):
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=np.number)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(numeric_df.values)
        normalized_df = pd.DataFrame(
            scaled_data, columns=numeric_df.columns, index=numeric_df.index)
        return normalized_df, scaler

    def download_stock_history(self, year, interval):
        ticker = yf.Ticker(self.symbol)
        tik_history = ticker.history(
            period=f'{year}y', interval=f'{interval}d')

        filename = save_to_csv(tik_history, self.path, f'{self.symbol}.csv')
        return filename

    def preprocess_stock_data(self, filename, output_name):

        # Call CSV cleaner for the newly created file
        csv_cleaner = dataPrep.CSVCleaner(
            filename, self.path, self.symbol)
        csv_cleaner.clean()
        transformed_df = csv_cleaner.transform()
        df_to_CSV(transformed_df, self.path, output_name)
        normalized_df, scaler = self.normalize_stock_data(transformed_df)

        return normalized_df, csv_cleaner.df.columns.get_loc('close'), csv_cleaner

    def download_and_preprocess_data(self, years, interval):
        filename = self.download_stock_history(years, interval)
        normalized_df, close_column_index, csv_cleaner = self.preprocess_stock_data(
            filename)
        return normalized_df, close_column_index, csv_cleaner

    def process_existing_data(self, filename, output_name):
        normalized_df, close_column_index, csv_cleaner = self.preprocess_stock_data(
            filename, output_name)
        return normalized_df, close_column_index, csv_cleaner

    def load_processed_data(self, filepath):
        processed_df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        close_column_index = processed_df.columns.get_loc('close')

        # Create a CSVCleaner object with the preprocessed data
        csv_cleaner = dataPrep.CSVCleaner(filepath, filepath, self.symbol)
        csv_cleaner.df = processed_df

        return processed_df, close_column_index, csv_cleaner
