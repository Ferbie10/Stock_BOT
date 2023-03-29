import os
import datetime
from MacroFact import *


def get_path_date(year, parent):
    today = datetime.date.today()
    years_past = int(year)
    start_year = today.year - years_past
    start_date = datetime.date(start_year, today.month, today.day)
    today_folder = os.path.join(parent, start_date.strftime('%Y-%m-%d'))
    if not os.path.exists(today_folder):
        os.makedirs(today_folder)
    return today_folder, start_date


def save_to_csv(data, path):
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        data.to_csv(full_path, index=True)
        print(f"File saved as: {full_path}")
    else:
        print(f"File {full_path} already exists.")
    return full_path


def edited_csv(data, path, symbol):
    full_path = os.path.join(path, f'{symbol}_edited.csv')
    if not os.path.exists(full_path):
        data.to_csv(full_path, index=True)
        print(f"File saved as: {full_path}")
    else:
        print(f"File {full_path} already exists.")
    return full_path


def split_string(path):
    # '/root/home/git/Stocks/Fed/macro_indicators.csv'
    parts = path.split('/')
    date = parts[-3]  # Stocks Fed macro_indicators.csv
    symbol = parts[-2]  # Fed macro_indicators.csv
    filename = parts[-1]  # macro_indicators.csv
    desired_path = "/".join(parts[:-1])  # /root/home/git/Stocks/Fed

    return date, symbol, filename, desired_path


def model_save_path(stockfolder, symbol):
    model_path = os.path.join(stockfolder, f'{symbol}.h5')
    return model_path


def stock_folder(symbol, path):

    filename = os.path.join(path, f'{symbol}')
    if not os.path.exists(filename):
        os.mkdir(filename)
    else:
        pass
    return filename


def indicators(start_date, path):
    # Define a dictionary with the series names and their corresponding FRED series IDs
    indicator_series_ids = {
        'GDP': 'GDPC1',  # Gross Domestic Product
        'CPI': 'CPIAUCSL',  # Consumer Price Index for All Urban Consumers
        'PPI': 'PPIACO',  # Producer Price Index for All Commodities
        'UnemploymentRate': 'UNRATE',  # Unemployment Rate
        'ConsumerConfidence': 'UMCSENT',  # University of Michigan: Consumer Sentiment
        'Fed-Funds': 'FEDFUNDS'  # Effective Federal Funds Rate
    }

    # Create a MacroFact DataFetcher object with the given start date
    macro_indicators = DataFetcher(start_date)

    # Fetch the macro indicators data for the specified series IDs
    all_indicators = macro_indicators.get_macro_indicators(
        indicator_series_ids)

    # Save the fetched data to a CSV file
    save_to_csv(all_indicators, path, 'macro_indicators.csv')
