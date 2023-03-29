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


def save_to_csv(data, path, name):
    full_path = os.path.join(path, name)
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
        'FedFunds': 'FEDFUNDS',  # Effective Federal Funds Rate
        'GoodsTradeBalance': 'BOPGSTB',  # US Balance on Goods and Services Trade
        'MonetaryBase': 'BASE',  # St. Louis Adjusted Monetary Base
        'FederalDebt': 'GFDEBTN',  # Federal Debt: Total Public Debt
        'FederalSurplusDeficit': 'MTSDS133FMS',  # Federal Surplus or Deficit
    }
    macro_indicators = DataFetcher(start_date)
    all_indicators = macro_indicators.get_macro_indicators(
        indicator_series_ids)

    return all_indicators


def df_to_CSV(data, path, desire_name):
    full_path = os.path.join(path, desire_name)
    if not os.path.exists(full_path):
        data.to_csv(full_path, index=True)
        print(f"File saved as123: {full_path}")
        pass
    else:
        print(f"File {full_path} already exists123.")
        pass
    return full_path
