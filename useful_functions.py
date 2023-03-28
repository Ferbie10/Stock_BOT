import os
import datetime


def save_to_csv(data, path, filename):
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        data.to_csv(full_path, index=True)
        print(f"File saved as: {full_path}")
    else:
        print(f"File {full_path} already exists.")
    return full_path


def split_string(path):
    parts = path.split('/')
    date = parts[-3]
    symbol = parts[-2]
    filename = parts[-1]
    desired_path = "/".join(parts[:-1])

    return date, symbol, filename, desired_path


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
    return today_folder, start_date


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
    indicator_series_ids = {'GDP': 'GDPC1', 'CPI': 'CPIAUCSL', 'PPI': 'PPIACO', 'UnemploymentRate': 'UNRATE',
                            'ConsumerConfidence': 'UMCSENT', 'HousingStarts': 'HOUST', 'ExistingHomeSales': 'EXHOSLUSM495S', 'NewHomeSales': 'HSN1F'}
    macro_indicators = MacroFact.DataFetcher(start_date)
    all_indicators = macro_indicators.get_macro_indicators(
        indicator_series_ids)
    save_to_csv(all_indicators, path, 'macro_indicators.csv')
