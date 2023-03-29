import numpy as np
import pandas as pd


class CSVCleaner:
    def __init__(self, csv_file_path, path, symbol):
        self.df = pd.read_csv(csv_file_path)
        self.symbol = symbol
        self.add_symbol()
        self.path

    def add_symbol(self):
        if 'symbol' in self.df.columns:
            # The column already exists, so we can move on
            pass
        else:
            # The column does not exist, so we need to add it
            self.df.insert(0, 'symbol', self.symbol)

    def clean(self):
        # Remove unnecessary columns and rename columns
        if 'Dividends' in self.df.columns or 'Stock Splits' in self.df.columns:
            self.df.drop(columns=['Dividends', 'Stock Splits'], inplace=True)

        self.df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'},
                       inplace=True)

        # Remove rows with missing values
        self.df.dropna(inplace=True)

        # Create a new column with the daily price change as a percentage
        self.df['change'] = (self.df['close'] -
                             self.df['open']) / self.df['open'] * 100

        # Convert date to datetime object
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Set date as index
        self.df.set_index('Date', inplace=True)

    def transform(self, output_file_path):
        # Calculate the 14-day RSI
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.df['rsi'] = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        self.df['20d_sma'] = self.df['close'].rolling(window=20).mean()
        self.df['20d_std'] = self.df['close'].rolling(window=20).std()
        self.df['bollinger_upper'] = self.df['20d_sma'] + \
            (self.df['20d_std'] * 2)
        self.df['bollinger_lower'] = self.df['20d_sma'] - \
            (self.df['20d_std'] * 2)

        # Calculate MACD
        self.df['12d_ema'] = self.df['close'].ewm(span=12).mean()
        self.df['26d_ema'] = self.df['close'].ewm(span=26).mean()
        self.df['macd'] = self.df['12d_ema'] - self.df['26d_ema']
        self.df['macd_signal'] = self.df['macd'].ewm(span=9).mean()

        # Calculate Stochastic Oscillator
        self.df['14d_high'] = self.df['high'].rolling(window=14).max()
        self.df['14d_low'] = self.df['low'].rolling(window=14).min()
        self.df['%K'] = (self.df['close'] - self.df['14d_low']) / \
            (self.df['14d_high'] - self.df['14d_low']) * 100
        self.df['%D'] = self.df['%K'].rolling(window=3).mean()

        # Calculate Average True Range (ATR)
        self.df['true_range'] = np.maximum(self.df['high'] - self.df['low'],
                                           np.maximum(np.abs(self.df['high'] - self.df['close'].shift(1)), np.abs(self.df['low'] - self.df['close'].shift(1))))
        self.df['atr'] = self.df['true_range'].rolling(window=14).mean()
        # Calculate On-Balance Volume (OBV)
        self.df['obv'] = np.where(self.df['close'] > self.df['close'].shift(1),
                                  self.df['volume'],
                                  np.where(self.df['close'] < self.df['close'].shift(1),
                                           -self.df['volume'],
                                           0)).cumsum()
        self.df.dropna(inplace=True)
        # Save the transformed data to a new CSV file
        self.df.to_csv(output_file_path)

        def clean_fed(self):
            # Drop rows with missing values
            self.df = self.df.dropna()

            # Convert the date column to a datetime object
            self.df['Date'] = pd.to_datetime(self.df[''])

            edited_csv(self.df, self.path, self.symbol)
