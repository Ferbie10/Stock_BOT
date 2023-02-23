import numpy as np
import pandas as pd

class CSVCleaner:
    def __init__(self, csv_file_path,output_file_path):
        self.df = pd.read_csv(csv_file_path)
        self.output_file_path = output_file_path
        
    def clean(self):
        # Remove unnecessary columns and rename columns
        self.df.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
        self.df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'},
                       inplace=True)

        # Remove rows with missing values
        self.df.dropna(inplace=True)

        # Create a new column with the daily price change as a percentage
        self.df['change'] = (self.df['close'] - self.df['open']) / self.df['open'] * 100

        # Convert date to datetime object
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])

        # Set date as index
        self.df.set_index('Datetime', inplace=True)

    def transform(self, output_file_path):
        # Add a column for the 5-day moving average of the price change
        self.df['change_5d_avg'] = self.df['change'].rolling(window=5).mean()

        # Add a column for the 20-day moving average of the price change
        self.df['change_20d_avg'] = self.df['change'].rolling(window=20).mean()

        # Create a binary classification column indicating whether the price increased or decreased
        self.df['price_increase'] = np.where(self.df['change'] >= 0, 1, 0)

        # Save the transformed data to a new CSV file
        self.df.to_csv(output_file_path)


