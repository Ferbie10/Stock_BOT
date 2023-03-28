import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
import ssl
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import dataPrep
import LSTM_Model
import datetime
from kerastuner.tuners import RandomSearch
from LSTM_Model import build_lstm_model
from tensorflow.keras.callbacks import TensorBoard


class Get_Stock_History:
    def __init__(self, path, symbol,start_date):
        self.path = path
        self.symbol = symbol
        self.start_date = start_date


    def save_to_csv(self, data, filename):
        data.to_csv(filename, index=True)

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

        filename = os.path.join(self.path, f'{self.symbol}.csv')

        if not os.path.exists(filename):
            tik_history.to_csv(filename)
        return filename
    def download_macro_indicators(self,):

        indicator_series_ids = {
        'GDP': 'GDPC1',
        'CPI': 'CPIAUCSL',
        'PPI': 'PPIACO',
        'UnemploymentRate': 'UNRATE',
        'ConsumerConfidence': 'UMCSENT',
        'HousingStarts': 'HOUST',
        'ExistingHomeSales': 'EXHOSLUSM495S',
        'NewHomeSales': 'HSN1F'
    }
        

    def preprocess_stock_data(self, filename):
        output_file_path = os.path.join(self.path, f'{self.symbol}_edited.csv')

        # Call CSV cleaner for the newly created file
        csv_cleaner = dataPrep.CSVCleaner(
            filename, output_file_path, self.symbol)
        csv_cleaner.clean()
        csv_cleaner.transform(output_file_path)
        normalized_df, scaler = self.normalize_stock_data(csv_cleaner.df)

        return normalized_df, csv_cleaner.df.columns.get_loc('close'), csv_cleaner

    def download_and_preprocess_data(self, years, interval):
        filename = self.download_stock_history(years, interval)
        normalized_df, close_column_index, csv_cleaner = self.preprocess_stock_data(
            filename)
        return normalized_df, close_column_index, csv_cleaner

    def process_existing_data(self, filename):
        normalized_df, close_column_index, csv_cleaner = self.preprocess_stock_data(
            filename, self.symbol)
        return normalized_df, close_column_index, csv_cleaner

    def load_processed_data(self, filepath):
        processed_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.symbol = os.path.basename(filepath).split('_')[0]
        close_column_index = processed_df.columns.get_loc('close')

        # Create a CSVCleaner object with the preprocessed data
        csv_cleaner = dataPrep.CSVCleaner(filepath, filepath, self.symbol)
        csv_cleaner.df = processed_df

        return processed_df, close_column_index, self.symbol, csv_cleaner

    def train_evaluate_and_predict(self, normalized_df, close_column_index, csv_cleaner, model_filepath=None):
        lstm_model = LSTM_Model.LSTMModel(
            cleaned_df=normalized_df, close_column_index=close_column_index, symbol=self.symbol, today_folder=self.path)

        if model_filepath:
            # Load the saved model
            lstm_model.load_model(model_filepath)
        else:
            # Build and train a new model
            x_train, y_train, x_test, y_test = lstm_model.preprocess()

            tuner = RandomSearch(
                lambda hp: build_lstm_model(hp, lstm_model.num_features),
                objective='val_loss',
                max_trials=10,
                executions_per_trial=1,
                directory=f'{self.path}/random_search',
                project_name='hyperparameter_tuning'
            )

            tuner.search_space_summary()

            tuner.search(x_train, y_train,
                         epochs=10,
                         batch_size=64,
                         validation_data=(x_test, y_test),
                         callbacks=[TensorBoard(log_dir='./logs')])

            tuner.results_summary()
            best_model = tuner.get_best_models(num_models=1)[0]
            best_model.save(
                f'{lstm_model.today_folder}/{lstm_model.symbol}.h5')
            lstm_model.model = best_model  # Update the model in lstm_model

        # Evaluate the model
        lstm_model.evaluate()

        # Get the predictions
        prediction_days = [1, 5, 20]
        predictions = lstm_model.predict_future_close_price(
            csv_cleaner, prediction_days)

        # Transform predictions dictionary to a list of tuples
        predictions_list = [(pd.Timestamp.now(), days, pred[0], pred[1])
                            for days, pred in predictions.items()]

        # Create a new DataFrame for the predictions
        predictions_df = pd.DataFrame(predictions_list, columns=[
            'timestamp', 'prediction_days', 'prediction', 'future_close_price'])

        # Add the self.symbol to the predictions DataFrame
        predictions_df['symbol'] = self.symbol

        # Rearrange the columns in the desired order
        predictions_df = predictions_df[[
            'symbol', 'timestamp', 'prediction_days', 'prediction', 'future_close_price']]

        # Save the predictions DataFrame to a new CSV file
        predictions_df.to_csv(
            f'{self.path}/{self.symbol}_Predictions_Separate.csv', index=False)
