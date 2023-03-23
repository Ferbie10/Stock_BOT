def main():
    parent = '/root/home/git/'

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # Initialize Date object to get start date
    today = datetime.date.today()
    years_past = 5
    start_year = today.year - years_past
    start_date = datetime.date(start_year, today.month, today.day)
    today_folder = os.path.join(parent, start_date.strftime('%Y-%m-%d'))
    if not os.path.exists(today_folder):
        os.makedirs(today_folder)
    elif os.path.exists(today_folder):
        print(f"Path  {today_folder}")
    loop = 0
    while (loop == 0):
        user_options = int(input("Please Enter 1 for to create a new model \nEnter 2 to load in a CSV\nEnter 3 to load a new Model\n"))

        if user_options == 1:
            indivdual_or_list = int(input("Enter 1 for indivdual stock or 2 for stock index:  "))
            if indivdual_or_list == 1:
                stock_list = input("Please enter the stock Symbol:  ")

            else:
                index_url = input("Please enter the url of the Index List:   ")
                Complist = Get_SP500(url)
                stock_list = Complist.download()
            stocks = Get_Stock_History(today_folder, stock_list, start_date, today)
            stocks.compstockdata()
            os.system('cls')
        elif user_options == 2:
            csv_path = input("Please enter the path of the CSV file: ")
        elif user_options == 3:
            model_Path = input("Please enter the model path: ")
            symbol = 'aapl'

            # Load and preprocess the data before creating the LSTMModel instance
            filename = stock_data.download_stock_history(symbol)
            normalized_df, close_column_index, csv_cleaner = stock_data.preprocess_stock_data(filename, symbol)

            lstm_model = LSTM_Model.LSTMModel(cleaned_df=normalized_df, close_column_index=close_column_index, symbol=symbol)
            lstm_model.evaluate()
        else:
            loop = 1