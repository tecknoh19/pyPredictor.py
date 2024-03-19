import argparse, os, requests, random, time, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings


def read_stock_data(file_name):
    df = pd.read_csv(file_name)
    return df

def preprocess_data(df):
    # You can add feature engineering or preprocessing steps here if needed
    return df

# Train a machine learning model
def train_model(X_train, y_train):
    model = RandomForestRegressor()  # You can choose any regression model
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Make predictions for the following day
def make_predictions(model, last_day_data):
    # You can use the last day's data to predict the next day's high/low
    prediction = model.predict(last_day_data)
    return prediction

def get_timestamp(ticker, ticker_timestamp_dict):
    # Check if the provided ticker exists in the dictionary
    if ticker in ticker_timestamp_dict:
        return ticker_timestamp_dict[ticker]
    else:
        return None

def yahooFinance(args):

    ticker = args.ticker.upper()

    # My Tickers (stock_ts = 1/1/2014)
    ticker_timestamp_dict = {
        "DOGE": cryto_ts, # Dogecoin
        "GOOGL": stock_ts, # Google
        "F": stock_ts, # Ford
        "NVDA": stock_ts, # NVIDIA
    }

    # Get query start and end dates.  Starts dates are hard coded as each ticker is capable
    # of having a different start date based on the commodities initiation date and / or when
    # yahoo started tracking its date.  DOGE for example cannot go back further than 2017, whereas
    # Ford Motors can go as far as 2000.
    end_date = int(time.time())
    start_date = get_timestamp(ticker, ticker_timestamp_dict)

    # URL to download file from
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true"

    if args.debug:
        print(url)
    #sys.exit()

    # Define the file path where the file will be saved
    filename = f"{ticker}-{int(time.time())}"
    file_path = f"./data_files/{filename}.csv"

    # Define custom headers as a list
    available_headers = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/94.0.992.50 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:93.0) Gecko/20100101 Firefox/93.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/94.0.4606.71 Safari/537.36",
    ]

    # Randomly select one user-agent header
    headers = random.choice(available_headers)

    # Use requests library to download the file
    response = requests.get(url, headers={"User-Agent": headers})
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File '{file_path}' has been successfully downloaded.")
        return file_path
    else:
        print("Failed to download file. Status code:", response.status_code)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download stock data CSV from Yahoo Finance")
    parser.add_argument("-t", "--ticker", type=str, help="Stock ticker symbol", required=True)
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging mode", default=False)
    args = parser.parse_args()

    # Get financial data for the provided ticker
    data_csv = yahooFinance(args)
    stock_data = read_stock_data(data_csv)

    # Suppress the warning about invalid feature names during model fitting
    warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names.*")
    #warnings.filterwarnings("ignore", message="A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy.*")


    # Preprocess the data
    processed_data = preprocess_data(stock_data)

    # Split the data into features and target variables
    X = processed_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    X.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    y_high = processed_data['High']
    y_low = processed_data['Low']

    # Split the data into training and testing sets
    X_train, X_test, y_train_high, y_test_high, y_train_low, y_test_low = train_test_split(
        X, y_high, y_low, test_size=0.2, random_state=42
    )

    # Train the machine learning models
    model_high = train_model(X_train, y_train_high)
    model_low = train_model(X_train, y_train_low)

    # Evaluate the models
    mse_high = evaluate_model(model_high, X_test, y_test_high)
    mse_low = evaluate_model(model_low, X_test, y_test_low)
    print(f"Mean Squared Error for High: {mse_high}")
    print(f"Mean Squared Error for Low: {mse_low}")

    # Make predictions for the following day
    last_day_data = X.iloc[-1:].values  # Assuming the last row contains data for the last day
    high_prediction = make_predictions(model_high, last_day_data)
    low_prediction = make_predictions(model_low, last_day_data)
    print("Predictions for the following day:")
    print(f"High: {high_prediction}")
    print(f"Low: {low_prediction}")

if __name__ == "__main__":
    main()
