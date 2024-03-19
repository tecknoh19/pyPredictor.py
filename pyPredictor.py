import argparse
import os
import sys
import requests
import random
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ta import add_all_ta_features
from colorama import Fore, Style, Back

# Ticker Timestamp Dictionary
stock_ts = int(time.time()) - (365 * 10 * 24 * 60 * 60)  # Last 10 years
crypto_ts = int(time.time()) - (365 * 5 * 24 * 60 * 60)  # Last 5 years
default_ts = int(time.time()) - (365 * 24 * 60 * 60)  # Default: Last 365 days for symbols not in ticker_timestamp_dict

ticker_timestamp_dict = {
    "AAPL": stock_ts, "MSFT": stock_ts, "AMZN": stock_ts, "GOOGL": stock_ts, "TSLA": stock_ts, "FB": stock_ts,
    "GOOG": stock_ts, "BRK.B": stock_ts, "JPM": stock_ts, "JNJ": stock_ts, "V": stock_ts, "PG": stock_ts,
    "UNH": stock_ts, "MA": stock_ts, "HD": stock_ts, "DIS": stock_ts, "PYPL": stock_ts, "BAC": stock_ts,
    "INTC": stock_ts, "CMCSA": stock_ts, "VZ": stock_ts, "ADBE": stock_ts, "T": stock_ts, "NFLX": stock_ts,
    "NKE": stock_ts, "NVDA": stock_ts, "CRM": stock_ts, "XOM": stock_ts, "CSCO": stock_ts, "KO": stock_ts,
    "ORCL": stock_ts, "WMT": stock_ts, "IBM": stock_ts, "PEP": stock_ts, "ABBV": stock_ts, "TMO": stock_ts,
    "MO": stock_ts, "COST": stock_ts, "MRK": stock_ts, "BA": stock_ts, "LLY": stock_ts, "MDT": stock_ts,
    "ABT": stock_ts, "MMM": stock_ts, "PM": stock_ts, "F": stock_ts, "NVDA": stock_ts, "DOGE": crypto_ts,
    "BTC": crypto_ts, "ETH": crypto_ts, "XRP": crypto_ts, "LTC": crypto_ts, "ADA": crypto_ts, "SOL": crypto_ts,
    "DOT": crypto_ts, "USDC": crypto_ts, "DOGE": crypto_ts, "SHIB": crypto_ts, "LINK": crypto_ts,
    "MATIC": crypto_ts, "XLM": crypto_ts, "ATOM": crypto_ts, "ETC": crypto_ts, "VET": crypto_ts, "FIL": crypto_ts,
    "TRX": crypto_ts, "UNI": crypto_ts
}


def console(text, msg_type):
    color_codes = {
        'info': Fore.CYAN,
        'warning': Fore.YELLOW,
        'error': Fore.RED,
        'fatal': Fore.RED + Style.BRIGHT,
        'success': Fore.GREEN,
        'header': Fore.YELLOW + Back.BLUE
    }

    color = color_codes.get(msg_type.lower(), Fore.RESET)
    print(color + text + Style.RESET_ALL)


def predict_next_day(model_high, model_low, last_day_data):
    # Predict the high and low for the next day
    next_day_high = model_high.predict(last_day_data)
    next_day_low = model_low.predict(last_day_data)
    return next_day_high, next_day_low


def read_stock_data(file_name):
    df = pd.read_csv(file_name)
    return df


def preprocess_data(df):
    # Add technical indicators
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df


def train_models(X_train, y_train):
    # Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(random_state=42)
    param_grid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]}
    gb_cv = GridSearchCV(gb_model, param_grid, cv=TimeSeriesSplit(n_splits=5))
    gb_cv.fit(X_train, y_train)

    return rf_model, gb_cv.best_estimator_


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return mse, mae, rmse


def get_timestamp(ticker):
    return ticker_timestamp_dict.get(ticker.upper(), default_ts)


def yahoo_finance(args):
    ticker = args.ticker.upper()
    console("Pulling financial data from Yahoo Finance for: " + ticker, "info")

    # Get query start and end dates
    end_date = int(time.time())
    start_date = get_timestamp(ticker)

    # Define custom headers as a list
    available_headers = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/94.0.992.50 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:93.0) Gecko/20100101 Firefox/93.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/94.0.4606.71 Safari/537.36",
    ]

    # URL to download file from
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true"

    # Define the file path where the file will be saved
    filename = f"{ticker}-{int(time.time())}"
    file_path = f"./data_files/{filename}.csv"

    # Download data
    response = requests.get(url, headers={"User-Agent": random.choice(available_headers)})
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        console("Financial data has been successfully downloaded: " + file_path, "success")
    else:
        console("Failed to download file. Status code: " + str(response.status_code), "fatal")
        sys.exit()
    return file_path


def main():
    warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names.*")
    warnings.filterwarnings("ignore", message="DataFrame.fillna with 'method' is deprecated")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download stock data CSV from Yahoo Finance")
    parser.add_argument("-t", "--ticker", type=str, help="Stock ticker symbol", required=True)
    args = parser.parse_args()

    # Get financial data for the provided ticker
    data_csv = yahoo_finance(args)
    console("Reading financial data", "info")
    stock_data = read_stock_data(data_csv)

    # Preprocess the data
    console("Performing data preprocessing.", "info")
    processed_data = preprocess_data(stock_data)

    # Split the data into features and target variables
    console("Splitting data into features and target variables", "info")
    X = processed_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_high = processed_data['High']
    y_low = processed_data['Low']

    # Split the data into training and testing sets
    console("Creating training and testing sets.", "info")
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X, y_high, test_size=0.2, random_state=42)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.2, random_state=42)

    # Train the machine learning models
    console("Training the predictive models.", "info")
    rf_model_high, gb_model_high = train_models(X_train_high, y_train_high)
    rf_model_low, gb_model_low = train_models(X_train_low, y_train_low)

    # Evaluate the models
    console("Evaluating models.", "info")
    rf_mse_high, rf_mae_high, rf_rmse_high = evaluate_model(rf_model_high, X_test_high, y_test_high)
    gb_mse_high, gb_mae_high, gb_rmse_high = evaluate_model(gb_model_high, X_test_high, y_test_high)
    rf_mse_low, rf_mae_low, rf_rmse_low = evaluate_model(rf_model_low, X_test_low, y_test_low)
    gb_mse_low, gb_mae_low, gb_rmse_low = evaluate_model(gb_model_low, X_test_low, y_test_low)

    # Print evaluation results
    console("Evaluation Results: High Prediction:", "header")
    print(f"Random Forest - MSE: {rf_mse_high}, MAE: {rf_mae_high}, RMSE: {rf_rmse_high}")
    print(f"Gradient Boosting - MSE: {gb_mse_high}, MAE: {gb_mae_high}, RMSE: {gb_rmse_high}")

    console("Evaluation Results: Low Prediction:", "header")
    print(f"Random Forest - MSE: {rf_mse_low}, MAE: {rf_mae_low}, RMSE: {rf_rmse_low}")
    print(f"Gradient Boosting - MSE: {gb_mse_low}, MAE: {gb_mae_low}, RMSE: {gb_rmse_low}")

    # Make predictions for the following day
    last_day_data = X.iloc[-1:].values  # Assuming the last row contains data for the last day
    next_day_high_rf, next_day_low_rf = predict_next_day(rf_model_high, rf_model_low, last_day_data)
    next_day_high_gb, next_day_low_gb = predict_next_day(gb_model_high, gb_model_low, last_day_data)

    # Print predictions for the following day
    console("Predictions for the following day:", "header")
    print(f"Random Forest - High: {next_day_high_rf}, Low: {next_day_low_rf}")
    print(f"Gradient Boosting - High: {next_day_high_gb}, Low: {next_day_low_gb}")


if __name__ == "__main__":
    main()
