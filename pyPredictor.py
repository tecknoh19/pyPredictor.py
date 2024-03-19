
'''
TODO:
    1. Modify so that if provide ticker is not in ticker_timestamp_dict, a default start stamp
       of X days is used (maybe a year)?
    
'''

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
stock_ts = int(time.time()) - (365 * 10 * 24 * 60 * 60) # last 10 years
cryto_ts = int(time.time()) - (365 * 5 * 24 * 60 * 60) # last 5 years
nf_ticker = int(time.time()) - (365 * 24 * 60 * 60) # default ticker (last 365 days) for symbols not in ticker_timestamp_dict

ticker_timestamp_dict = {
    "AAPL": stock_ts,  # Apple
    "MSFT": stock_ts,  # Microsoft
    "AMZN": stock_ts,  # Amazon
    "GOOGL": stock_ts,  # Google
    "TSLA": stock_ts,  # Tesla
    "FB": stock_ts,  # Facebook
    "GOOG": stock_ts,  # Alphabet Inc.
    "BRK.B": stock_ts,  # Berkshire Hathaway
    "JPM": stock_ts,  # JPMorgan Chase
    "JNJ": stock_ts,  # Johnson & Johnson
    "V": stock_ts,  # Visa Inc.
    "PG": stock_ts,  # Procter & Gamble
    "UNH": stock_ts,  # UnitedHealth Group
    "MA": stock_ts,  # Mastercard
    "HD": stock_ts,  # Home Depot
    "DIS": stock_ts,  # The Walt Disney Company
    "PYPL": stock_ts,  # PayPal Holdings
    "BAC": stock_ts,  # Bank of America
    "INTC": stock_ts,  # Intel Corporation
    "CMCSA": stock_ts,  # Comcast
    "VZ": stock_ts,  # Verizon Communications
    "ADBE": stock_ts,  # Adobe Inc.
    "T": stock_ts,  # AT&T Inc.
    "NFLX": stock_ts,  # Netflix
    "NKE": stock_ts,  # Nike, Inc.
    "NVDA": stock_ts,  # NVIDIA
    "CRM": stock_ts,  # Salesforce.com
    "XOM": stock_ts,  # ExxonMobil
    "CSCO": stock_ts,  # Cisco Systems
    "KO": stock_ts,  # The Coca-Cola Company
    "ORCL": stock_ts,  # Oracle Corporation
    "WMT": stock_ts,  # Walmart
    "IBM": stock_ts,  # IBM
    "PEP": stock_ts,  # PepsiCo
    "ABBV": stock_ts,  # AbbVie Inc.
    "TMO": stock_ts,  # Thermo Fisher Scientific
    "ABBV": stock_ts,  # AbbVie Inc.
    "TMO": stock_ts,  # Thermo Fisher Scientific
    "MO": stock_ts,  # Altria Group
    "COST": stock_ts,  # Costco Wholesale
    "MRK": stock_ts,  # Merck & Co.
    "BA": stock_ts,  # Boeing
    "LLY": stock_ts,  # Eli Lilly and Company
    "MDT": stock_ts,  # Medtronic
    "ABT": stock_ts,  # Abbott Laboratories
    "MMM": stock_ts,  # 3M
    "PM": stock_ts,  # Philip Morris International
    "GOOGL": stock_ts,  # Google
    "F": stock_ts,  # Ford
    "NVDA": stock_ts,  # NVIDIA
    "DOGE": cryto_ts,  # Dogecoin
    "BTC": cryto_ts,  # Bitcoin
    "ETH": cryto_ts,  # Ethereum
    "XRP": cryto_ts,  # Ripple
    "LTC": cryto_ts,  # Litecoin
    "ADA": cryto_ts,  # Cardano
    "SOL": cryto_ts,  # Solana
    "DOT": cryto_ts,  # Polkadot
    "USDC": cryto_ts,  # USD Coin
    "DOGE": cryto_ts,  # Dogecoin
    "SHIB": cryto_ts,  # Shiba Inu
    "LINK": cryto_ts,  # Chainlink
    "MATIC": cryto_ts,  # Polygon
    "XLM": cryto_ts,  # Stellar
    "ATOM": cryto_ts,  # Cosmos
    "ETC": cryto_ts,  # Ethereum Classic
    "VET": cryto_ts,  # VeChain
    "FIL": cryto_ts,  # Filecoin
    "TRX": cryto_ts,  # TRON
    "UNI": cryto_ts,  # Uniswap
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

def get_timestamp(ticker, ticker_timestamp_dict = ticker_timestamp_dict, nf_ticker = nf_ticker):
    if ticker in ticker_timestamp_dict:
        return ticker_timestamp_dict[ticker]
    else:
        return nf_ticker

def yahooFinance(args):
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
        #print(f"File '{file_path}' has been successfully downloaded.")
    else:
        console("Failed to download file. Status code: " + str(response.status_code),"fatal")
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
    data_csv = yahooFinance(args)
    console("Reading financial data","info")
    stock_data = read_stock_data(data_csv)

    # Preprocess the data
    console("Performing data preprocessing.","info")
    processed_data = preprocess_data(stock_data)

    # Split the data into features and target variables
    console("Splitting data intp features and target variables","info")
    X = processed_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_high = processed_data['High']
    y_low = processed_data['Low']

    # Split the data into training and testing sets
    console("Creating training and testing sets.","info")
    X_train, X_test, y_train_high, y_test_high = train_test_split(X, y_high, test_size=0.2, random_state=42)
    X_train, X_test, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.2, random_state=42)

    # Train the machine learning models
    console("Training the predicitive model.","info")
    rf_model_high, gb_model_high = train_models(X_train, y_train_high)
    rf_model_low, gb_model_low = train_models(X_train, y_train_low)

    # Evaluate the models
    console("Evaluating models.","info")
    rf_mse_high, rf_mae_high, rf_rmse_high = evaluate_model(rf_model_high, X_test, y_test_high)
    gb_mse_high, gb_mae_high, gb_rmse_high = evaluate_model(gb_model_high, X_test, y_test_high)
    rf_mse_low, rf_mae_low, rf_rmse_low = evaluate_model(rf_model_low, X_test, y_test_low)
    gb_mse_low, gb_mae_low, gb_rmse_low = evaluate_model(gb_model_low, X_test, y_test_low)

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
