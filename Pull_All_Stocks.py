###############################################################################
# DSE 6300 Final Project
#   
# Stock Prediction and Analysis
#
# Karolina, Ahalya, Priya
#
###############################################################################

###############################################################################
# Import Required Packages
###############################################################################

import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import mysql.connector
import findspark
from pyspark.sql import SparkSession


###############################################################################
# Define Constants
###############################################################################

num_days = 1825            # The number of days of historical data to retrieve (5 Years)
interval_pulled = '1d'     # Sample rate of historical data
appended_data = []


###############################################################################
# Pull Top 500 Stocks from Wikipedia
###############################################################################

# There are 2 tables on the Wikipedia page
# we want the first table

list_of_stocks=pd.read_html('https://en.wikipedia.org/wiki/Russell_1000_Index')
first_table = list_of_stocks[2]
df = first_table

symbols = df['Ticker'].values.tolist()
df= df.rename(columns={"Ticker": "Symbol"})

###############################################################################
# Pull Stock Prices from Yahoo Finance
###############################################################################

start = (datetime.date.today() - datetime.timedelta( num_days ) )
end = datetime.datetime.today()


for symbol in symbols:
    stock_data      = yf.download(symbol, start=start, end=end, interval=interval_pulled)
    stock_data['Symbol'] = symbol
    appended_data.append(stock_data)

all_stock_data = pd.concat(appended_data)
all_stock_data = all_stock_data.reset_index()

joined_data = pd.merge(df, all_stock_data)


joined_data.to_csv("Stock_Data_Top_1000.csv")
