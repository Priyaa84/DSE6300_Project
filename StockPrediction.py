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
from finta import TA
import matplotlib.pyplot as plt


from flask import Flask, request
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.preprocessing import MinMaxScaler

#For matplotlib
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


#Define Server port
server_port = 5000
app = Flask(__name__)

@app.route('/stocks')
def predictions():

    ###############################################################################
    # Define Constants
    ###############################################################################


    num_days = 730     # The number of days of historical data to retrieve
    INTERVAL = '1d'     # Sample rate of historical data

    # List of symbols for technical indicators
    INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']


    ###############################################################################
    # Pull data from Yahoo Finance
    ###############################################################################

    start = (datetime.date.today() - datetime.timedelta( num_days ) )
    end = datetime.datetime.today()

    stock_name = ""
    while stock_name == "":
        try:
            symbol = input("Enter in Stock Symbol: ")
            stock_data = yf.download(symbol, start=start, end=end, interval=INTERVAL)
            stock_name = symbol
        except:
            print ("Please Enter a Valid Stock Symbol\n")

    stock_data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)

    temp_data = stock_data.iloc[-60:]
    temp_data['close'].plot(color='green')


    ###############################################################################
    # Create a DataFrame
    ###############################################################################

    # 1. Filter out the closing market price data
    close_data = stock_data.filter(['close'])

    # 2. Convert the data into array for easy evaluation
    dataset = close_data.values

    # 3. Scale/Normalize the data to make all values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 4. Creating training data size : 70% of the data
    training_data_len = math.ceil(len(dataset) *.7)
    train_data = scaled_data[0:training_data_len  , : ]

    # 5. Separating the data into x and y data
    x_train_data=[]
    y_train_data =[]
    for i in range(60,len(train_data)):
        x_train_data=list(x_train_data)
        y_train_data=list(y_train_data)
        x_train_data.append(train_data[i-60:i,0])
        y_train_data.append(train_data[i,0])

        # 6. Converting the training x and y values to numpy arrays
        x_train_data1, y_train_data1 = np.array(x_train_data), np.array(y_train_data)

        # 7. Reshaping training s and y data to make the calculations easier
        x_train_data2 = np.reshape(x_train_data1, (x_train_data1.shape[0],x_train_data1.shape[1],1))


    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train_data2.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train_data2, y_train_data1, batch_size=1, epochs=10)

    test_data = scaled_data[training_data_len - 60: , : ]
    x_test = []
    y_test =  dataset[training_data_len : , : ]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    # 2.  Convert the values into arrays for easier computation
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    # 3. Making predictions on the testing data
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse=np.sqrt(np.mean(((predictions - y_test)**2)))
    print("\n\nThe RSME of this prediction was: ", rmse, "\n\n")



    train = stock_data[300:training_data_len]
    valid = stock_data[training_data_len:]



    valid['Predictions'] = predictions
    symbol = symbol.upper()

#matplotlib Plot png
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    fig.suptitle(symbol+' Prediction Model')
    fig.supxlabel('Date')
    fig.supylabel('Close Price')
    axis.plot(train['close'])
    axis.plot(valid[['close', 'Predictions']])
    axis.legend(['Trained', 'Actual', 'Predicted'], loc='upper left')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
###############################################################################
# Create a Dashboard
###############################################################################


if __name__ == "__main__":
    app.run('0.0.0.0',port=server_port)
