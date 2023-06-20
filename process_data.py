import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime
import numpy as np
import random
import calendar

def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    n_features = ts.shape[1]
    # Creating placeholder lists
    X, Y = [], []
    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    X = np.reshape(X, (X.shape[0], lag, n_features))
    return X, Y

def clean_data(filename):
    #  xử lý data
    d = pd.read_csv(filename)

    d['Year'] = d['Year'].astype(int)
    d['Month'] = d['Month'].astype(int)
    d['Day'] = d['Day'].astype(int)

    d['Date'] = pd.to_datetime(d[['Year', 'Month', 'Day']])

    d = d.drop(['Year', 'Month', 'Day'], axis=1)

    # Featues used in models
    features = ['Temperature_Avg', 'Dew_Point_Avg', 'Humidity_Avg', 'Wind_Speed_Avg', 'Pressure_Avg']
    d['Temperature_Avg'] = (d['Temperature_Avg'] - 32) * 5/9
    # Aggregating to hourly level
    d = d.groupby('Date', as_index=False)[features].mean()

    # Creating the data column
    d['date'] = [x.date() for x in d['Date']]
    return d

def transform_data(d, lag=60, n_ahead=1, test_share=0.1):
    # The features used in the modeling 
    features_final = ['Temperature_Avg', 'Dew_Point_Avg', 'Humidity_Avg', 'Wind_Speed_Avg', 'Pressure_Avg']

    # Subseting only the needed columns 
    ts = d[features_final]
    nrows = ts.shape[0]

    # Spliting into train and test sets
    train = ts[0:int(nrows * (1 - test_share))]
    test = ts[int(nrows * (1 - test_share)):]


    # Scaling the data 
    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std

    # Creating the final scaled frame 
    ts_s = pd.concat([train, test])

    X, Y = create_X_Y(ts_s.values, lag=lag, n_ahead=n_ahead)

    n_ft = X.shape[2]

    # Spliting into train and test sets 
    Xtrain, Ytrain = X[0:int(X.shape[0] * (1 - test_share))], Y[0:int(X.shape[0] * (1 - test_share))]
    Xval, Yval = X[int(X.shape[0] * (1 - test_share)):], Y[int(X.shape[0] * (1 - test_share)):]
    return Xtrain, Ytrain, Xval, Yval, n_ft