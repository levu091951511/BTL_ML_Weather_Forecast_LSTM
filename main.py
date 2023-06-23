import matplotlib.pyplot as plt 
import pandas as pd
from model import R2ScoreCallback, Models
from process_data import clean_data, transform_data
from numpy import random
import seaborn as sns
import json
import numpy as np
from keras.models import load_model

data = clean_data('data_1.csv')

# Number of lags (day back) to use for models
lag = 30
# Steps ahead to forecast 
n_ahead = 1
# Share of obs in testing 
test_share = 0.1
#  Share of obs in validation 
val_share = 0.1
# Epochs for training
epochs = 100
# Batch size 
batch_size = 128
# Learning rates
lr = 0.001
# Number of neurons in LSTM layer
n_layer = 5

Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, n_ft, train_mean, train_std= transform_data(data, lag, n_ahead, test_share, val_share)

# create model
model = Models(
    X=Xtrain,
    Y=Ytrain,
    n_outputs=n_ahead,
    n_lag=lag,
    n_ft=n_ft,
    n_layer=n_layer,
    batch=batch_size,
    epochs=epochs, 
    lr=lr,
    Xval=Xval,
    Yval=Yval,
)
# summary model
model.model.summary()

# print("start training....")
# history = model.train()
# print("complete training")

# # save models to Model
# model.save_model("model_1.h5", history)

# load model from Foder Model
models_test = load_model("Model/model_1.h5")

# get values history into train model
# loss, val_loss, accuracy, val_accuracy = model.get_acc_and_loss(history)

# Đọc lịch sử từ file JSON
with open('Model/model.json', 'r') as f:
    history = json.load(f)
    # Truy cập các giá trị lịch sử
    loss = history['loss'][0]
    val_loss = history['val_loss'][0]
    accuracy = history['accuracy'][0]
    val_accuracy = history['val_accuracy'][0]

n_epochs = range(len(loss))

#  ====== vẽ loss train và loss validation=======
plt.figure(figsize=(9, 7))
plt.plot(n_epochs, loss, 'r', label='Training loss', color='blue')
if val_loss is not None:
    plt.plot(n_epochs, val_loss, 'r', label='Validation loss', color='red')
plt.legend(loc=0)
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.show()

# ======== vẽ accuracy train và accuracy accuracy validation==========
plt.figure(figsize=(9, 7))
plt.plot(n_epochs, accuracy, 'r', label='Training accuracy', color='blue')
if val_accuracy is not None:
    plt.plot(n_epochs, val_accuracy, 'r', label='Validation accuracy', color='red')
plt.legend(loc=0)
plt.xlabel('Epoch')
plt.ylabel('Accuracy value')
plt.show()

# predict ======================================
y_pred = models_test.predict(Xtest)

# Featues used in models
features = ['Temperature_Avg']
# Hiển thị dự đoán và thực tế
for i in range(len(features)):
    plt.figure(figsize=(10, 6))
    plt.title(f"Forecast vs Actual for {features[i]}", fontsize=14)
    plt.plot(pd.Series(np.ravel(Ytest[:,i])), "bo", markersize=5, label="Actual")
    plt.plot(pd.Series(np.ravel(y_pred[:,i])), "r.", markersize=5, label="Forecast")
    plt.legend(loc="upper left")
    plt.xlabel("Time Periods")
    plt.show()


# Comparing the forecasts with the actual values
# yhat = [x[0] for x in model.predict(Xval)]
# y = [y[0] for y in Yval]

# # Creating the frame to store both predictions
# days = data['Date'].values[-len(y):]

# frame = pd.concat([
#     pd.DataFrame({'day': days, 'Temperature_Avg': y, 'type': 'original'}),
#     pd.DataFrame({'day': days, 'Temperature_Avg': yhat, 'type': 'forecast'})
# ])
# # frame = pd.DataFrame()
# # Creating the unscaled values column
# frame['temp_absolute'] = [(x * train_std['Temperature_Avg']) + train_mean['Temperature_Avg'] for x in frame['Temperature_Avg']]

# pivoted = frame.pivot_table(index='day', columns='type')
# pivoted.columns = ['_'.join(x).strip() for x in pivoted.columns.values]
# print(pivoted.tail(len(pivoted)))

