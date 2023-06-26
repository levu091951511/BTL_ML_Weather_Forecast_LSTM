import matplotlib.pyplot as plt 
import pandas as pd
from model import Models
from process_data import clean_data, transform_data
from numpy import random
import seaborn as sns
import json
import numpy as np
from keras.models import load_model
from sklearn.metrics import r2_score, mean_absolute_error

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

print("start training....")
history = model.train()
print("complete training")

model_path = "Model/model_2"
# save models to Model
model.save_model(model_path + "h5", history)

# load model from Foder Model
models_test = load_model(model_path + "h5")

# Đọc lịch sử từ file JSON
with open(model_path + 'json', 'r') as f:
    history = json.load(f)
    # Truy cập các giá trị lịch sử
    loss = history['loss'][0]
    val_loss = history['val_loss'][0]

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

# predict ======================================
y_pred = models_test.predict(Xtest)

# Featues used in models
features = ['Temperature_Avg']

print('MAE TEST: {:.3f}'.format(mean_absolute_error(y_pred, Ytest)))

# Comparing the forecasts with the actual values
yhat = [x[0] for x in models_test.predict(Xtest)]
y = [y[0] for y in Ytest]

# Creating the frame to store both predictions
days = data['Date'].values[-len(y):]

frame = pd.concat([
    pd.DataFrame({'day': days, 'Temperature_Avg': y, 'type': 'original'}),
    pd.DataFrame({'day': days, 'Temperature_Avg': yhat, 'type': 'forecast'})
])

# Creating the unscaled values column
frame['temp_absolute'] = [(x * train_std['Temperature_Avg']) + train_mean['Temperature_Avg'] for x in frame['Temperature_Avg']]

pivoted = frame.pivot_table(index='day', columns='type')
pivoted.columns = ['_'.join(x).strip() for x in pivoted.columns.values]
pivoted.to_csv("predict_model_2.csv")

# Đọc dữ liệu từ file CSV
data = pd.read_csv('predict_model_2.csv')

# Lấy cột 'day' và cột 'temp_absolute_forecast'
day = data['day']
temp_forecast = data['temp_absolute_forecast']
temp_original = data['temp_absolute_original']

# Tạo biểu đồ đường
plt.plot(day, temp_forecast, label='Forecast')
plt.plot(day, temp_original, label='Original')

# Đặt tên cho trục x và y, và tiêu đề cho biểu đồ
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Temperature Forecast vs Original')

# Tạo mảng các vị trí trên trục hoành và nhãn tương ứng
step = 100
xticks_positions = np.arange(0, len(day), step)
xticks_labels = day[::step]

# Đặt các vị trí và nhãn trên trục hoành
plt.xticks(xticks_positions, xticks_labels)

# Hiển thị chú thích cho các đường
plt.legend()

# Hiển thị biểu đồ
plt.show()