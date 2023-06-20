import matplotlib.pyplot as plt 
import pandas as pd
from model import R2ScoreCallback, NNMultistepModel
from process_data import clean_data, transform_data

data = clean_data('data_1.csv')

# Number of lags (hours back) to use for models
lag = 60
# Steps ahead to forecast 
n_ahead = 1
# Share of obs in testing 
test_share = 0.1
# Epochs for training
epochs = 100
# Batch size 
batch_size = 512
# Learning rates
lr = 0.001
# Number of neurons in LSTM layer
n_layer = 10

Xtrain, Ytrain, Xval, Yval, n_ft= transform_data(data, lag, n_ahead, test_share)

# create model
model = NNMultistepModel(
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

# model.save_model("model.h5")
# model.save_model()

# history = model.load_model("model.h5")

loss = history.history.get('loss')
val_loss = history.history.get('val_loss')

n_epochs = range(len(loss))

plt.figure(figsize=(9, 7))
plt.plot(n_epochs, loss, 'r', label='Training loss', color='blue')
if val_loss is not None:
    plt.plot(n_epochs, val_loss, 'r', label='Validation loss', color='red')
plt.legend(loc=0)
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.show()


accuracy = history.history.get('accuracy')
val_accuracy = history.history.get('val_accuracy')

plt.figure(figsize=(9, 7))
plt.plot(n_epochs, accuracy, 'r', label='Training accuracy', color='blue')
if val_accuracy is not None:
    plt.plot(n_epochs, val_accuracy, 'r', label='Validation accuracy', color='red')
plt.legend(loc=0)
plt.xlabel('Epoch')
plt.ylabel('Accuracy value')
plt.show()