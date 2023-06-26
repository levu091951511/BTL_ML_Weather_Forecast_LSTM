from tensorflow import keras
from sklearn.metrics import mean_absolute_error  
from keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer
from keras.layers import Dense, Dropout, LSTM, Concatenate, SimpleRNN, Masking, Flatten
from keras import losses
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal
from keras.callbacks import Callback
from keras.models import load_model
import json

class Models():
    
    def __init__(self, X, Y, n_outputs, n_lag, n_ft, units, batch, epochs, lr,
                 Xval=None, Yval=None, mask_value=-999.0, min_delta=0.001, patience=5):
        
        lstm_input = InputLayer(input_shape=(n_lag, n_ft)).output
        lstm_layer = LSTM(units, activation='tanh')(lstm_input)
        x = Dense(n_outputs)(lstm_layer)
        
        self.model = Model(inputs=lstm_input, outputs=x)
        self.batch = batch 
        self.epochs = epochs
        self.units = units
        self.lr = lr 
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        self.mask_value = mask_value
        self.min_delta = min_delta
        self.patience = patience

    def trainCallback(self):
        return EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta)

    def train(self):
        empty_model = self.model
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        empty_model.compile(loss=losses.MeanAbsoluteError(), optimizer=optimizer)

        history = empty_model.fit(
            self.X, 
            self.Y, 
            epochs=self.epochs, 
            batch_size=self.batch, 
            validation_data=(self.Xval, self.Yval), 
            shuffle=False,
            callbacks=[self.trainCallback()]
        )

        self.model = empty_model
        return history
    
    def save_model(self, filename, model):
        history = {
            'loss': [],
            'val_loss': [],
        }
        history['loss'].append(model.history.get('loss'))
        history['val_loss'].append(model.history.get('val_loss'))
        with open(filename.replace("h5", "json"), 'w') as f:

            json.dump(history, f)
        self.model.save(filename)