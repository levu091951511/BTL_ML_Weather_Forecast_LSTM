from tensorflow import keras
from sklearn.metrics import r2_score
from keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer
from keras.layers import Dense, Dropout, LSTM, Concatenate, SimpleRNN, Masking, Flatten
from keras import losses
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal
from keras.callbacks import Callback
from keras.models import load_model

class R2ScoreCallback(Callback):
    def __init__(self, X, Y, Xval=None, Yval=None):
        super(R2ScoreCallback, self).__init__()
        self.X = X
        self.Y = Y
        self.Xval = Xval
        self.Yval = Yval

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict(self.X)
        r2_train = r2_score(self.Y, y_pred_train)
        logs['accuracy'] = r2_train

        if self.Xval is not None and self.Yval is not None:
            y_pred_val = self.model.predict(self.Xval)
            r2_val = r2_score(self.Yval, y_pred_val)
            logs['val_accuracy'] = r2_val

class NNMultistepModel():
    
    def __init__(
        self, 
        X, 
        Y, 
        n_outputs,
        n_lag,
        n_ft,
        n_layer,
        batch,
        epochs, 
        lr,
        Xval=None,
        Yval=None,
        mask_value=-999.0,
        min_delta=0.001,
        patience=5
    ):

        # Series signal 
        lstm_input = InputLayer(input_shape=(n_lag, n_ft)).output
        
        lstm_layer = LSTM(n_layer, activation='tanh')(lstm_input)
        x = Dense(n_outputs)(lstm_layer)
        
        self.model = Model(inputs=lstm_input, outputs=x)
        self.batch = batch 
        self.epochs = epochs
        self.n_layer=n_layer
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
        # Getting the untrained model 
        empty_model = self.model
        
        # Initiating the optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # Compiling the model
        empty_model.compile(loss=losses.MeanAbsoluteError(), optimizer=optimizer)

        if (self.Xval is not None) & (self.Yval is not None):
            history = empty_model.fit(
                self.X, 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch, 
                validation_data=(self.Xval, self.Yval), 
                shuffle=False,
                callbacks=[self.trainCallback(), R2ScoreCallback(self.X, self.Y, self.Xval, self.Yval)]
            )
        else:
            history = empty_model.fit(
                self.X, 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch,
                shuffle=False,
                callbacks=[self.trainCallback(), R2ScoreCallback(self.X, self.Y, self.Xval, self.Yval)]
            )
        
        # Saving to original model attribute in the class
        self.model = empty_model
        # self.save_model("model.h5")
        
        # Returning the training history
        return history
    
    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        return load_model(filename, compile=False)
        