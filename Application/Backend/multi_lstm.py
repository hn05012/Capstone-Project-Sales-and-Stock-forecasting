from cv2 import rotate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import os



class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.001):
            self.model.stop_training = True



def create_df(file):
    df = pd.read_csv(file,)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    return df



def segment_df(df, start_date: str, end_date: str):        
    data = df.loc[start_date:end_date]
    return data    




def resample_data(df, size: str):               # size will will specify weekly or monthly
    data = df.resample(size).agg({'Sales':'sum', 'Dollar to Pkr':'mean', 'Daily Cases':'sum'})
    return data



def moving_average(df, window_size, column_name, new_column_name):
    data = df
    data[new_column_name] = data[column_name].rolling(window=window_size).mean()
    return data



def view_df(df, head: bool=False, tail: bool=False):
    if head == False and tail == False:
        print(df)
    elif head == True:
        print(df.head(10))
    else:
        print(df.tail(10))



def line_plot(df, columns: list, color):
    for c in columns:
        plt.plot(df.index, df[c], color = color)
        plt.scatter(df.index, df[c], color = color)
        plt.show()



def replace_missing(df):
    data = df
    for column in data:
        data[column].interpolate(inplace = True)
    return data



def train_test_split(df, split):
    train_size = int(len(df)*split)
    train_dataset, test_dataset = df.iloc[:train_size],df.iloc[train_size:]
    return train_dataset, test_dataset



def create_X_Y_train(training_df, testing_df, target_variable):
    X_train = training_df.drop(target_variable, axis = 1)
    y_train = training_df.loc[:,target_variable]
    
    X_test = testing_df.drop(target_variable, axis = 1)
    y_test = testing_df.loc[:,target_variable]

    return X_train, y_train, X_test, y_test



def scale_and_transform(x_train, y_train, x_test, y_test):
    scaler_x = MinMaxScaler(feature_range = (0,1))
    scaler_y = MinMaxScaler(feature_range = (0,1))
    input_scaler = scaler_x.fit(x_train)
    output_scaler = scaler_y.fit(y_train)
    train_y_norm = output_scaler.transform(y_train)
    train_x_norm = input_scaler.transform(x_train)
    test_y_norm = output_scaler.transform(y_test)
    test_x_norm = input_scaler.transform(x_test)

    return scaler_x, scaler_y, train_x_norm, train_y_norm, test_x_norm, test_y_norm



def create_dataset (x_norm, y_norm, time_steps = 1):
    Xs, ys = [], []
    for i in range(len(x_norm)-time_steps):
        v = x_norm[i:i+time_steps, :]
        Xs.append(v)
        ys.append(y_norm[i+time_steps])
    return np.array(Xs), np.array(ys)



def create_model_bilstm(module, units, loss, x_train, target_size):
    model = Sequential()

    if module == 'Stocks':
        model.add(Bidirectional(LSTM(units = units, activation='relu',                             
                return_sequences=True),
                input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Bidirectional(LSTM(units = units, activation='relu',return_sequences = True)))
        model.add(Bidirectional(LSTM(units = units, activation='relu',return_sequences = True)))
        model.add(Bidirectional(LSTM(units = units, activation='relu',return_sequences = True)))
        model.add(Bidirectional(LSTM(units = units, activation='relu',)))
        model.add(Dense(target_size, activation='relu',))
    
    elif module == 'Sales':
        model.add(Bidirectional(LSTM(units = units,                             
              return_sequences=True),
              input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Bidirectional(LSTM(units = units, return_sequences = True)))
        model.add(Bidirectional(LSTM(units = units, return_sequences = True)))
        model.add(Bidirectional(LSTM(units = units)))
        model.add(Dense(target_size))

    model.compile(loss=loss, optimizer='adam')
    return model



def create_model(module, model_name, units, loss, x_train, target_size):
    model = Sequential()
    
    if module == 'Stocks':
        model.add(model_name (units = units, activation='relu', return_sequences = True,
                    input_shape = [x_train.shape[1], x_train.shape[2]]))
        model.add(Dropout(0.2))
        model.add(model_name (units = units, activation='relu', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(model_name (units = units, activation='relu', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(model_name (units = units, activation='relu', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(model_name (units = units, activation='relu', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(model_name (units = units, activation='relu',))
        model.add(Dropout(0.2))
        model.add(Dense(units = target_size, activation='relu',))

    elif module == 'Sales':
        model.add(model_name (units = units, return_sequences = True,
                input_shape = [x_train.shape[1], x_train.shape[2]]))
        model.add(Dropout(0.2))
        model.add(model_name (units = units, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(model_name (units = units, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(model_name (units = units))
        model.add(Dropout(0.2))
        model.add(Dense(units = target_size))

    model.compile(loss=loss, optimizer='adam')
    return model



def fit_model(module, model, epochs, x_train, y_train):
    
    if module == 'Stocks':
        trainingStopCallback = haltCallback()
        history = model.fit(x_train, y_train, epochs = epochs,  
                            validation_split = 0.2, batch_size = 32, 
                            shuffle = False
                            , callbacks = trainingStopCallback
                            )
    else:
        history = model.fit(x_train, y_train, epochs = epochs,  
                            validation_split = 0.2, batch_size = 32, 
                            shuffle = False
                            # , callbacks = trainingStopCallback
                            )
    return history


def inverse_transformation(scaler_y, y_train:np.ndarray, y_test:np.ndarray):
    y_test = scaler_y.inverse_transform(y_test)
    y_train = scaler_y.inverse_transform(y_train)
    return y_train, y_test



def model_fitting(model, x_train, scaler_y):

    fitting = model.predict(x_train)
    return scaler_y.inverse_transform(fitting)




