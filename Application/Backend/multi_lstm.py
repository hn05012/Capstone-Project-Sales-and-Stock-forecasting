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



def create_model_bilstm(units, loss, x_train, target_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(units = units,                             
              return_sequences=True),
              input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Bidirectional(LSTM(units = units, return_sequences = True)))
    model.add(Bidirectional(LSTM(units = units, return_sequences = True)))
    model.add(Bidirectional(LSTM(units = units)))
    model.add(Dense(target_size))
    #Compile model
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    logcosh = tf.keras.losses.LogCosh()
    model.compile(loss=loss, optimizer='adam')
    return model



def create_model(model_name, units, loss, x_train, target_size):
    model = Sequential()
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
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    logcosh = tf.keras.losses.LogCosh()
    #Compile model
    model.compile(loss=loss, optimizer='adam')
    return model



def fit_model(model, epochs, x_train, y_train):
    # early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
    #                                            patience = 10)
    history = model.fit(x_train, y_train, epochs = epochs,  
                        validation_split = 0.2, batch_size = 32, 
                        shuffle = False
                        # , callbacks = [early_stop]
                        )
    return history



def plot_loss (history, name, path, epochs, neurons, timesteps):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    title = 'loss' + '_' + name + '_' + 'epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps)
    filename = path + '/' + title + '.png'
    plt.title(title)
    plt.savefig(filename)



def plot_loss_stocks (history, name, path, epochs, neurons, timesteps):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    title = 'loss' + '_' + name + '_' + 'epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps)
    filename = path + '/' + title + '.png'
    plt.title(title)
    plt.savefig(filename)
    # plt.show()



def inverse_transformation(scaler_y, y_train:np.ndarray, y_test:np.ndarray):
    y_test = scaler_y.inverse_transform(y_test)
    y_train = scaler_y.inverse_transform(y_train)
    return y_train, y_test



def model_fitting(model, x_train, scaler_y):

    fitting = model.predict(x_train)
    return scaler_y.inverse_transform(fitting)



def plot_fit(fit, y_train, name, path, epochs, neurons, timesteps):
    plt.figure(figsize=(10, 6))
    range_future = len(fit)
    plt.plot(np.arange(range_future), np.array(y_train), 
             label='Training Data')     
    plt.plot(np.arange(range_future),np.array(fit),
            label='Model Fit ' + name)
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    title = 'model_fitting' + '_' + name + '_' + 'epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps)
    filename = path + '/' + title + '.png'
    plt.savefig(filename)
    # plt.show()



def plot_fit_Stocks(training_interval, target_brands, fit, y_train, name, path, epochs, neurons, timesteps):
    
    t_b_trained = [x + ' train' for x in target_brands]   
    t_b_actual = [x + ' actual' for x in target_brands]

    df_fit = pd.DataFrame(fit, columns=t_b_trained)    
    df_fit['Date'] = training_interval[-df_fit.shape[0]:]
    df_fit.set_index('Date',inplace=True)
    df_fit = df_fit.where(df_fit<0, 0)
    
    df_actual = y_train.copy(deep=True)
    df_actual = df_actual.iloc[-df_fit.shape[0]:]
    df_actual.columns = t_b_actual

    df = pd.concat([df_actual, df_fit], axis=1)
    for c in df.columns:
        if 'actual' in c:
            
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df[c], 
                    label='Training Data')
            trained = c.split()[0] + ' ' + c.split()[1] + ' train'     
            plt.plot(df.index,df[trained],
                    label='Model Fit ' + name)
            plt.xticks(rotation = 'vertical')
            plt.legend(loc='upper left')
            plt.xlabel('Time')
            plt.ylabel('Stocks')
            title = 'model_fitting' + '_' + name + '_' + c.split()[0] + '_' + c.split()[1] + 'epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps)
            filename = path + '/' + title + '.png'
            plt.savefig(filename)


def prediction(model, x_test, scaler_y):
    prediction = model.predict(x_test)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction



def plot_future(prediction, y_test, name, filename, path, epochs, neurons, timesteps):
    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), 
             label='Test Data')     
    plt.plot(np.arange(range_future),np.array(prediction),
            label='Prediction ' + name)
    plt.legend(loc='upper left')
    plt.xlabel('Time (week)')
    plt.ylabel('Sales')
    title = 'forecast' + '_' + name + '_' + 'epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps)
    filename = path + '/' + title + '.png'
    plt.savefig(filename)
    # plt.show()



def plot_future_Stocks(forecasting_interval, target_brands, prediction, y_test, name, path, epochs, neurons, timesteps):
    
    t_b_forecast = [x + ' forecast' for x in target_brands]   
    t_b_actual = [x + ' actual' for x in target_brands]

    df_forecast = pd.DataFrame(prediction, columns=t_b_forecast)    
    df_forecast['Date'] = forecasting_interval[-df_forecast.shape[0]:]
    df_forecast.set_index('Date',inplace=True)
    df_forecast = df_forecast.where(df_forecast<0, 0)
    
    df_actual = y_test.copy(deep=True)
    df_actual = df_actual.iloc[-df_forecast.shape[0]:]
    df_actual.columns = t_b_actual
    
    df = pd.concat([df_actual, df_forecast], axis=1)
    for c in df.columns:
        if 'actual' in c:

            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df[c], 
                    label='Actual Data')
            frcast = c.split()[0] + ' ' + c.split()[1] + ' forecast' 
            plt.plot(df.index,df[frcast],
                    label='Prediction/ forecast ' + name)
            plt.xticks(rotation = 'vertical')
            plt.legend(loc='upper left')
            plt.xlabel('Time')
            plt.ylabel('Stocks')
            title = 'forecast' + '_' + name + '_' + c.split()[0] + '_' + c.split()[1] + 'epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps)
            
            filename = path + '/' + title + '.png'
            plt.savefig(filename)
            



def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('')



def flatten_nd_list(l:list):
    result = []
    for sublist in l:
        for item in sublist:
            result.append(item)
    return result


