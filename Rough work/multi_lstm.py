import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt



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

def plot_loss (history, name):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    plt.title(name)
    filename = 'loss ' + name
    plt.savefig(filename)
    # plt.show()

def inverse_transformation(scaler_y, y_train, y_test):
    y_test = scaler_y.inverse_transform(y_test)
    y_train = scaler_y.inverse_transform(y_train)
    return y_train, y_test

def model_fitting(model, x_train, scaler_y):

    fitting = model.predict(x_train)
    return scaler_y.inverse_transform(fitting)

def plot_fit(fit, y_train, name, column_name, epochs):
    plt.figure(figsize=(10, 6))
    range_future = len(fit)
    plt.plot(np.arange(range_future), np.array(y_train), 
             label='Training Data')     
    plt.plot(np.arange(range_future),np.array(fit),
            label='Model Fit ' + name)
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.title(column_name)
    filename = column_name + ' model fitting ' + str(epochs)
    plt.savefig(filename)
    # plt.show()

def prediction(model, x_test, scaler_y):
    prediction = model.predict(x_test)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction

def plot_future(prediction, y_test, name, column_name, epochs):
    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), 
             label='Test Data')     
    plt.plot(np.arange(range_future),np.array(prediction),
            label='Prediction ' + name)
    plt.legend(loc='upper left')
    plt.xlabel('Time (week)')
    plt.ylabel('Weekly Sales')
    plt.title(column_name)
    filename = column_name + ' prediction ' + str(epochs)
    plt.savefig(filename)
    # plt.show()

def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('')


def main():
    # file = 'Data\multi_feature_sales_data.csv'
    # data = create_df(file)
    # data = segment_df(data, '2017-04-01','2021-12-01')
    # # view_df(data)
    # data = replace_missing(data)
    # training_df, testing_df = train_test_split(data, 0.8)
    
    # x_train, y_train, x_test, y_test = create_X_Y_train(training_df, testing_df, 'Sales')
    
    # scaler_x, scaler_y, train_x_norm, train_y_norm, test_x_norm, test_y_norm = scale_and_transform(x_train, y_train, x_test, y_test)
    
    # TIME_STEPS = 10
    
    # x_test_3d, y_test_3d = create_dataset(test_x_norm, test_y_norm, TIME_STEPS)
    # x_train_3d, y_train_3d = create_dataset(train_x_norm, train_y_norm, TIME_STEPS)
    
    # model_bilstm = create_model_bilstm(128, 'huber_loss', x_train_3d)
    # model_lstm = create_model(LSTM, 128, 'huber_loss', x_train_3d)

    # history_bilstm = fit_model(model_bilstm, 1000   , x_train_3d, y_train_3d)
    # history_lstm = fit_model(model_lstm, 1000, x_train_3d, y_train_3d)

    # plot_loss (history_bilstm, 'BILSTM')
    # plot_loss (history_lstm, 'LSTM')

    # y_train, y_test = inverse_transformation(scaler_y, y_train_3d, y_test_3d)

    # bilstm_fit = model_fitting(model_bilstm, x_train_3d, scaler_y)
    # lstm_fit = model_fitting(model_lstm, x_train_3d, scaler_y)

    # plot_fit(bilstm_fit, y_train, 'BiLSTM')
    # plot_fit(lstm_fit, y_train, 'LSTM')

    # prediction_bilstm = prediction(model_bilstm, x_test_3d, scaler_y)
    # prediction_lstm = prediction(model_lstm, x_test_3d, scaler_y)

    # plot_future(prediction_bilstm, y_test, 'BiLSTM')
    # plot_future(prediction_lstm, y_test, 'LSTM')

    # evaluate_prediction(prediction_bilstm, y_test, 'Bidirectional LSTM')
    # evaluate_prediction(prediction_lstm, y_test, 'LSTM')




    file = 'Data\main_categories_features.csv'
    data = create_df(file)
    data = segment_df(data, '2017-04-01','2021-12-01')
    # view_df(data)
    data = replace_missing(data)
    training_df, testing_df = train_test_split(data, 0.8)
    
    main_cat = ['category r179f','category d230v','category X107H','category o52A','category I108K','category u85c','category a228a','category B163E',
 'category R72O',
 'category C176l',
 'category c27J',
 'category z239x',
 'category Q239q',
 'category Y159X',
 'category e136a',
 'category o148Y',
 'category N189r',
 'category B9q',
 'category M3Y'
 ]

    x_train, y_train, x_test, y_test = create_X_Y_train(training_df, testing_df, main_cat)
    
    scaler_x, scaler_y, train_x_norm, train_y_norm, test_x_norm, test_y_norm = scale_and_transform(x_train, y_train, x_test, y_test)
    
    TIME_STEPS = 7
    
    x_test_3d, y_test_3d = create_dataset(test_x_norm, test_y_norm, TIME_STEPS)
    x_train_3d, y_train_3d = create_dataset(train_x_norm, train_y_norm, TIME_STEPS)
    
    model_bilstm = create_model_bilstm(128, 'huber_loss', x_train_3d, len(main_cat))
    model_lstm = create_model(LSTM, 128, 'huber_loss', x_train_3d, len(main_cat))

    epochs = 500
    
    history_bilstm = fit_model(model_bilstm, epochs , x_train_3d, y_train_3d)
    history_lstm = fit_model(model_lstm, epochs, x_train_3d, y_train_3d)

    plot_loss (history_bilstm, 'BILSTM')
    plot_loss (history_lstm, 'LSTM')

    y_train, y_test = inverse_transformation(scaler_y, y_train_3d, y_test_3d)

    bilstm_fit = model_fitting(model_bilstm, x_train_3d, scaler_y)
    lstm_fit = model_fitting(model_lstm, x_train_3d, scaler_y)

    for i in range(len(main_cat)):
        plot_fit(bilstm_fit[i], y_train[i], 'BiLSTM', main_cat[i], epochs)
        plot_fit(lstm_fit[i], y_train[i], 'LSTM', main_cat[i], epochs)

    prediction_bilstm = prediction(model_bilstm, x_test_3d, scaler_y)
    prediction_lstm = prediction(model_lstm, x_test_3d, scaler_y)

    for i in range(len(main_cat)):
        plot_future(prediction_bilstm[i], y_test[i], 'BiLSTM', main_cat[i], epochs)
        plot_future(prediction_lstm[i], y_test[i], 'LSTM', main_cat[i], epochs)

    evaluate_prediction(prediction_bilstm, y_test, 'Bidirectional LSTM')
    evaluate_prediction(prediction_lstm, y_test, 'LSTM')


main()