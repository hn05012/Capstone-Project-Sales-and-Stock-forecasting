
from distutils.log import error
import imp
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from multi_lstm import evaluate_accuracy

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
    
    title = 'loss' + '_' + name + '_' + 'epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps)
    filename = path + '/' + title + '.png'
    
    if not os.path.exists(filename):

        plt.figure(figsize = (10, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['Train loss', 'Validation loss'], loc='upper right')
        plt.title(title)
        plt.savefig(filename)
        # plt.show()



def plot_fit(fit, y_train, name, path, epochs, neurons, timesteps):
    errors = fit - y_train
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)

    plt.figure(figsize=(10, 6))
    range_future = len(fit)
    plt.plot(np.arange(range_future), np.array(y_train), 
             label='Training Data')     
    plt.plot(np.arange(range_future),np.array(fit),
            label='Model Fit ' + name)
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    title = 'model_fitting' + '_' + name + '_' + 'epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps) + '_' + 'rmse' + str(rmse)
    filename = path + '/' + title + '.png'
    plt.savefig(filename)
    # plt.show()



def plot_fit_Stocks(training_interval, target_brands, fit, y_train, name, path, epochs, neurons, timesteps):

    t_b_trained = [x + ' train' for x in target_brands]   
    t_b_actual = [x + ' actual' for x in target_brands]

    df_fit = pd.DataFrame(fit, columns=t_b_trained)    
    df_fit['Date'] = training_interval[-df_fit.shape[0]:]
    df_fit.set_index('Date',inplace=True)
    
    for col in df_fit.columns:
        df_fit[col][df_fit[col] < 0] = 0
    
    

    df_actual = pd.DataFrame(y_train,columns=t_b_actual)
    df_actual['Date'] = training_interval[-df_fit.shape[0]:]
    df_actual.set_index('Date',inplace=True) 

    df = pd.concat([df_actual, df_fit], axis=1)
    for c in df.columns:
        if 'actual' in c:

            trained = c.split()[0] + ' ' + c.split()[1] + ' train'     
            errors = df[trained].to_numpy() - df[c].to_numpy()
            mse = np.square(errors).mean()
            rmse = np.sqrt(mse)
            
            title = 'model_fitting' + '_' + name + '_' + c.split()[0] + '_' + c.split()[1] + '_epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps) + '_' + 'rmse=' + str(rmse)
            filename = path + '/' + title + '.png'
            

            if not os.path.exists(filename):

                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df[c], 
                        label='Training Data')
                plt.plot(df.index,df[trained],
                        label='Model Fit ' + name)
                plt.xticks(rotation = 'vertical')
                plt.legend(loc='upper left')
                plt.xlabel('Time')
                plt.ylabel('Stocks')
                plt.title(title)
                plt.savefig(filename)
                # plt.show()

def prediction(model, x_test, scaler_y):
    prediction = model.predict(x_test)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction



def plot_future(prediction, y_test, name, path, epochs, neurons, timesteps):
    errors = prediction - y_test
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    r_sqr = evaluate_accuracy(prediction, y_test)


    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), 
             label='Test Data')     
    plt.plot(np.arange(range_future),np.array(prediction),
            label='Prediction ' + name)
    plt.legend(loc='upper left')
    plt.xlabel('Time (week)')
    plt.ylabel('Sales')
    title = 'forecast' + '_' + name + '_' + 'epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps) + '_' + 'rmse=' + str(rmse) + '_' + 'r_sqr=' + str(r_sqr)
    filename = path + '/' + title + '.png'
    plt.savefig(filename)
    # plt.show()



def plot_future_Stocks(forecasting_interval, target_brands, prediction, y_test, name, path, epochs, neurons, timesteps):
    
    t_b_forecast = [x + ' forecast' for x in target_brands]   
    t_b_actual = [x + ' actual' for x in target_brands]

    df_forecast = pd.DataFrame(prediction, columns=t_b_forecast)    
    df_forecast['Date'] = forecasting_interval[-df_forecast.shape[0]:]
    df_forecast.set_index('Date',inplace=True)
    for col in df_forecast.columns:
        df_forecast[col][df_forecast[col] < 0] = 0
    
    df_actual = pd.DataFrame(y_test, columns=t_b_actual)    
    df_actual['Date'] = forecasting_interval[-df_forecast.shape[0]:]
    df_actual.set_index('Date',inplace=True)
    
    for col in df_forecast.columns:
        df_forecast[col][df_forecast[col] < 0] = 0
    
    df = pd.concat([df_actual, df_forecast], axis=1)
    for c in df.columns:
        if 'actual' in c:

            frcast = c.split()[0] + ' ' + c.split()[1] + ' forecast'
            errors = df[frcast].to_numpy() - df[c].to_numpy()
            mse = np.square(errors).mean()
            rmse = np.sqrt(mse)

            title = 'forecast' + '_' + name + '_' + c.split()[0] + '_' + c.split()[1] + '_epochs=' + str(epochs) + '_' + 'neurons=' + str(neurons) + '_' + 'timesteps=' + str(timesteps) + '_' + 'rmse=' + str(rmse)
            filename = path + '/' + title + '.png'

            if not os.path.exists(filename):
        
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df[c], 
                        label='Actual Data')
                plt.plot(df.index,df[frcast],
                        label='Prediction/ forecast ' + name)
                plt.xticks(rotation = 'vertical')
                plt.legend(loc='upper left')
                plt.xlabel('Time')
                plt.ylabel('Stocks')
                plt.title(title)
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



def flatten_nd_list(l:list):
    result = []
    for sublist in l:
        for item in sublist:
            result.append(item)
    return result
