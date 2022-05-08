
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

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
