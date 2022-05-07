from cgi import print_arguments
from unittest import result
from multi_lstm import *
import os


def main(epochs, time_steps, loss, neurons):

    results = dict()
    directory = 'Data\Monthly_Stocks_Data'
    for filename in os.listdir(directory):

        file = directory + '/' + filename
        
        temp = filename.split()
        fname = temp[0] + '_' + temp[1].split('-')[0]

        data = create_df(file)

        training_df, testing_df = train_test_split(data, 0.8)

        target_brands = []
        for column in data.columns:
            if "brand" in column:
                target_brands.append(column)
        
        x_train, y_train, x_test, y_test = create_X_Y_train(training_df, testing_df, target_brands)
        scaler_x, scaler_y, train_x_norm, train_y_norm, test_x_norm, test_y_norm = scale_and_transform(x_train, y_train, x_test, y_test)

        x_test_3d, y_test_3d = create_dataset(test_x_norm, test_y_norm, time_steps)
        x_train_3d, y_train_3d = create_dataset(train_x_norm, train_y_norm, time_steps)

        model_bilstm = create_model_bilstm(neurons, loss, x_train_3d, len(target_brands))
        model_lstm = create_model(LSTM, neurons, loss, x_train_3d, len(target_brands))
        
        history_bilstm = fit_model(model_bilstm, epochs , x_train_3d, y_train_3d)
        history_lstm = fit_model(model_lstm, epochs, x_train_3d, y_train_3d)

        path = '../Results/Stock Predictions'

        plot_loss_stocks (history_bilstm, fname, 'BILSTM', path, epochs, neurons, time_steps)
        plot_loss_stocks (history_lstm, fname, 'LSTM', path, epochs, neurons, time_steps)

        y_train, y_test = inverse_transformation(scaler_y, y_train_3d, y_test_3d)
        
        bilstm_fit = model_fitting(model_bilstm, x_train_3d, scaler_y)
        lstm_fit = model_fitting(model_lstm, x_train_3d, scaler_y)

        prediction_bilstm = prediction(model_bilstm, x_test_3d, scaler_y)
        prediction_lstm = prediction(model_lstm, x_test_3d, scaler_y)

        df = pd.DataFrame(prediction_lstm, columns=target_brands)
        forecast_dates = testing_df.index.tolist()[-df.shape[0]:]
        df['Date'] = forecast_dates
        df.set_index('Date', inplace=True)
        
        results[fname] = df
    
    return results


stock_predictions = main(epochs=1, time_steps=1, loss = 'huber_loss', neurons=128)