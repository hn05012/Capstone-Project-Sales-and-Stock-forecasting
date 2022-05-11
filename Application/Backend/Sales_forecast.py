
from django import conf
from multi_lstm import *
from plotting_helper_functions import*


def main(epochs, time_steps, loss, neurons):
    file = 'Data\Sales\multi_feature_sales_data.csv'
    data = create_df(file)
    data = segment_df(data, '2017-04-01','2021-12-01')
    # view_df(data)

    data = replace_missing(data)
    training_df, testing_df = train_test_split(data, 0.8)

    
    x_train, y_train, x_test, y_test = create_X_Y_train(training_df, testing_df, ['Sales'])

    scaler_x, scaler_y, train_x_norm, train_y_norm, test_x_norm, test_y_norm = scale_and_transform(x_train, y_train, x_test, y_test)
    
    TIME_STEPS = time_steps
    
    x_test_3d, y_test_3d = create_dataset(test_x_norm, test_y_norm, TIME_STEPS)
    x_train_3d, y_train_3d = create_dataset(train_x_norm, train_y_norm, TIME_STEPS)
    
    model_bilstm = create_model_bilstm(neurons, loss, x_train_3d, 1)
    model_lstm = create_model(LSTM, neurons, loss, x_train_3d, 1)

    history_bilstm = fit_model(model_bilstm, epochs , x_train_3d, y_train_3d)
    history_lstm = fit_model(model_lstm, epochs, x_train_3d, y_train_3d)

    path = '../Results/Sales Predictions'

    plot_loss (history_bilstm, 'BILSTM', path, epochs, neurons, time_steps)                 # hassam comment this line to run forecast onli
    plot_loss (history_lstm, 'LSTM', path, epochs, neurons, time_steps)                     # hassam comment this line to run forecast onli

    y_train, y_test = inverse_transformation(scaler_y, y_train_3d, y_test_3d)

    bilstm_fit = model_fitting(model_bilstm, x_train_3d, scaler_y)
    lstm_fit = model_fitting(model_lstm, x_train_3d, scaler_y)

    plot_fit(bilstm_fit, y_train, 'BiLSTM', path, epochs, neurons, time_steps)              # hassam comment this line to run forecast onli
    plot_fit(lstm_fit, y_train, 'LSTM', path, epochs, neurons, time_steps)                  # hassam comment this line to run forecast onli

    prediction_bilstm = prediction(model_bilstm, x_test_3d, scaler_y)
    prediction_lstm = prediction(model_lstm, x_test_3d, scaler_y)

    sales = flatten_nd_list(prediction_lstm.tolist())
    forecasted_dates = testing_df.index.tolist()[-len(sales):]

    plot_future(prediction_bilstm, y_test, 'BiLSTM', path, epochs, neurons, time_steps)     # hassam comment this line to run forecast onli
    plot_future(prediction_lstm, y_test, 'LSTM', path, epochs, neurons, time_steps)         # hassam comment this line to run forecast onli

    # evaluate_prediction(prediction_bilstm, y_test, 'Bidirectional LSTM')
    # evaluate_prediction(prediction_lstm, y_test, 'LSTM')

    return forecasted_dates, sales




config = [(200, 7, 32), (500, 7, 32), (400, 14, 64), (500, 10, 64), 
        (700, 7, 32), (700, 14, 128), (700, 10, 128)
        ]
for c in config:
    f,s = main(epochs=c[0], time_steps=c[1], loss='huber_loss', neurons=c[2])


# plot sales forecast
# plt.figure(figsize = (10, 6))
# plt.plot(f, s)
# plt.ylabel('Sales')
# plt.xlabel('dates')
# plt.title('forecast')
# plt.xticks(rotation = 'vertical')
# plt.show()