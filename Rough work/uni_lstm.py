from datetime import date
import datetime as dt
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.dates as mdates



df = pd.read_csv('2021_daily_sales_data.csv', index_col="Date", parse_dates=True)



train = df.iloc[:180]
test = df.iloc[180:]
scaler = MinMaxScaler()

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


n_input = 15
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(generator,epochs=2000)


last_train_batch = scaled_train[-15:]

last_train_batch = last_train_batch.reshape((1, n_input, n_features))

test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions

x = pd.DataFrame(test)
a = x.values.tolist()
sales = []
prediction = []
for i in a:
    sales.append(int(i[0]))
    prediction.append(int(i[1]))


print(len(sales))
print(len(prediction))






sales_report = open('2021_daily_sales_data.csv')
sales_report_reader = csv.reader(sales_report)
sales_report_reader = list(sales_report_reader)
sales_report.close()

N = 100
y = np.random.rand(N)
now = dt.datetime.now()
then = now + dt.timedelta(days=100)
days = mdates.drange(now,then,dt.timedelta(days=1))

dates = []
for i in range(181, len(sales_report_reader)):
    dates.append(sales_report_reader[i][0])
print(len(dates))
x = [dt.datetime.strptime(d,'%d/%m/%Y').date() for d in dates]


plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
plt.plot(x, sales)
plt.plot(x, prediction)
plt.gcf().autofmt_xdate()
plt.title('Sales Trend 2017')
plt.show()

