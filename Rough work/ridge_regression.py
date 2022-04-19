

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import math
import datetime as dt
from matplotlib import pyplot as plot


# code written for sales.csv

# ******************************************************************
# sales_data = pd.read_csv('sales.csv')
# sales_data['Date'] = pd.to_datetime(sales_data['Date'])
# sales_data['Date']=sales_data['Date'].map(dt.datetime.toordinal)
# # sales_data['Amount after Discount'] = pd.to_numeric(sales_data['Amount after Discount'], downcast="float")
# x = []
# for value in sales_data['Amount after Discount'].values:
#     x.append(float(value.replace(',', '')))
# sales_data['Amount after Discount'] = x


# target = ['Amount after Discount']
# predictor = ['Date']

# X = sales_data[predictor].values
# Y = sales_data[target].values
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=40)


# model = Ridge(alpha=0.01)
# model.fit(X_train, y_train)
# d = dt.datetime(2020, 2, 9)
# d = d.toordinal()
# y_predict = model.predict([[d]])
# print('Predicted: %.3f' % y_predict)
# ******************************************************************



# code for sales_time_data.csv

# ******************************************************************
# sales_time = pd.read_csv('sales_time_data.csv')
# sales_time['Date'] = pd.to_datetime(sales_time['Date'])
# sales_time['Date']=sales_time['Date'].map(dt.datetime.toordinal)

# target = ['Sales']
# predictor = ['Date']

# X = sales_time[predictor].values
# Y = sales_time[target].values
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=40)
# model = Ridge(alpha=1)
# model.fit(X_train, y_train)

# d = dt.datetime(2020, 2, 9)
# d = d.toordinal()
# y_predict = model.predict([[d]])
# print('Predicted: %.3f' % y_predict)

# plot.scatter(X_train, y_train, color = 'red')
# plot.plot(X_train, model.predict(X_train))
# plot.title('Sales against time')
# plot.xlabel('Date')
# plot.ylabel('Sales')
# plot.show()

# plot.scatter(X_test, y_test, color = 'red')
# plot.plot(X_train, model.predict(X_train), color = 'blue')
# plot.title('Sales against time')
# plot.xlabel('Date')
# plot.ylabel('Sales')
# plot.show()
# ******************************************************************




# baseline NN implementation for sales_time data

# ******************************************************************

sales_time = pd.read_csv('sales_time_data.csv')
