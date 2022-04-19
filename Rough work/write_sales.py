

import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
# from testcases import get_brand_categ_codes

# ************************************
# sales = open('sales.csv', 'r+')
# sales.truncate(0)
# sales.close()
# ************************************


# code to get sales data in required form

# ************************************
# output = open('output_1 (1).csv')
# output_reader = csv.reader(output)
# with open('sales.csv', 'w', newline='') as sales:
#     sales_writer = csv.writer(sales)
#     output_reader = list(output_reader)
    
#     for i in range(len(output_reader)):
#         if output_reader[i][0] != '':
#             sales_writer.writerow(output_reader[i])
# ************************************


# code to check empty cells

# ************************************
# with open('sales.csv') as sales:
#     sales_reader = list(csv.reader(sales))
#     for i in range(len(sales_reader)):
#         for j in range(len(sales_reader[i])):
#             if sales_reader[i][j] == '':
#                 print("There is an empty cell on line number" + str((i + 1)) + ", on cell number " + str(j))
# ************************************




# code written for sales time data

# ************************************
# sales = open('sales_time_data.csv', 'r+')
# sales.truncate(0)
# sales.close()

# output = open('sales.csv')
# output_reader = csv.reader(output)
# sales_time_dict = dict()

# output_reader = list(output_reader)
# output.close()

# for i in range(1, len(output_reader)):
#     if output_reader[i][0] not in sales_time_dict:
#         sales_time_dict[output_reader[i][0]] = float(output_reader[i][8].replace(',', ''))
#     else:
#         sales_time_dict[output_reader[i][0]] += float(output_reader[i][8].replace(',', ''))

# with open('sales_time_data.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Date", "Sales"])
#     for key in sales_time_dict:
#         sales_writer.writerow([key, sales_time_dict[key]])
# sales_time.close()








# sales_2017 = dict()
# sales_2018 = dict()
# sales_2019 = dict()
# sales_2020 = dict()
# sales_2021 = dict()

# sales_report = open('SalesReport 30th March 2017 to 1st Dec 2021.csv')
# sales_report_reader = csv.reader(sales_report)
# sales_report_reader = list(sales_report_reader)
# sales_report.close()

# for i in range(1, len(sales_report_reader)):

#     if sales_report_reader[i][0][-4:] == '2017':
#         date = sales_report_reader[i][0]
        
#         sale = sales_report_reader[i][8]
#         comma_count = 0
#         for x in sale:
#             if x == ',':
#                 comma_count +=1
#         sale = sale.replace(',', '', comma_count)
        
#         if date not in sales_2017:
#             sales_2017[date] = []
#             sales_2017[date].append(int(sale))
#         else:
#             sales_2017[date].append(int(sale))

#     elif sales_report_reader[i][0][-4:] == '2018':
#         date = sales_report_reader[i][0]
#         sale = sales_report_reader[i][8]

#         comma_count = 0
#         for x in sale:
#             if x == ',':
#                 comma_count +=1
#         sale = sale.replace(',', '', comma_count)

#         if date not in sales_2018:
#             sales_2018[date] = []
#             sales_2018[date].append(int(sale))
#         else:
#             sales_2018[date].append(int(sale))

#     elif sales_report_reader[i][0][-4:] == '2019':
#         date = sales_report_reader[i][0]
        
#         sale = sales_report_reader[i][8]
        
#         comma_count = 0
#         for x in sale:
#             if x == ',':
#                 comma_count +=1
#         sale = sale.replace(',', '', comma_count)
        
#         if date not in sales_2019:
#             sales_2019[date] = []
#             sales_2019[date].append(int(sale))
#         else:
#             sales_2019[date].append(int(sale))

#     elif sales_report_reader[i][0][-4:] == '2020':
#         date = sales_report_reader[i][0]

#         sale = sales_report_reader[i][8]
        
#         comma_count = 0
#         for x in sale:
#             if x == ',':
#                 comma_count +=1
#         sale = sale.replace(',', '', comma_count)
        
#         if date not in sales_2020:
#             sales_2020[date] = []
#             sales_2020[date].append(int(sale))
#         else:
#             sales_2020[date].append(int(sale))

#     elif sales_report_reader[i][0][-4:] == '2021': 
#         date = sales_report_reader[i][0]

#         sale = sales_report_reader[i][8]
        
#         comma_count = 0
#         for x in sale:
#             if x == ',':
#                 comma_count +=1
#         sale = sale.replace(',', '', comma_count)
        
#         if date not in sales_2021:
#             sales_2021[date] = []
#             sales_2021[date].append(int(sale))
#         else:
#             sales_2021[date].append(int(sale))

# for date in sales_2017:
#     sales_2017[date] = sum(sales_2017[date])


# for date in sales_2018:
#     sales_2018[date] = sum(sales_2018[date])

# for date in sales_2019:
#     sales_2019[date] = sum(sales_2019[date])

# for date in sales_2020:
#     sales_2020[date] = sum(sales_2020[date])

# for date in sales_2021:
#     sales_2021[date] = sum(sales_2021[date])

# with open('sales_2017.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Date", "Sales"])
#     for date in sales_2017:
#         sales_writer.writerow([date, sales_2017[date]])
# sales_time.close()

# with open('sales_2018.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Date", "Sales"])
#     for date in sales_2018:
#         sales_writer.writerow([date, sales_2018[date]])
# sales_time.close()

# with open('sales_2019.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Date", "Sales"])
#     for date in sales_2019:
#         sales_writer.writerow([date, sales_2019[date]])
# sales_time.close()

# with open('sales_2020.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Date", "Sales"])
#     for date in sales_2020:
#         sales_writer.writerow([date, sales_2020[date]])
# sales_time.close()

# with open('sales_2021.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Date", "Sales"])
#     for date in sales_2021:
#         sales_writer.writerow([date, sales_2021[date]])
# sales_time.close()




# np.random.seed(1)

# N = 100
# y = np.random.rand(N)
# now = dt.datetime.now()
# then = now + dt.timedelta(days=100)
# days = mdates.drange(now,then,dt.timedelta(days=1))

# date_2017 = []
# sales_2017_list = []
# for key in sales_2017:
#     date_2017.append(key)
#     sales_2017_list.append(sales_2017[key])
# x = [dt.datetime.strptime(d,'%d/%m/%Y').date() for d in date_2017]


# # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
# # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
# plt.plot(x, sales_2017_list)
# plt.gcf().autofmt_xdate()
# plt.title('Sales Trend 2017')
# plt.show()



# itemlist_info = get_brand_categ_codes('ItemList.csv')



# sales_report = open('SalesReport 30th March 2017 to 1st Dec 2021.csv')
# sales_report_reader = csv.reader(sales_report)
# sales_report_reader = list(sales_report_reader)
# sales_report.close()

# with open('monthly_sales.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Customer Date", "Month", "Inv. #", "Design Number", "Quantity", "Rate", "Amount", "Dis (%)", "Discount Amount", "Amount after Discount"])
#     for i in range(1, len(sales_report_reader)):
#         record = sales_report_reader[i]
#         date_str = sales_report_reader[i][0]        # string representaion of date
#         date_obj = dt.datetime.strptime(date_str, '%d/%m/%Y').date()       # date_obj
        
#         x = dt.datetime.strptime(str(date_obj.month), "%m")
#         month_name = x.strftime("%B")
#         record.insert(1, month_name)
#         sales_writer.writerow(record)
# sales_time.close()






# monthly_sales_2017 = dict()
# monthly_sales_2018 = dict()
# monthly_sales_2019 = dict()
# monthly_sales_2020 = dict()
# monthly_sales_2021 = dict()


# monthly_sales = open('monthly_sales.csv')
# monthly_sales_reader = csv.reader(monthly_sales)
# monthly_sales_reader = list(monthly_sales_reader)
# monthly_sales.close()

# for i in range(1, len(monthly_sales_reader)):

#     month = monthly_sales_reader[i][1]
#     sale = monthly_sales_reader[i][9]
#     comma_count = 0
#     sale = sale.replace(',', '', sale.count(','))

#     if monthly_sales_reader[i][0][-4:] == '2017':
#         if month not in monthly_sales_2017:
#             monthly_sales_2017[month] = []
#             monthly_sales_2017[month].append(int(sale))
#         else:
#             monthly_sales_2017[month].append(int(sale))

#     elif monthly_sales_reader[i][0][-4:] == '2018':
#         if month not in monthly_sales_2018:
#             monthly_sales_2018[month] = []
#             monthly_sales_2018[month].append(int(sale))
#         else:
#             monthly_sales_2018[month].append(int(sale))

#     elif monthly_sales_reader[i][0][-4:] == '2019':
#         if month not in monthly_sales_2019:
#             monthly_sales_2019[month] = []
#             monthly_sales_2019[month].append(int(sale))
#         else:
#             monthly_sales_2019[month].append(int(sale))

#     elif monthly_sales_reader[i][0][-4:] == '2020':
#         if month not in monthly_sales_2020:
#             monthly_sales_2020[month] = []
#             monthly_sales_2020[month].append(int(sale))
#         else:
#             monthly_sales_2020[month].append(int(sale))

#     elif monthly_sales_reader[i][0][-4:] == '2021': 
#         if month not in monthly_sales_2021:
#             monthly_sales_2021[month] = []
#             monthly_sales_2021[month].append(int(sale))
#         else:
#             monthly_sales_2021[month].append(int(sale))




# for date in monthly_sales_2017:
#     monthly_sales_2017[date] = sum(monthly_sales_2017[date])

# for date in monthly_sales_2018:
#     monthly_sales_2018[date] = sum(monthly_sales_2018[date])

# for date in monthly_sales_2019:
#     monthly_sales_2019[date] = sum(monthly_sales_2019[date])

# for date in monthly_sales_2020:
#     monthly_sales_2020[date] = sum(monthly_sales_2020[date])

# for date in monthly_sales_2021:
#     monthly_sales_2021[date] = sum(monthly_sales_2021[date])

# with open('monthly_sales_2017.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Month", "Sales"])
#     for month in monthly_sales_2017:
#         sales_writer.writerow([month, monthly_sales_2017[month]])
# sales_time.close()

# with open('monthly_sales_2018.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Month", "Sales"])
#     for month in monthly_sales_2018:
#         sales_writer.writerow([month, monthly_sales_2018[month]])
# sales_time.close()

# with open('monthly_sales_2019.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Month", "Sales"])
#     for month in monthly_sales_2019:
#         sales_writer.writerow([month, monthly_sales_2019[month]])
# sales_time.close()

# with open('monthly_sales_2020.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Month", "Sales"])
#     for month in monthly_sales_2020:
#         sales_writer.writerow([month, monthly_sales_2020[month]])
# sales_time.close()

# with open('monthly_sales_2021.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Month", "Sales"])
#     for month in monthly_sales_2021:
#         sales_writer.writerow([month, monthly_sales_2021[month]])
# sales_time.close()




# np.random.seed(1)

# N = 100
# y = np.random.rand(N)
# now = dt.datetime.now()
# then = now + dt.timedelta(days=100)
# days = mdates.drange(now,then,dt.timedelta(days=1))

# month_yr = []
# sales_yr_list = []
# for key in monthly_sales_2017:
#     month_yr.append(key)
#     sales_yr_list.append(monthly_sales_2017[key])

# plt.scatter(month_yr, sales_yr_list)
# plt.plot(month_yr, sales_yr_list)
# plt.title('Monthly Sales Trend 2017')
# plt.show()



# sales = open('SalesReport 30th March 2017 to 1st Dec 2021.csv')
# sales_report_reader = csv.reader(sales)
# sales_report_reader = list(sales_report_reader)
# sales.close()

# daily_sales = dict()

# for i in range(1, len(sales_report_reader)):
#     date = sales_report_reader[i][0]
#     sale = sales_report_reader[i][8]
#     sale = sale.replace(',', '', sale.count(','))
#     if date not in daily_sales:
#         daily_sales[date] = []
#         daily_sales[date].append(int(sale))
#     else:
#         daily_sales[date].append(int(sale))

# for date in daily_sales:
#     daily_sales[date] = sum(daily_sales[date])

# with open('daily_sales.csv', 'w', newline='') as sales_time:
#     sales_writer = csv.writer(sales_time)
#     sales_writer.writerow(["Month", "Sales"])
#     for date in daily_sales:
#         sales_writer.writerow([date, daily_sales[date]])
# sales_time.close()

    
# conversion = open('USD_PKR Historical Data (1).csv')
# conversion_report_reader = csv.reader(conversion)
# conversion_report_reader = list(conversion_report_reader)
# conversion.close()

# conversion_csv = dict()

# for i in range(1, len(conversion_report_reader)):
#     date = conversion_report_reader[i][0]
#     rate = conversion_report_reader[i][1]
#     date = date.replace(',', '')
#     date = date.split(" ")
#     day = date[1]
#     month = date[0]
#     datetime_object = dt.datetime.strptime(month, "%b")
#     month_number = str(datetime_object.month)
#     year = date[2]
#     corr_date = day + '/' + month_number + '/' + year
#     conversion_csv[corr_date] = rate

# with open('conversion_rate.csv', 'w', newline='') as conversion:
#     sales_writer = csv.writer(conversion)
#     sales_writer.writerow(["Date", "Rate"])
#     for date in conversion_csv:
#         sales_writer.writerow([date, conversion_csv[date]])
# conversion.close()

# sales = dict()

# conversion = open('Daily Sales with Pkr conversion.csv')
# conversion_report_reader = csv.reader(conversion)
# conversion_report_reader = list(conversion_report_reader)
# conversion.close()


# rates = []
# for i in range(1, len(conversion_report_reader)):
#     date = conversion_report_reader[i][0]
#     if conversion_report_reader[i][2] == "#N/A":
#         rate = rates[-1]
#         rates.append(rate)
#         sales[date] = []
#         sales[date].append(conversion_report_reader[i][1])
#         sales[date].append(rate)
#     else:
#         rates.append(conversion_report_reader[i][2])
#         sales[date] = []
#         sales[date].append(conversion_report_reader[i][1])
#         sales[date].append(conversion_report_reader[i][2])

# with open('Daily Sales with Pkr conversion.csv', 'w', newline='') as conversion:
#     sales_writer = csv.writer(conversion)
#     sales_writer.writerow(["Date", "Sales", "Dollar to Pkr"])
#     for date in sales:
#         sales_writer.writerow([date, sales[date][0], sales[date][1]])
# conversion.close()





# daily covid cases
# file = open('Daily Sales with Pkr conversion and daily covid cases.csv')
# file_reader = csv.reader(file)
# file_reader = list(file_reader)
# file.close()
# new_df = []
# new_df.append(file_reader[1])
# new_df[0].append(0)

# for i in range(2, len(file_reader) - 1):
#     if file_reader[i][3].isdigit():         # check for NA value 
#         if file_reader[i-1][3].isdigit():       # check if previous value is also int
#             if int(file_reader[i-1][3]) < int(file_reader[i][3]):    # cases increase
#                 cases = int(file_reader[i][3]) - int(file_reader[i-1][3])
#                 record = file_reader[i] + [cases]
#             else:                                                    # no new cases
#                 cases = 0
#                 record = file_reader[i] + [cases]
#         else:                                  # first case
#             cases = int(file_reader[i][3])
#             record = file_reader[i] + [cases]
#     else:
#         cases = 0
#         record = file_reader[i] + [cases]
#     new_df.append(record)

# with open('Daily Sales with Pkr conversion and daily covid cases.csv', 'w', newline='') as x:
#     writer = csv.writer(x)
#     writer.writerow(["Date", "Sales", "Dollar to Pkr", "Total Cases", "Daily Cases"])
#     for i in new_df:
#         # print(i)
#         writer.writerow(i)
# x.close()


