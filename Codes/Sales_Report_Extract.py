from os import X_OK
import pandas as pd

import csv
from csv import writer
from csv import reader
import datetime


def validate(S):

    try:
        datetime.datetime.strptime(S, '%d/%m/%Y')
        return S
    except ValueError:
        return 0


# Open the input_file in read mode and output_file in write mode
with open('Code/Sales_CSV/SR1.csv', 'r') as read_obj, \
        open('Code/Sales_CSV/SalesReport 30th March 2017 to 1st Dec 2021.csv', 'w', newline='') as write_obj:               # FILE TO READ AND NEW FILE NAME TO GIVE OUTPUT
    # Create a csv.reader object from the input file object
    csv_reader = reader(read_obj)
    # Create a csv.writer object from the output file object
    csv_writer = writer(write_obj)
    # Read each row of the input csv file as list

    d = 0
    D = []
    Header = 0
    Date = 0
    Last_Date = ''


    for row in csv_reader:
        
        xrow = [word for line in row for word in line.split()]
        
        if row[0] == validate(xrow[0]):
            Date = row[0]

        if ('Amount' in xrow or 'Design' in xrow or 'Quantity' in xrow or 'Rate' in xrow or 'Dis (%)' or 'Discount' in xrow) and Header == 0:
            xrow = ['Customer Date','Inv. #', 'Design Number', 'Quantity', 'Rate', 'Amount', 'Dis (%)', 'Discount Amount', 'Amount after Discount']
            csv_writer.writerow(xrow)
            Header = 1
            
        elif ('Duration' in row or 'Duration :' in row or 'Duration:' in row) and d < 1:
            xrow = [word for line in row for word in line.split(' ')]
            D = ['Duration']
            for i in xrow:
                if i == 'Duration'  or i == '' or i == ':' or i == 'Duration:' or i == 'Duration :':
                    pass
                else:
                    D.append(i)  
                        
            d = 1

        elif d == 1:
            xrow = [word for line in row for word in line.split(' ')]
            for i in xrow:
                if i != '':
                    D.append(i)
            Last_Date = D[-1]
            
            d = 2
            
        
        elif len(xrow) >= 8:
            
            if len(xrow) == 9:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2], xrow[3], xrow[4], xrow[5],xrow[6],xrow[7],xrow[8]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7]]
            elif len(xrow) == 10:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3], xrow[4], xrow[5], xrow[6],xrow[7],xrow[8],xrow[9]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7]]
            elif len(xrow) == 11:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4], xrow[5], xrow[6], xrow[7],xrow[8],xrow[9],xrow[10]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7]]
            elif len(xrow) == 12:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4] + ' ' + xrow[5], xrow[6], xrow[7], xrow[8],xrow[9],xrow[10],xrow[11]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7]]
            elif len(xrow) == 13:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4] + ' ' + xrow[5] + ' ' +xrow[6], xrow[7], xrow[8],xrow[9],xrow[10],xrow[11],xrow[12]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7]]
            elif len(xrow) == 14:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4] + ' ' + xrow[5] + ' ' +xrow[6] + ' ' + xrow[7], xrow[8],xrow[9],xrow[10],xrow[11],xrow[12], xrow[13]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7]]
            else:
                
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7]]   
   
      
        
        if len(xrow) < 7 or xrow[0] == '' or 'End' in xrow or 'Report' in xrow or 'Of' in xrow or 'Sales Report' in xrow or 'CASH' in xrow or 'CASH CUSTOMERS' in xrow or 'Sales' in xrow or 'Duration' in xrow or 'Print' in xrow or 'Duration :' in xrow or 'After Discount' in row or 'Amount After Discount' in row or 'Amount' in row or 'Quantity' in row or 'Rate' in row or ':' in xrow or xrow == ['', '', '', '', '', '', 'Amount', 'After', 'Discount'] or 'Amount' in xrow or 'Discount' in xrow or 'Quantity' in xrow:    
            pass
        elif row[0] == validate(row[0]):
            pass
        else:
            csv_writer.writerow(xrow)


      
        
# f = open('Code/Sales_CSV/SalesReport 30th March 2017 to 1st Dec 2021.csv')
# csv_f = csv.reader(f)
# for row in csv_f:
#     if row[0] == '08/04/2017':
#         break
#     else:
#         print(row)

        


# read_file = pd.read_csv (r'text.txt')
# read_file.to_csv (r'New_Products.csv', index=None)

# df = pd.read_fwf('text.txt',delimiter = ', ',header=None)
# df.to_csv('lllllog.csv')

