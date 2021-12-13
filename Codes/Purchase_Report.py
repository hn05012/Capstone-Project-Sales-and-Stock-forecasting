from logging import Filter
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
with open('PurchaseReport-30th-Mar-2017-to-1st-Dec-2021.csv', 'r') as read_obj, \
        open('Code/Purchase_Report/PR2_out.csv', 'w', newline='') as write_obj:               # FILE TO READ AND NEW FILE NAME TO GIVE OUTPUT
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
    FirstDate = ''
    chek = 0

    for row in csv_reader:
        
        xrow = [word for line in row for word in line.split()]
        
        if row != [] and xrow != [] and row[0] != '' and row[0] == validate(xrow[0]) and xrow[0] != FirstDate and xrow[0] != Last_Date:
            Date = xrow[0]

        if ('Amount' in xrow or 'Design' in xrow or 'Quantity' in xrow or 'Rate' in xrow or 'Inv. #' in xrow) and Header == 0:
            xrow = ['Date','Inv. #', 'Design Number', 'Quantity', 'Rate', 'Amount']
            csv_writer.writerow(xrow)
            Header = 1

        elif row != [] and xrow != [] and row[0] != '' and row[0] == validate(xrow[0]) and chek <= 1:

            if chek == 0:
                FirstDate = xrow[0]
                chek = chek + 1
            elif chek == 1:
                Last_Date = xrow[0]
                chek = chek + 1

        elif len(xrow) >= 5:
            
            if len(xrow) == 6:

                xrow = [xrow[0], xrow[1] + ' ' + xrow[2], xrow[3], xrow[4], xrow[5]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4]]

            elif len(xrow) == 7:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3], xrow[4], xrow[5], xrow[6]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4]]
            elif len(xrow) == 8:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' + xrow[4], xrow[5], xrow[6], xrow[7]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4]]
            elif len(xrow) == 9:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' + xrow[4] + ' ' + xrow[5], xrow[6], xrow[7], xrow[8]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4]]
            elif len(xrow) == 10:
               
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' + xrow[4] + ' ' + xrow[5] + ' ' + xrow[6], xrow[7], xrow[8], xrow[9]]
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4]]
            else:
               
                xrow = [Date,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4]]
              
        
        if len(xrow) < 5 or xrow == [] or '' in xrow or xrow == [FirstDate] or xrow == [Last_Date] or 'End' in xrow or 'Software' in xrow or 'Purchase Report' in xrow or 'Purchase' in xrow or 'KAMRAN WAREHOUSE' in xrow or 'Report' in xrow or 'Duration' in xrow or 'Print' in xrow or 'Duration :' in xrow or 'Page' in xrow or 'Amount' in xrow or 'Quantity' in xrow or ':' in xrow or 'Amount' in xrow or 'Discount' in xrow or 'Quantity' in xrow:    
            pass
        else:
            csv_writer.writerow(xrow)


      
        
f = open('PurchaseReport-30th-Mar-2017-to-1st-Dec-2021.csv')
csv_f = csv.reader(f)
for row in csv_f:
    if row == ['17   SH838                                         1                       795             795.00']:
        break
    else:
        print(row)

        


# read_file = pd.read_csv (r'text.txt')
# read_file.to_csv (r'New_Products.csv', index=None)

# df = pd.read_fwf('text.txt',delimiter = ', ',header=None)
# df.to_csv('lllllog.csv')

