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
with open('Code/Profit_sheet_itemwise/PSI1.csv', 'r') as read_obj, \
        open('Code/Profit_sheet_itemwise/PSI1_out.csv', 'w', newline='') as write_obj:               # FILE TO READ AND NEW FILE NAME TO GIVE OUTPUT
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
    Category = ''

    for row in csv_reader:
      
        xrow = row  

        if isinstance(xrow[0], str) == True and xrow[1] == '' and xrow[2] != '':
            if xrow[0] == 'A5988':
                xrow = [xrow[0],'NONB',xrow[2],xrow[3],xrow[4],xrow[5]]
                xrow = [word for line in xrow for word in line.split()]
                

        if ('Item Code' in xrow or 'Bar Code Qty Sold Rate Amount' in xrow or 'Item Code' in xrow or 'Qty Sold' in xrow or 'Amount' in xrow or 'Rate' in xrow or 'P N L' in xrow or 'Avg' in xrow or 'Current' in xrow) and Header == 0:
            xrow = ['Category','Item Code','Item Description', 'Bar Code', 'Qty Sold', 'Avg Rate', 'Total Avg Amount', 'Current Pur Rate', 'Total Current Amount', 'P N L']
            csv_writer.writerow(xrow)
            Header = 1


        elif len([word for line in xrow for word in line.split()]) >= 9:
            if xrow[0] == 'A5988':
                print(xrow)
            if len([word for line in xrow for word in line.split()]) == 10:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2], xrow[3], xrow[4], xrow[5],xrow[6],xrow[7],xrow[8],xrow[9]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7],xrow[8]]
            elif len([word for line in xrow for word in line.split()]) == 11:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3], xrow[4], xrow[5], xrow[6],xrow[7],xrow[8],xrow[9],xrow[10]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7],xrow[8]]
            elif len([word for line in xrow for word in line.split()]) == 12:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4], xrow[5], xrow[6], xrow[7],xrow[8],xrow[9],xrow[10],xrow[11]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7],xrow[8]]
            elif len([word for line in xrow for word in line.split()]) == 13:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4] + ' ' + xrow[5], xrow[6], xrow[7], xrow[8],xrow[9],xrow[10],xrow[11],xrow[12]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7],xrow[8]]
            elif len([word for line in xrow for word in line.split()]) == 14:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4] + ' ' + xrow[5] + ' ' +xrow[6], xrow[7], xrow[8],xrow[9],xrow[10],xrow[11],xrow[12], xrow[13]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7],xrow[8]]
            elif len([word for line in xrow for word in line.split()]) == 15:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4] + ' ' + xrow[5] + ' ' +xrow[6] + ' ' + xrow[7], xrow[8],xrow[9],xrow[10],xrow[11],xrow[12], xrow[13],xrow[14]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7],xrow[8]]
            else:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6],xrow[7],xrow[8]]     


        elif isinstance(xrow[0], str) == True and xrow[1] == '' and xrow[2] == '' and xrow[3] == '':
            Category = xrow[0]
            
              
        if  (xrow[0] == Category and xrow[1] == '') or '' == xrow[0] or 'Item Code' in xrow or 'Item' in xrow or 'P N L' in xrow or 'Avg' in xrow or 'Amount' in xrow or 'Sold' in xrow or 'Duration' in xrow or 'Page' in xrow or 'Rate' in xrow:    
            pass
        # if  '' == xrow[0] or 'Item Code' in xrow or 'Bar Code Opening' in xrow or 'Purchase' in xrow or 'Sale' in xrow or 'Closing' in xrow or 'Opening' in xrow or 'Duration' in xrow or 'Page' in xrow:    
        #     pass
        # elif row[0] == validate(row[0]):
        #     pass
        else:
            csv_writer.writerow(xrow)


      
        
# f = open('Code/Profit_sheet_itemwise/PSI1_out.csv')
# csv_f = csv.reader(f)
# for row in csv_f:
#     print(row)

        


# read_file = pd.read_csv (r'text.txt')
# read_file.to_csv (r'New_Products.csv', index=None)

# df = pd.read_fwf('text.txt',delimiter = ', ',header=None)
# df.to_csv('lllllog.csv')

