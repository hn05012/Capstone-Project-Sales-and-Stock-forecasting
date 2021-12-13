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
with open('Code/Curr_Stock/CS1.csv', 'r') as read_obj, \
        open('Code/Curr_Stock/CS1_out.csv', 'w', newline='') as write_obj:               # FILE TO READ AND NEW FILE NAME TO GIVE OUTPUT
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
        
        if row[5] == 'NON B':
            row[5] = 'PCS'


        
        xrow = [word for line in row for word in line.split()]

        if xrow[0] == 'A5988':
            xrow = [xrow[0],'NON B',xrow[1],xrow[2],'PCS',xrow[3],xrow[4]]


        if len(xrow) >= 7 and (xrow[1] == 'WW' or xrow[1] == 'LACE' or xrow[1] == 'LUXURY'):
            a = xrow[0]
            b = xrow[1]
            x = [xrow[0] + ' ' + xrow[1]]
            for i in xrow:
                if i != a and i != b:
                    x.append(i)
            xrow = x
            print(x)


        if ('Item Code' in xrow or 'Bar Code Qty Sold Rate Amount' in xrow or 'Item Code' in xrow or 'Qty Sold' in xrow or 'Amount' in xrow or 'Rate' in xrow or 'P N L' in xrow or 'Avg' in xrow or 'Current' in xrow) and Header == 0:
            xrow = ['Category','Item Code','Item Description', 'Bar Code', 'Stock In Hand', 'Unit', 'Rate', 'Amount']
            csv_writer.writerow(xrow)
            Header = 1


        elif len(xrow) >= 7:
            
            if len(xrow) == 8:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2], xrow[3], xrow[4], xrow[5],xrow[6],xrow[7]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],'PCS',xrow[5],xrow[6]]
            elif len(xrow) == 9:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3], xrow[4], xrow[5], xrow[6],xrow[7],xrow[8]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],'PCS',xrow[5],xrow[6]]
            elif len(xrow) == 10:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4], xrow[5], xrow[6], xrow[7],xrow[8],xrow[9]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],'PCS',xrow[5],xrow[6]]
            elif len(xrow) == 11:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4] + ' ' + xrow[5], xrow[6], xrow[7], xrow[8],xrow[9],xrow[10]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],'PCS',xrow[5],xrow[6]]
            elif len(xrow) == 12:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4] + ' ' + xrow[5] + ' ' +xrow[6], xrow[7], xrow[8],xrow[9],xrow[10],xrow[11]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],'PCS',xrow[5],xrow[6]]
            elif len(xrow) == 13:
                
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3] + ' ' +xrow[4] + ' ' + xrow[5] + ' ' +xrow[6] + ' ' + xrow[7], xrow[8],xrow[9],xrow[10],xrow[11],xrow[12]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],'PCS',xrow[5],xrow[6]]
            else:
                
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],'PCS',xrow[5],xrow[6]]  


        elif isinstance(row[0], str) == True and row[1] == '' and row[2] == '' and row[3] == '':
            Category = row[0]
            
              
        if  (row[0] == Category and row[1] == '') or xrow[0] == '' or 'Item Code' in xrow or 'Item' in xrow or 'P N L' in xrow or 'Avg' in xrow or 'Amount' in xrow or 'Sold' in xrow or 'Duration' in xrow or 'Page' in xrow or 'Rate' in xrow:    
            pass
        # if  '' == xrow[0] or 'Item Code' in xrow or 'Bar Code Opening' in xrow or 'Purchase' in xrow or 'Sale' in xrow or 'Closing' in xrow or 'Opening' in xrow or 'Duration' in xrow or 'Page' in xrow:    
        #     pass
        # elif row[0] == validate(row[0]):
        #     pass
        else:
            csv_writer.writerow(xrow)


      
        
f = open('Code/Curr_Stock/CS1_out.csv')
csv_f = csv.reader(f)
for row in csv_f:
    if row[0] == 'USP66004 U S POLO':
        break
    else:
        print(row)

        

