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
with open('Code/Periodic_Inv/P1.csv', 'r') as read_obj, \
        open('Code/Periodic_Inv/P1_out.csv', 'w', newline='') as write_obj:               # FILE TO READ AND NEW FILE NAME TO GIVE OUTPUT
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
        # xrow = [word for line in row for word in line.split()]
        xrow = row

        # if xrow[0] == validate(xrow[0]):
        #     Date = xrow[0]
        if isinstance(xrow[0], str) == True and xrow[1] == '' and xrow[2] != '':
            if xrow[0] == 'A5988':
                xrow = [xrow[0],'NONB',xrow[2],xrow[3],xrow[4],xrow[5]]
                xrow = [word for line in xrow for word in line.split()]

        if ('Item Code' in xrow or 'Bar Code Opening' in xrow or 'Purchase' in xrow or 'Sale' in xrow or 'Closing' in xrow or 'Opening' in xrow) and Header == 0:
            xrow = ['Category','Item Code','Item Description', 'Bar Code', 'Opening', 'Purchase', 'Sale', 'Closing']
            csv_writer.writerow(xrow)
            Header = 1


        elif len([word for line in xrow for word in line.split()]) >= 7:
            if len([word for line in xrow for word in line.split()]) == 8:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2], xrow[3], xrow[4], xrow[5],xrow[6],xrow[7]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6]]

            elif len([word for line in xrow for word in line.split()]) == 9:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3], xrow[4], xrow[5], xrow[6],xrow[7],xrow[8]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6]]
            elif len([word for line in xrow for word in line.split()]) == 10:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3]+' '+ xrow[4], xrow[5], xrow[6],xrow[7],xrow[8],xrow[9]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6]]
            elif len([word for line in xrow for word in line.split()]) == 11:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3]+' '+ xrow[4] + ' ' + xrow[5], xrow[6],xrow[7],xrow[8],xrow[9],xrow[10]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6]]
            elif len([word for line in xrow for word in line.split()]) == 12:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3]+' '+ xrow[4] + ' ' + xrow[5] + ' ' + xrow[6],xrow[7],xrow[8],xrow[9],xrow[10],xrow[11]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6]]
            elif len([word for line in xrow for word in line.split()]) == 13:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3]+' '+ xrow[4] + ' ' + xrow[5] + ' ' + xrow[6] + ' ' + xrow[7],xrow[8],xrow[9],xrow[10],xrow[11],xrow[12]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6]]
            elif len([word for line in xrow for word in line.split()]) == 14:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [xrow[0], xrow[1] + ' ' + xrow[2] + ' ' + xrow[3]+' '+ xrow[4] + ' ' + xrow[5] + ' ' + xrow[6] + ' ' + xrow[7] + ' ' + xrow[8],xrow[9],xrow[10],xrow[11],xrow[12],xrow[13]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6]]
            else:
                xrow = [word for line in xrow for word in line.split()]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5],xrow[6]]     


        elif isinstance(xrow[0], str) == True and xrow[1] == '' and xrow[2] == '' and xrow[3] == '':
            Category = xrow[0]

       
            
              
        if  (xrow[0] == Category and xrow[1] == '') or '' == xrow[0] or 'Item Code' in xrow or 'Bar Code Opening' in xrow or 'Purchase' in xrow or 'Sale' in xrow or 'Closing' in xrow or 'Opening' in xrow or 'Duration' in xrow or 'Page' in xrow:    
            pass
        # if  '' == xrow[0] or 'Item Code' in xrow or 'Bar Code Opening' in xrow or 'Purchase' in xrow or 'Sale' in xrow or 'Closing' in xrow or 'Opening' in xrow or 'Duration' in xrow or 'Page' in xrow:    
        #     pass
        # elif row[0] == validate(row[0]):
        #     pass
        else:
            csv_writer.writerow(xrow)


      
        
f = open('Code/Periodic_Inv/P1_out.csv')
csv_f = csv.reader(f)
for row in csv_f:
    print(row)

        


# read_file = pd.read_csv (r'text.txt')
# read_file.to_csv (r'New_Products.csv', index=None)

# df = pd.read_fwf('text.txt',delimiter = ', ',header=None)
# df.to_csv('lllllog.csv')

