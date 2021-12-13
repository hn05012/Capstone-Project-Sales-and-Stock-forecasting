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
with open('Code/Item_list/IL1.csv', 'r') as read_obj, \
        open('Code/Item_list/IL1_out.csv', 'w', newline='') as write_obj:               # FILE TO READ AND NEW FILE NAME TO GIVE OUTPUT
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

        if isinstance(row[0], str) == True and row[1] == '' and row[2] == '' and row[3] == '' and row[5] == '':
            Category = row[0]
        xrow = [word for line in row for word in line.split()]

        if len(xrow) > 1 and xrow[1].isnumeric() == False and 'Item' not in xrow and 'ItemCode' not in xrow and Category not in row:
            print(xrow)
            a = xrow[0]
            b = xrow[1]
            x = [xrow[0] + ' ' + xrow[1]]
            for i in xrow:
                if i == a or i == b:
                    pass
                else:
                    x.append(i)
            xrow = x
        
        if xrow[0] == 'AN8531':
            xrow = [xrow[0],xrow[1],xrow[2],'PCS',xrow[3],xrow[4]]

        if xrow[0] == 'A5988':
            xrow = [xrow[0],xrow[1],'NON B','PCS',xrow[2],xrow[3]]
    
     

        if ('ItemCode' in xrow or 'ItemName' in xrow or 'Unit' in xrow or 'P/Rate' in xrow or 'S/Rate' in xrow) and Header == 0:
            xrow = ['Category','ItemCode', 'Barcode', 'ItemName', 'Unit', 'P/Rate', 'S/Rate']
            csv_writer.writerow(xrow)
            Header = 1


        

        elif len(xrow) >= 6:

            if len(xrow) == 7:

                xrow = [xrow[0], xrow[1],xrow[2] +' '+ xrow[3], xrow[4], xrow[5],xrow[6]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],'PCS',xrow[4],xrow[5]]

            elif len(xrow) == 8:
                
                xrow = [xrow[0], xrow[1], xrow[2] + ' ' + xrow[3] +' '+ xrow[4], xrow[5], xrow[6],xrow[7]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],'PCS',xrow[4],xrow[5]]
            elif len(xrow) == 9:
                
                xrow = [xrow[0], xrow[1], xrow[2] + ' ' + xrow[3] +' '+ xrow[4] + ' ' + xrow[5], xrow[6],xrow[7],xrow[8]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],'PCS',xrow[4],xrow[5]]
            elif len(xrow) == 10:
                
                xrow = [xrow[0], xrow[1], xrow[2] + ' ' + xrow[3] +' '+ xrow[4] + ' ' + xrow[5] + ' ' + xrow[6],xrow[7],xrow[8],xrow[9]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],'PCS',xrow[4],xrow[5]]
            elif len(xrow) == 11:
               
                xrow = [xrow[0], xrow[1], xrow[2] + ' ' + xrow[3] +' '+ xrow[4] + ' ' + xrow[5] + ' ' + xrow[6] + ' ' + xrow[7],xrow[8],xrow[9],xrow[10]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],'PCS',xrow[4],xrow[5]]
            elif len(xrow) == 12:
        
                xrow = [xrow[0], xrow[1], xrow[2] + ' ' + xrow[3] +' '+ xrow[4] + ' ' + xrow[5] + ' ' + xrow[6] + ' ' + xrow[7] + ' ' + xrow[8],xrow[9],xrow[10],xrow[11]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],xrow[3],xrow[4],xrow[5]]
            elif len(xrow) == 13:
                
                xrow = [xrow[0], xrow[1], xrow[2] + ' ' + xrow[3] +' '+ xrow[4] + ' ' + xrow[5] + ' ' + xrow[6] + ' ' + xrow[7] + ' ' + xrow[8] + ' ' + xrow[9],xrow[10],xrow[11],xrow[12]]
                xrow = [Category,xrow[0],xrow[1],xrow[2],'PCS',xrow[4],xrow[5]]
            else:
               
                xrow = [Category,xrow[0],xrow[1],xrow[2],'PCS',xrow[4],xrow[5]]     


       
            
              
        if  (Category in row) or '' == xrow[0] or 'Item' in xrow or 'List' in xrow or 'Unit' in xrow or 'P/Rate' in xrow or 'S/Rate' in xrow or 'Duration' in xrow or 'Page' in xrow:    
            pass
        # if  '' == xrow[0] or 'Item Code' in xrow or 'Bar Code Opening' in xrow or 'Purchase' in xrow or 'Sale' in xrow or 'Closing' in xrow or 'Opening' in xrow or 'Duration' in xrow or 'Page' in xrow:    
        #     pass
        # elif row[0] == validate(row[0]):
        #     pass
        else:  

            csv_writer.writerow(xrow)


      
        
f = open('Code/Item_list/IL1_out.csv')
csv_f = csv.reader(f)
for row in csv_f:
    if row[1] == 'MS02023649':
        break
    else:
        print(row)

        


# read_file = pd.read_csv (r'text.txt')
# read_file.to_csv (r'New_Products.csv', index=None)

# df = pd.read_fwf('text.txt',delimiter = ', ',header=None)
# df.to_csv('lllllog.csv')

