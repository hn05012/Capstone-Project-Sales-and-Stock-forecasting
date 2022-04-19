import csv


def get_brand_categ_codes(file_name: str):          # yahan item list kyy csv dalgy

    categories = []
    brands = []
    item_codes = []
    blank_spaces = []
    incorrect_barcodes = []

    itemlist = open(file_name)          
    itemlist_reader = csv.reader(itemlist)
    itemlist_reader = list(itemlist_reader)
    itemlist.close()

    for i in range(1, len(itemlist_reader)):

        if len(itemlist_reader[i][2]) != 6:             # checks if len of barcode is not 6
            print("The length of barcode is not 6 in row # " + str(i+1))
            incorrect_barcodes.append((i+1, 3))         # row and column

        if not itemlist_reader[i][2].isdecimal():
            print("The barcode should only contain numeric digits but contains something else as well in row # " + str(i+1))
            incorrect_barcodes.append((i+1, 3))         # row and column

        for j in range(len(itemlist_reader[i])):
            if itemlist_reader[i][j].isspace():
                print("There is a blank space in row # " + str(i + 1) + ", column # " + str(j+1))
                blank_spaces.append((i+1, j+1))              # row and column
        
    if len(blank_spaces) == 0 and len(incorrect_barcodes) == 0:           #if no blank spaces and incorrect barcodes then we will 
                                                                          # start fishing for categories, brands and item codes
        for i in range(1, len(itemlist_reader)):
            categ = itemlist_reader[i][0]
            code = itemlist_reader[i][1]
            brand = itemlist_reader[i][3]
            
            if categ not in categories:
                categories.append(categ)
            if code not in item_codes:
                item_codes.append(code)
            if brand not in brands:
                brands.append(brand)

    info = {
        'categories': categories,
        'brands':brands,
        'item_codes':item_codes,
    }
    return info




def check_Periodic_Inventory(file_name: str, info: dict):           # filename mein Periodic Inventory kyy csv dalegy and info; the dict returned by get_brand_categ_codes
    
    per_invent = open(file_name)          
    per_invent_reader = csv.reader(per_invent)
    per_invent_reader = list(per_invent_reader)
    per_invent.close()

    blank_spaces = []
    incorrect_barcodes = []
    new_categories = []
    new_brands = []
    new_item_codes = []
    

    for i in range(1, len(per_invent_reader)):

        if len(per_invent_reader[i][3]) != 6:           
            print("The length of barcode is not 6 in row # " + str(i+1))
            incorrect_barcodes.append((i+1, 4))          

        if not per_invent_reader[i][3].isdecimal():
            print("The barcode should only contain numeric digits but contains something else as well in row # " + str(i+1))
            incorrect_barcodes.append((i+1, 3))         

        for j in range(len(per_invent_reader[i])):
            if per_invent_reader[i][j].isspace():
                blank_spaces.append((i+1,j+1))              

    if len(blank_spaces) == 0 and len(incorrect_barcodes) == 0:           

        for i in range(1, len(per_invent_reader)):        
            categ = per_invent_reader[i][0]
            code = per_invent_reader[i][1]
            brand = per_invent_reader[i][2]

            if categ not in info['categories']: 
                print("new category identified in Periodic Inventory Report")
                new_categories.append(categ)
            if code not in info['item_codes']:
                print("new code identified in Periodic Inventory Report")
                new_item_codes.append(code)
            if brand not in info['brands']:
                print("new brand identified in Periodic Inventory Report")
                new_brands.append(brand)
    new_info = {
        'new categories':new_categories,
        'new brands': new_brands,
        'new codes': new_item_codes
    }
    
    return new_info


def check_ProfitSheet_Itemwise(file_name: str, info: dict):                 # file_name mein ProfitSheetItemwise dalegy and info; the dict returned by get_brand_categ_codes
    
    profit_sheet = open(file_name)          
    profit_sheet_reader = csv.reader(profit_sheet)
    profit_sheet_reader = list(profit_sheet_reader)
    profit_sheet.close()

    blank_spaces = []
    incorrect_barcodes = []
    new_categories = []
    new_brands = []
    new_item_codes = []
    

    for i in range(1, len(profit_sheet_reader)):

        if len(profit_sheet_reader[i][3]) != 6: 
            print(profit_sheet_reader[i][3])           
            print("The length of barcode is not 6 in row # " + str(i+1))
            incorrect_barcodes.append((i+1, 4))          

        if not profit_sheet_reader[i][3].isdecimal():
            print("The barcode should only contain numeric digits but contains something else as well in row # " + str(i+1))
            incorrect_barcodes.append((i+1, 3))         

        for j in range(len(profit_sheet_reader[i])):
            if profit_sheet_reader[i][j].isspace():
                blank_spaces.append((i+1,j+1))              

    if len(blank_spaces) == 0 and len(incorrect_barcodes) == 0:           
        for i in range(1, len(profit_sheet_reader)):        
            categ = profit_sheet_reader[i][0]
            code = profit_sheet_reader[i][1]
            brand = profit_sheet_reader[i][2]

            if categ not in info['categories']: 
                print("new category identified in ProfitSheet Itemwise Report")
                new_categories.append(categ)
            if code not in info['item_codes']:
                print("new code identified in ProfitSheet Itemwise Report")
                new_item_codes.append(code)
            if brand not in info['brands']:
                print("new brand identified in ProfitSheet Itemwise Report")
                new_brands.append(brand)
    new_info = {
        'new categories':new_categories,
        'new brands': new_brands,
        'new codes': new_item_codes
    }
    
    return new_info


def check_sales_Report(file_name: str, info: dict):
    
    sales_report = open(file_name)          
    sales_report = open(file_name)          
    sales_report_reader = csv.reader(sales_report)
    sales_report_reader = list(sales_report_reader)
    sales_report.close()

    blank_spaces = []
    new_item_codes = []

    for i in range(1, len(sales_report_reader)):
        for j in range(len(sales_report_reader[i])):
            if sales_report_reader[i][j].isspace():
                blank_spaces.append((i+1,j+1))         

    if len(blank_spaces) == 0:
        for i in range(1, len(sales_report_reader)):
            
            code = sales_report_reader[i][2]
            if code not in info['item_codes']:
                print("new code identified in Sales Report")
                new_item_codes.append(code)

    new_info = {
        'new codes': new_item_codes
    }
    return new_info



itemlist_info = get_brand_categ_codes('ItemList.csv')
item_codes = itemlist_info['item_codes']
item_codes_with_spaces = []
for code in item_codes:
    for character in code:
        if character == " ":
            item_codes_with_spaces.append(code)
            break
print(item_codes_with_spaces)





# a = check_Periodic_Inventory('Periodic Inventory Report 30th March 2017 to 1st Dec 2021.csv', itemlist_info)
# b = check_ProfitSheet_Itemwise('ProfitSheetItemWise 30th March 2017 to 1st Dec 2021 (2).csv', itemlist_info)
# print(a['new categories'])
# print(a['new brands'])
# print(a['new codes'])