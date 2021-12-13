import tabula


df = tabula.read_pdf("Lahore Branch Data (E4)/PDF Reports/Current Stock/CurrStock 5 dec 2021.pdf")
tabula.convert_into(
    "Lahore Branch Data (E4)/PDF Reports/Current Stock/CurrStock 5 dec 2021.pdf", "Code/Curr_Stock/CS1.csv", pages='all')



