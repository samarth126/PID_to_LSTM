import pandas as pd
import time

# Replace 'your_file.xlsx' with the path to your Excel file
excel_file_path = 'another.xlsx'

while True:
    try:
        # Open the Excel file
        xls = pd.ExcelFile(excel_file_path)

        # Get the last sheet in the Excel file (use -1 as the index)
        last_sheet_name = xls.sheet_names[-1]

        # Read the second column (column index 1, as it's 0-based) of the last sheet
        df = pd.read_excel(excel_file_path, sheet_name=last_sheet_name, usecols=[1])

        # Get the last row value from the second column
        last_row_index = len(df) - 1
        last_row_value = df.iloc[last_row_index, 0]  # 0 for the second column

        print(f"Last Row, Column 2: {last_row_value}")
        
        time.sleep(1)  # Pause for one second before reading the next value

    except pd.errors.EmptyDataError:
        print("No data found in the Excel file.")
        break
    except pd.errors.ParserError:
        print("Error parsing the Excel file.")
        break
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        break

print("Finished reading the Excel file.")