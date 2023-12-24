import pandas as pd

# Load the Excel file
# Replace 'your_file.xlsx' with the path to your Excel file
df = pd.read_excel('NOV_15_1.xlsx')

# Process the first column
df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: f"NOV_14_2_{x}.mp4")

# Process the third and fourth columns to the format 00:ab.cd
for col in ['Start', 'End']:
    df[col] = df[col].apply(lambda x: f"00:{x:05.2f}")

# Save the processed DataFrame back to an Excel file
# Replace 'processed_file.xlsx' with the path where you want to save the processed file
df.to_excel('final_nov_15_1.xlsx', index=False)
