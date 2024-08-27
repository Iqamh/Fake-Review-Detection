import pandas as pd
import re

# Read the Excel file
df = pd.read_excel(
    'C:\\Users\\iqbal\\Downloads\\Penting\\TA\\New folder\\MachineLearning\\Label Final.xlsx')

# Remove unnecessary columns
df.drop(columns=['Label Pelabel 1', 'Label Pelabel 2',
        'Label Pelabel 3'], inplace=True)

# Replace 'v' with 1 and empty cells with 0
columns_to_replace = ['Untruthful Opinion Pelabel 1', 'Untruthful Opinion Pelabel 2',
                      'Untruthful Opinion Pelabel 3', 'Non-Reviews Pelabel 1',
                      'Non-Reviews Pelabel 2', 'Non-Reviews Pelabel 3',
                      'Review Length', 'Burstiness',
                      'Maximum Content Similarity']

for col in columns_to_replace:
    df[col] = df[col].apply(lambda x: 1 if str(x).startswith('v') else 0)

# Change 'v(31)', 'v(14)', etc to 1
for col in columns_to_replace:
    df[col] = df[col].apply(
        lambda x: 1 if re.match(r'v\((\d+)\)', str(x)) else x)

# Save the modified DataFrame to CSV
df.to_csv('C:\\Users\\iqbal\\Downloads\\Penting\\TA\\New folder\\MachineLearning\\FinalLabel_2.csv',
          sep=';', index=False)
