import pandas as pd

# Load the data from the CSV file
data = pd.read_csv(
    'C:\\Users\\iqbal\\Downloads\\Penting\\TA\\New folder\\MachineLearning\\Final_Label_Preprocessed_Balance_Final-Plis.csv', delimiter=";")

# Count the occurrences of "Palsu" and "Asli" in the 'Label Final' column
count_palsu = data['Label Final'].value_counts().get('Palsu', 0)
count_asli = data['Label Final'].value_counts().get('Asli', 0)

# Print the counts
print(f"Count of 'Palsu': {count_palsu}")
print(f"Count of 'Asli': {count_asli}")

# Filter the rows with 'Asli' label
asli_rows = data[data['Label Final'] == 'Asli']

# Randomly select 2 rows to drop
rows_to_drop = asli_rows.sample(n=3900).index

# Drop the selected rows
data_dropped = data.drop(rows_to_drop)

# Save the resulting data to a new CSV file
data_dropped.to_csv(
    'C:\\Users\\iqbal\\Downloads\\Penting\\TA\\New folder\\MachineLearning\\FinalLabel_Balance.csv', index=False, sep=";")

print("2 rows with 'Asli' label have been dropped and the new file has been saved.")
