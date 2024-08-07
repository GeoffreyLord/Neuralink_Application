import pandas as pd

def normalize_by_average(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Get the number of columns
    num_columns = df.shape[1]
    
    # Normalize each column by its average, except the last column
    for col in df.columns[:-1]:
        avg = df[col].mean()
        df[col] = df[col] / avg
    
    # Write the normalized data to a new CSV file
    df.to_csv(output_csv, index=False)
    
    print(f"Normalized data has been written to {output_csv}")

# Example usage
input_csv = 'neural_data_eyes_opened_pressed.csv'
output_csv = 'testing_data.csv'
normalize_by_average(input_csv, output_csv)