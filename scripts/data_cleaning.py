import pandas as pd

def load_data(file_path):
    # Load the CSV without skipping rows
    return pd.read_csv(file_path)

def clean_data(data):
    # Print original data shape and first few rows for debugging
    print("Original Data Shape:", data.shape)
    print("Original Data Preview:\n", data.head())

    # Remove the first two rows (they contain non-data header information)
    data = data.iloc[2:]  # Keep rows starting from the 3rd row

    # Reset the index so 'Date' is included as a column
    data.reset_index(drop=True, inplace=True)

    # Set custom column names directly
    data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Print to verify the new header
    print("Data after setting custom header:\n", data.head())

    # Convert 'Date' to datetime format
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Check if 'Date' is now a column
    if 'Date' not in data.columns:
        raise ValueError("The 'Date' column is missing after cleaning the data.")

    # Set 'Date' as the index
    data.set_index('Date', inplace=True)

    # Select relevant columns (keeping 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume')
    data = data[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]

    # Convert columns to numeric, coercing errors
    data = data.apply(pd.to_numeric, errors='coerce')

    # Check for NaN values and handle them
    if data.isnull().values.any():
        print("Data contains NaN values. Filling NaNs...")
        data.ffill(inplace=True)  # Forward fill to handle NaNs
        print("NaN values handled.")

    # Drop any remaining NaN values
    data.dropna(inplace=True)

    # Check final cleaned data shape and preview
    print("Cleaned Data Shape:", data.shape)
    print("Cleaned Data Preview:\n", data.head())

    return data
