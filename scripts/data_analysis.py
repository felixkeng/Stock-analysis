import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def analyze_data(data):
    # Perform analysis (example)
    summary = data.describe()  # Get summary statistics
    return summary

def analyze_stock_data(df):
    df['Day_Number'] = np.arange(len(df))  # Create a day number column
    X = df[['Day_Number']]
    y = df['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict stock prices
    predicted_prices = model.predict(X_test)

    # Create a DataFrame to hold the predictions alongside the actual prices for comparison
    results = pd.DataFrame({'Actual': y_test.values, 'Predicted': predicted_prices})
    return results
