import pandas as pd
import matplotlib.pyplot as plt

def plot_results(analysis_results):
    # Ensure analysis_results is numeric for plotting
    if not analysis_results.empty and all(analysis_results.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
        plt.figure(figsize=(10, 6))
        plt.plot(results.index, results['Actual'], label='Actual Prices', color='blue')
        plt.plot(results.index, results['Predicted'], label='Predicted Prices', color='orange')
        plt.title('Actual vs Predicted Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
    else:
        print("No numeric data to plot.")
