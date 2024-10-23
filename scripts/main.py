import pandas as pd
import data_analysis
import data_cleaning
def main():
    # Load the data
    file_path = 'data/stock_data.csv'  # Adjust to your actual file path
    raw_data = data_cleaning.load_data(file_path)

    # Clean the data
    cleaned_data = data_cleaning.clean_data(raw_data)

    # Proceed with analysis
    analysis_results = data_analysis.analyze_stock_data(cleaned_data)
    print("Analysis Results:")
    print(analysis_results)

if __name__ == '__main__':
    main()
