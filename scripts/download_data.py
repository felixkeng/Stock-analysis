import yfinance as yf

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.to_csv(f'data/raw/{ticker}_stock_data.csv')  # Save data
    return stock_data

# Example usage
if __name__ == '__main__':
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    apple_stock_data = download_stock_data(ticker, start_date, end_date)
