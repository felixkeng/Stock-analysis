import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# Function to download data with retry logic
def download_data_with_retry(tickers, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start=start_date, end=end_date, interval='1d', auto_adjust=False)
            if len(tickers) > 1:
                data = data['Adj Close']
            if 'QQQ' in data.columns:
                data = data.rename(columns={'QQQ': 'NASDAQ'})
            existing_columns = [col for col in tickers if col in data.columns]
            data = data[existing_columns].dropna()
            return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {tickers}: {str(e)}")
            if attempt < max_retries - 1:
                sleep(5)  # Wait 5 seconds before retrying
            else:
                print(f"Max retries reached for {tickers}. Skipping unavailable tickers.")
                return pd.DataFrame()
    return pd.DataFrame()

# Define assets and parameters
index_funds = ['SPY', 'QQQ', 'XLK', 'SMH', 'USD']
tickers_scen1 = index_funds + ['NVDA', 'AAPL', 'TSM']  # Original 8
index_names_scen1 = index_funds + ['NVDA', 'AAPL', 'TSM']
tickers_scen2 = ['AAPL', 'MSFT', 'META', 'NVDA', 'TSLA', 'UNH', 'AMD']  # Swapped BRK.B with UNH, 7 stocks
index_names_scen2 = ['AAPL', 'MSFT', 'META', 'NVDA', 'TSLA', 'UNH', 'AMD']
initial_investment = 100000
daily_investment_total = 100  # $100 total daily investment
start_date = '2005-07-21'
end_date = '2025-07-25'  # Updated to today, 10:18 PM +08

# Download data for all scenarios with retry
data_scen1 = download_data_with_retry(tickers_scen1, start_date, end_date)
data_scen2 = download_data_with_retry(tickers_scen2, start_date, end_date)

# Check if 'SPY' is available in at least one dataset to proceed
if not any('SPY' in df.columns for df in [data_scen1, data_scen2] if not df.empty):
    raise ValueError("SPY data is not available in any scenario. Please check your internet connection or ticker list.")

# Simulate buy-and-hold for a single asset
def simulate_buy_hold(data, asset, daily_investment, initial_investment):
    if asset not in data.columns or data.empty:
        return pd.DataFrame(columns=['value', asset, 'cash'], index=[pd.Timestamp(start_date)])
    portfolio = pd.DataFrame(index=data.index, columns=['value', asset, 'cash'])
    portfolio.iloc[0] = [initial_investment, 0, initial_investment]
    portfolio[f'{asset}_shares'] = 0.0
    initial_price = data[asset].iloc[0]
    shares = initial_investment / initial_price
    portfolio.loc[portfolio.index[0], f'{asset}_shares'] = shares
    portfolio.loc[portfolio.index[0], asset] = portfolio[f'{asset}_shares'].iloc[0] * data[asset].iloc[0]
    
    for i in range(1, len(data)):
        current_value = portfolio['cash'].iloc[i-1]
        daily_investment_per_asset = daily_investment
        shares_to_buy = (daily_investment_per_asset / data[asset].iloc[i]) * (1 - 0.001)
        portfolio.loc[portfolio.index[i], f'{asset}_shares'] = portfolio[f'{asset}_shares'].iloc[i-1] + shares_to_buy
        portfolio.loc[portfolio.index[i], asset] = portfolio[f'{asset}_shares'].iloc[i] * data[asset].iloc[i]
        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - daily_investment_per_asset
        portfolio.loc[portfolio.index[i], 'value'] = portfolio[asset].iloc[i] + portfolio['cash'].iloc[i]
    return portfolio

# Simulate strategy with 2.6% drop threshold and 100% sell/redistribute all
def simulate_strategy_2_6_100(data, indices, initial_investment, daily_investment_total):
    if data.empty:
        return pd.DataFrame(columns=['value'] + indices + ['cash'] + [f"{idx}_shares" for idx in indices], index=[pd.Timestamp(start_date)])
    num_indices = len([idx for idx in indices if idx in data.columns])
    if num_indices == 0:
        return pd.DataFrame(columns=['value'] + indices + ['cash'] + [f"{idx}_shares" for idx in indices], index=[pd.Timestamp(start_date)])
    initial_investment_per_index = initial_investment / num_indices
    daily_investment_per_index = daily_investment_total / num_indices
    all_columns = ['value'] + indices + ['cash'] + [f"{idx}_shares" for idx in indices]
    portfolio = pd.DataFrame(index=data.index, columns=all_columns)
    portfolio.iloc[0] = [initial_investment] + [0] * len(indices) + [initial_investment] + [0] * len(indices)
    for idx in indices:
        if idx in data.columns:
            portfolio.loc[portfolio.index[0], f"{idx}_shares"] = float(initial_investment_per_index / data[idx].iloc[0])
            portfolio.loc[portfolio.index[0], idx] = portfolio.loc[portfolio.index[0], f"{idx}_shares"] * data[idx].iloc[0]
    initial_values = [portfolio.loc[portfolio.index[0], idx] for idx in indices if idx in data.columns]
    total_initial_value = sum(initial_values) if initial_values else 0
    if abs(total_initial_value - initial_investment) > 0.01:
        cash_adjustment = initial_investment - total_initial_value
        portfolio.loc[portfolio.index[0], 'cash'] += cash_adjustment
    
    for i in range(1, len(data)):
        current_value = portfolio['cash'].iloc[i-1]
        for idx in indices:
            if f"{idx}_shares" in portfolio.columns and idx in data.columns:
                current_value += portfolio[f"{idx}_shares"].iloc[i-1] * data[idx].iloc[i]
        portfolio.loc[portfolio.index[i], 'value'] = current_value
        
        remaining_investment = daily_investment_total
        for idx in indices:
            if idx in data.columns and remaining_investment > 0:
                shares_to_buy = (min(remaining_investment / num_indices, daily_investment_per_index) / data[idx].iloc[i]) * (1 - 0.001)
                portfolio.loc[portfolio.index[i], f"{idx}_shares"] = portfolio[f"{idx}_shares"].iloc[i-1] + shares_to_buy
                portfolio.loc[portfolio.index[i], idx] = portfolio[f"{idx}_shares"].iloc[i] * data[idx].iloc[i]
                remaining_investment -= (shares_to_buy * data[idx].iloc[i]) / (1 - 0.001)
        
        dropped_indices = []
        for idx in indices:
            if idx in data.columns and i > 0:
                daily_return = (data[idx].iloc[i] - data[idx].iloc[i-1]) / data[idx].iloc[i-1]
                if daily_return <= -0.026:
                    dropped_indices.append(idx)
        
        if dropped_indices:  # If any or all stocks drop
            total_redistribution = 0
            # Sell 100% of all assets (dropping and non-dropping)
            for idx in indices:
                if idx in data.columns:
                    value_to_sell = portfolio[idx].iloc[i-1] * 1.0
                    shares_to_sell = (value_to_sell / data[idx].iloc[i]) * (1 - 0.001)
                    if portfolio[f"{idx}_shares"].iloc[i] >= shares_to_sell:
                        portfolio.loc[portfolio.index[i], f"{idx}_shares"] -= shares_to_sell
                        portfolio.loc[portfolio.index[i], idx] = 0  # Reset value to 0 after selling
                        total_redistribution += value_to_sell * (1 - 0.001)
            
            if total_redistribution > 0:
                cash_per_drop = total_redistribution / len(dropped_indices) if dropped_indices else total_redistribution / len(indices)
                for idx in dropped_indices:  # Only redistribute to dropping stocks
                    if idx in data.columns:
                        shares_to_buy = (cash_per_drop / data[idx].iloc[i]) * (1 - 0.001)
                        portfolio.loc[portfolio.index[i], f"{idx}_shares"] += shares_to_buy
                        portfolio.loc[portfolio.index[i], idx] = portfolio[f"{idx}_shares"].iloc[i] * data[idx].iloc[i]
        
        for idx in indices:
            if idx in data.columns and idx not in dropped_indices:
                portfolio.loc[portfolio.index[i], idx] = portfolio[f"{idx}_shares"].iloc[i] * data[idx].iloc[i]
        portfolio.loc[portfolio.index[i], 'cash'] = max(0, portfolio['cash'].iloc[i-1] - (daily_investment_total - remaining_investment))
    
    return portfolio

# Run strategies for all scenarios
buy_hold_portfolios_scen1 = {asset: simulate_buy_hold(data_scen1, asset, daily_investment_total, initial_investment) for asset in index_names_scen1 if asset in data_scen1.columns}
strategy_2_6_100_scen1 = simulate_strategy_2_6_100(data_scen1, index_names_scen1, initial_investment, daily_investment_total)

buy_hold_portfolios_scen2 = {asset: simulate_buy_hold(data_scen2, asset, daily_investment_total, initial_investment) for asset in index_names_scen2 if asset in data_scen2.columns}
strategy_2_6_100_scen2 = simulate_strategy_2_6_100(data_scen2, index_names_scen2, initial_investment, daily_investment_total)

# Function to add data labels to plot
def add_data_labels(ax, x, y, label_prefix):
    if len(y) > 0:
        for i in [0, len(x)//4, len(x)//2, 3*len(x)//4, -1]:
            if 0 <= i < len(y):
                ax.annotate(f'{label_prefix} ${y.iloc[i]:,.0f}', (x[i], y.iloc[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)

# Create two plots with available lines and end labels
plt.figure(figsize=(14, 6))

# Plot 1: Original 8 (5 indices + NVDA, AAPL, TSM)
plt.subplot(2, 1, 1)
for asset, portfolio in buy_hold_portfolios_scen1.items():
    if not portfolio.empty:
        plt.plot(portfolio.index, portfolio['value'], label=f'Buy & Hold {asset}', linestyle='-', alpha=0.7)
        if len(portfolio) > 0:
            plt.annotate(f'Buy & Hold {asset}', (portfolio.index[-1], portfolio['value'].iloc[-1]), xytext=(5, 0), textcoords='offset points', fontsize=8)
if not strategy_2_6_100_scen1.empty:
    plt.plot(strategy_2_6_100_scen1.index, strategy_2_6_100_scen1['value'], label='Strategy 2.6% & 100% Sell', color='red', linewidth=2)
    if len(strategy_2_6_100_scen1) > 0:
        plt.annotate('Strategy 2.6% & 100% Sell', (strategy_2_6_100_scen1.index[-1], strategy_2_6_100_scen1['value'].iloc[-1]), xytext=(5, 0), textcoords='offset points', fontsize=8)
plt.title('Buy & Hold vs Strategy 2.6% & 100% Sell (5 Indices + NVDA, AAPL, TSM) (2005–2025)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
add_data_labels(plt.gca(), strategy_2_6_100_scen1.index, strategy_2_6_100_scen1['value'], 'Strat')
for asset, portfolio in buy_hold_portfolios_scen1.items():
    if not portfolio.empty:
        add_data_labels(plt.gca(), portfolio.index, portfolio['value'], f'BH-{asset[:3]}')

# Plot 2: All 8 Stocks (AAPL, MSFT, META, NVDA, TSLA, UNH, AMD)
plt.subplot(2, 1, 2)
for asset, portfolio in buy_hold_portfolios_scen2.items():
    if not portfolio.empty:
        plt.plot(portfolio.index, portfolio['value'], label=f'Buy & Hold {asset}', linestyle='-', alpha=0.7)
        if len(portfolio) > 0:
            plt.annotate(f'Buy & Hold {asset}', (portfolio.index[-1], portfolio['value'].iloc[-1]), xytext=(5, 0), textcoords='offset points', fontsize=8)
if not strategy_2_6_100_scen2.empty:
    plt.plot(strategy_2_6_100_scen2.index, strategy_2_6_100_scen2['value'], label='Strategy 2.6% & 100% Sell', color='red', linewidth=2)
    if len(strategy_2_6_100_scen2) > 0:
        plt.annotate('Strategy 2.6% & 100% Sell', (strategy_2_6_100_scen2.index[-1], strategy_2_6_100_scen2['value'].iloc[-1]), xytext=(5, 0), textcoords='offset points', fontsize=8)
plt.title('Buy & Hold vs Strategy 2.6% & 100% Sell (AAPL, MSFT, META, NVDA, TSLA, UNH, AMD) (2005–2025)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
add_data_labels(plt.gca(), strategy_2_6_100_scen2.index, strategy_2_6_100_scen2['value'], 'Strat')
for asset, portfolio in buy_hold_portfolios_scen2.items():
    if not portfolio.empty:
        add_data_labels(plt.gca(), portfolio.index, portfolio['value'], f'BH-{asset[:3]}')

plt.tight_layout()
plt.show()