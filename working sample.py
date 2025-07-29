import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define assets and parameters
tickers_good = ['NVDA', 'TSM', 'MSFT', 'SPY', 'QQQ', 'SMH', 'XLK', 'USD']
index_names_good = ['NVDA', 'TSM', 'MSFT', 'SPY', 'QQQ', 'SMH', 'XLK', 'USD']
tickers_bad = ['SPY', 'QQQ', 'SMH', 'XLK', 'USD', 'NHTC', 'BKNG', 'ZD']
index_names_bad = ['SPY', 'QQQ', 'SMH', 'XLK', 'USD', 'NHTC', 'BKNG', 'ZD']
tickers_worst = ['SPY', 'QQQ', 'SMH', 'XLK', 'USD', 'NHTC', 'CLAR', 'EGAN']
index_names_worst = ['SPY', 'QQQ', 'SMH', 'XLK', 'USD', 'NHTC', 'CLAR', 'EGAN']
initial_investment = 100000
daily_investment_total = 100  # $100 total daily investment
start_date = '2005-07-21'
end_date = '2025-07-24'  # Updated to today, 06:44 PM +08

# Download data for all scenarios
data_good = yf.download(tickers_good, start=start_date, end=end_date, interval='1d', auto_adjust=False)
if len(tickers_good) > 1:
    data_good = data_good['Adj Close']
if 'QQQ' in data_good.columns:
    data_good = data_good.rename(columns={'QQQ': 'NASDAQ'})
    index_names_good = [col if col != 'QQQ' else 'NASDAQ' for col in index_names_good]
existing_columns_good = [col for col in index_names_good if col in data_good.columns]
data_good = data_good[existing_columns_good]
data_good = data_good.dropna()

data_bad = yf.download(tickers_bad, start=start_date, end=end_date, interval='1d', auto_adjust=False)
if len(tickers_bad) > 1:
    data_bad = data_bad['Adj Close']
if 'QQQ' in data_bad.columns:
    data_bad = data_bad.rename(columns={'QQQ': 'NASDAQ'})
    index_names_bad = [col if col != 'QQQ' else 'NASDAQ' for col in index_names_bad]
existing_columns_bad = [col for col in index_names_bad if col in data_bad.columns]
data_bad = data_bad[existing_columns_bad]
data_bad = data_bad.dropna()

data_worst = yf.download(tickers_worst, start=start_date, end=end_date, interval='1d', auto_adjust=False)
if len(tickers_worst) > 1:
    data_worst = data_worst['Adj Close']
if 'QQQ' in data_worst.columns:
    data_worst = data_worst.rename(columns={'QQQ': 'NASDAQ'})
    index_names_worst = [col if col != 'QQQ' else 'NASDAQ' for col in index_names_worst]
existing_columns_worst = [col for col in index_names_worst if col in data_worst.columns]
data_worst = data_worst[existing_columns_worst]
data_worst = data_worst.dropna()

# Check if 'NVDA' is in good data and 'SPY' in bad/worst data to avoid IndexError
if 'NVDA' not in data_good.columns:
    raise ValueError("NVDA data is not available. Please check the ticker list or data source.")
if 'SPY' not in data_bad.columns or 'SPY' not in data_worst.columns:
    raise ValueError("SPY data is not available. Please check the ticker list or data source.")

# Simulate buy-and-hold for a single asset
def simulate_buy_hold(data, asset, daily_investment, initial_investment):
    portfolio = pd.DataFrame(index=data.index, columns=['value', asset, 'cash'])
    portfolio.iloc[0] = [initial_investment, 0, initial_investment]
    portfolio[f'{asset}_shares'] = 0.0
    initial_price = data[asset].iloc[0]
    shares = initial_investment / initial_price
    portfolio.loc[portfolio.index[0], f'{asset}_shares'] = shares
    portfolio.loc[portfolio.index[0], asset] = portfolio[f'{asset}_shares'].iloc[0] * data[asset].iloc[0]
    
    for i in range(1, len(data)):
        current_value = portfolio['cash'].iloc[i-1]
        daily_investment_per_asset = daily_investment  # All daily investment goes to the asset
        shares_to_buy = (daily_investment_per_asset / data[asset].iloc[i]) * (1 - 0.001)
        portfolio.loc[portfolio.index[i], f'{asset}_shares'] = portfolio[f'{asset}_shares'].iloc[i-1] + shares_to_buy
        portfolio.loc[portfolio.index[i], asset] = portfolio[f'{asset}_shares'].iloc[i] * data[asset].iloc[i]
        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - daily_investment_per_asset
        portfolio.loc[portfolio.index[i], 'value'] = portfolio[asset].iloc[i] + portfolio['cash'].iloc[i]
    return portfolio

# Simulate strategy with 2.6% drop threshold and 100% sell percentage
def simulate_strategy_2_6_100(data, indices, initial_investment, daily_investment_total):
    num_indices = len(indices)
    initial_investment_per_index = initial_investment / num_indices
    daily_investment_per_index = daily_investment_total / num_indices
    all_columns = ['value'] + indices + ['cash'] + [f"{idx}_shares" for idx in indices]
    portfolio = pd.DataFrame(index=data.index, columns=all_columns)
    portfolio.iloc[0] = [initial_investment] + [0] * num_indices + [initial_investment] + [0] * num_indices
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
        
        # Check drops and sell/buy logic
        dropped_indices = []
        for idx in indices:
            if idx in data.columns and i > 0:
                daily_return = (data[idx].iloc[i] - data[idx].iloc[i-1]) / data[idx].iloc[i-1]
                if daily_return <= -0.026:
                    dropped_indices.append(idx)
        
        if dropped_indices:
            total_redistribution = 0
            for idx in indices:
                if idx in data.columns and idx not in dropped_indices:
                    value_to_sell = portfolio[idx].iloc[i-1] * 1.0  # Sell 100% of non-dropped stocks
                    shares_to_sell = (value_to_sell / data[idx].iloc[i]) * (1 - 0.001)
                    if portfolio[f"{idx}_shares"].iloc[i] >= shares_to_sell:
                        portfolio.loc[portfolio.index[i], f"{idx}_shares"] -= shares_to_sell
                        portfolio.loc[portfolio.index[i], idx] = portfolio[f"{idx}_shares"].iloc[i] * data[idx].iloc[i]
                        total_redistribution += value_to_sell * (1 - 0.001)
            
            if total_redistribution > 0:
                cash_per_drop = total_redistribution / len(dropped_indices)
                for idx in dropped_indices:
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
buy_hold_portfolios_good = {asset: simulate_buy_hold(data_good, asset, daily_investment_total, initial_investment) for asset in index_names_good if asset in data_good.columns}
strategy_2_6_100_good = simulate_strategy_2_6_100(data_good, index_names_good, initial_investment, daily_investment_total)

buy_hold_portfolios_bad = {asset: simulate_buy_hold(data_bad, asset, daily_investment_total, initial_investment) for asset in index_names_bad if asset in data_bad.columns}
strategy_2_6_100_bad = simulate_strategy_2_6_100(data_bad, index_names_bad, initial_investment, daily_investment_total)

buy_hold_portfolios_worst = {asset: simulate_buy_hold(data_worst, asset, daily_investment_total, initial_investment) for asset in index_names_worst if asset in data_worst.columns}
strategy_2_6_100_worst = simulate_strategy_2_6_100(data_worst, index_names_worst, initial_investment, daily_investment_total)

# Function to add data labels to plot
def add_data_labels(ax, x, y, label_prefix):
    for i in [0, len(x)//4, len(x)//2, 3*len(x)//4, -1]:
        ax.annotate(f'{label_prefix} ${y.iloc[i]:,.0f}', (x[i], y.iloc[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)

# Create three plots with 9 lines each
plt.figure(figsize=(14, 12))

# Plot 1: Good Scenario
plt.subplot(3, 1, 1)
for asset, portfolio in buy_hold_portfolios_good.items():
    plt.plot(portfolio.index, portfolio['value'], label=f'Buy & Hold {asset}', linestyle='-', alpha=0.7)
plt.plot(strategy_2_6_100_good.index, strategy_2_6_100_good['value'], label='Strategy 2.6% & 100% Sell', color='red', linewidth=2)
plt.title('Buy & Hold vs Strategy 2.6% & 100% Sell (Good Picks: NVDA, TSM, MSFT, SPY, QQQ, SMH, XLK, USD) (2005–2025)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
add_data_labels(plt.gca(), strategy_2_6_100_good.index, strategy_2_6_100_good['value'], 'Strat')
for asset, portfolio in buy_hold_portfolios_good.items():
    add_data_labels(plt.gca(), portfolio.index, portfolio['value'], f'BH-{asset[:3]}')

# Plot 2: Bad Scenario
plt.subplot(3, 1, 2)
for asset, portfolio in buy_hold_portfolios_bad.items():
    plt.plot(portfolio.index, portfolio['value'], label=f'Buy & Hold {asset}', linestyle='-', alpha=0.7)
plt.plot(strategy_2_6_100_bad.index, strategy_2_6_100_bad['value'], label='Strategy 2.6% & 100% Sell', color='red', linewidth=2)
plt.title('Buy & Hold vs Strategy 2.6% & 100% Sell (Bad Picks: SPY, QQQ, SMH, XLK, USD, NHTC, BKNG, ZD) (2005–2025)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
add_data_labels(plt.gca(), strategy_2_6_100_bad.index, strategy_2_6_100_bad['value'], 'Strat')
for asset, portfolio in buy_hold_portfolios_bad.items():
    add_data_labels(plt.gca(), portfolio.index, portfolio['value'], f'BH-{asset[:3]}')

# Plot 3: Worst Scenario
plt.subplot(3, 1, 3)
for asset, portfolio in buy_hold_portfolios_worst.items():
    plt.plot(portfolio.index, portfolio['value'], label=f'Buy & Hold {asset}', linestyle='-', alpha=0.7)
plt.plot(strategy_2_6_100_worst.index, strategy_2_6_100_worst['value'], label='Strategy 2.6% & 100% Sell', color='red', linewidth=2)
plt.title('Buy & Hold vs Strategy 2.6% & 100% Sell (Worst Picks: SPY, QQQ, SMH, XLK, USD, NHTC, CLAR, EGAN) (2005–2025)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
add_data_labels(plt.gca(), strategy_2_6_100_worst.index, strategy_2_6_100_worst['value'], 'Strat')
for asset, portfolio in buy_hold_portfolios_worst.items():
    add_data_labels(plt.gca(), portfolio.index, portfolio['value'], f'BH-{asset[:3]}')

plt.tight_layout()
plt.show()