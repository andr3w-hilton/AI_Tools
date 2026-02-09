# This script will use the yfinance library to get the current stock price from my portfolio of stocks.
# I will use this to compare the current price to the price I paid for the stock to see if I am making or losing money.
# I will also use this to see if I should sell the stock or not.

# Example of stocks in your portfolio:
# 1. VUAG
# 2. PLTR
# 3. TSLA
# 4. AMZN
# 5. AAPL
# 6. NVDA
# 7. SOUN

# Import libraries
import yfinance as yf
import pandas as pd
from tabulate import tabulate
import datetime


class colors:
    GREEN = '\033[32m'  # GREEN
    YELLOW = '\033[33m'  # YELLOW
    RED = '\033[31m'  # RED
    RESET = '\033[0m'  # RESET color


class bkgrn_colors:
    GREEN = '\033[42m'  # GREEN
    YELLOW = '\033[43m'  # YELLOW
    RED = '\033[41m'  # RED
    RESET = '\033[0m'  # RESET color


# Define your portfolio and price paid
portfolio = {
    'VUAG.L': 64.1,  # Price paid for Vanguard S&P 500 ETF
    'TSLA': 257.9,  # Price paid for Tesla
    'AMZN': 129.3,  # Price paid for Amazon
    'AAPL': 191.6,  # Price paid for Apple
    'SOUN': 04.1  # Price paid for Sound Hound
}


def get_company_name(symbol):
    ticker = yf.Ticker(symbol)
    return ticker.info['shortName']


def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]


def get_action(current_price, price_paid):
    if current_price > price_paid:
        return colors.GREEN + 'Safe: Buy/Hold' + colors.RESET
    elif current_price < (0.9 * price_paid):
        return bkgrn_colors.RED + 'Urgent: Sell' + colors.RESET
    else:
        return colors.YELLOW + 'Caution: Hold' + colors.RESET


data = {
    'company_name': [],
    'current_price': [],
    'price_paid': [],
    'action': []
}

for stock_name, price_paid in portfolio.items():
    company_name = get_company_name(stock_name)
    current_price = get_current_price(stock_name)
    action = get_action(current_price, price_paid)

    data['company_name'].append(company_name)
    data['current_price'].append(current_price)
    data['price_paid'].append(price_paid)
    data['action'].append(action)


df = pd.DataFrame(data)

df.rename(columns={
    'company_name': 'Company Name',
    'current_price': 'Current Price',
    'price_paid': 'Price I Paid',
    'action': 'Rating'
}, inplace=True)

# print multiple line ascii art

print("""
   ___            __       _ _           ___ _               _             
  / _ \___  _ __ / _| ___ | (_) ___     / __\ |__   ___  ___| | _____ _ __ 
 / /_)/ _ \| '__| |_ / _ \| | |/ _ \   / /  | '_ \ / _ \/ __| |/ / _ \ '__|
/ ___/ (_) | |  |  _| (_) | | | (_) | / /___| | | |  __/ (__|   <  __/ |   
\/    \___/|_|  |_|  \___/|_|_|\___/  \____/|_| |_|\___|\___|_|\_\___|_|   
""")

# This line prints Stock info as of: the date and time
print('Portfolio Status as of:', (datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
print('')

# use tabulate to print df with boxes
print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

# I would like a feature to save the data to a csv file
df.to_csv('portfolio.csv')

# I would like a feature to save the data to a database
# I would like a feature to email me the data


