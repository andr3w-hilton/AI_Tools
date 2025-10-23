# Portfolio Health Checker - TODO List

## Current Status
Terminal-based portfolio tracker with basic buy/hold/sell recommendations based on price paid vs current price.

## Priority Improvements

### 1. Add Profit/Loss Calculations
- [ ] Calculate dollar amount gain/loss per stock (current_price - price_paid)
- [ ] Calculate percentage gain/loss per stock ((current_price - price_paid) / price_paid * 100)
- [ ] Add gain/loss columns to the output table
- [ ] Calculate total portfolio value
- [ ] Calculate overall portfolio P&L (sum of all gains/losses)
- [ ] Add summary section showing totals

### 2. Error Handling & Reliability
- [ ] Add try/except blocks around yfinance API calls
- [ ] Handle cases where ticker data isn't available (invalid symbols)
- [ ] Handle network failures gracefully
- [ ] Add retry logic for failed API calls (with exponential backoff)
- [ ] Add timeout handling
- [ ] Log errors to file for debugging
- [ ] Provide user-friendly error messages

### 3. Performance Optimization
- [ ] Implement parallel stock data fetching (threading or asyncio)
- [ ] Add caching mechanism to avoid repeated API calls
- [ ] Cache results for configurable time period (e.g., 5 minutes)
- [ ] Add --force-refresh flag to bypass cache
- [ ] Measure and display fetch time

### 4. Additional Stock Metrics
- [ ] Add number of shares owned for each stock
- [ ] Calculate total position value (shares * current_price)
- [ ] Show 52-week high price
- [ ] Show 52-week low price
- [ ] Display day's change percentage
- [ ] Show volume information
- [ ] Add market cap information
- [ ] Display P/E ratio

### 5. Portfolio Configuration
- [ ] Move portfolio data to separate JSON config file
- [ ] Support loading portfolio from CSV file
- [ ] Add fields: symbol, price_paid, shares_owned, purchase_date
- [ ] Create example config file template
- [ ] Add validation for config file format
- [ ] Support multiple portfolios (profiles)

Example portfolio.json structure:
```json
{
  "portfolio": [
    {
      "symbol": "VUAG.L",
      "price_paid": 64.1,
      "shares": 10,
      "purchase_date": "2024-01-15"
    }
  ]
}
```

### 6. Command-Line Arguments
- [ ] Add `--refresh` or `-r` flag to force refresh data
- [ ] Add `--save` or `-s` flag to save to CSV
- [ ] Add `--history` or `-h` flag to show historical data
- [ ] Add `--config <file>` to specify config file location
- [ ] Add `--no-color` flag for plain text output
- [ ] Add `--format <csv|json|table>` for output format
- [ ] Add `--help` for usage information
- [ ] Add `--verbose` or `-v` for detailed output

### 7. Historical Tracking & Trending
- [ ] Append to CSV with timestamp instead of overwriting
- [ ] Create historical_portfolio.csv with date column
- [ ] Track portfolio value over time
- [ ] Show performance trends (week/month/year)
- [ ] Generate simple ASCII charts for performance
- [ ] Compare current performance to past snapshots
- [ ] Calculate best/worst performing days

### 8. Improved Action/Rating Logic
- [ ] Fix color reset bug on line 62 (bkgrn_colors.RESET → colors.RESET)
- [ ] Make thresholds configurable per stock
- [ ] Add stop-loss threshold field to portfolio config
- [ ] Add target price field for profit-taking
- [ ] Add more granular ratings (Strong Buy, Buy, Hold, Sell, Strong Sell)
- [ ] Consider time held (don't panic sell recent purchases)
- [ ] Add market condition context (bull/bear market indicators)
- [ ] Add volatility-based recommendations

Rating Tiers:
- Strong Buy: > 10% above price paid
- Buy: 5-10% above price paid
- Hold: -5% to +5% of price paid
- Sell: -5% to -10% of price paid
- Strong Sell: < -10% of price paid

### 9. Summary & Analytics
- [ ] Add summary section at top or bottom
- [ ] Show total amount invested
- [ ] Show current total portfolio value
- [ ] Show overall portfolio P&L (dollars and percentage)
- [ ] Display best performing stock (highest % gain)
- [ ] Display worst performing stock (highest % loss)
- [ ] Show portfolio diversity metrics
- [ ] Add sector allocation if available
- [ ] Sort stocks by performance (best to worst)

### 10. Alerts & Notifications
- [ ] Add threshold-based alerts (e.g., stock drops 5%)
- [ ] Highlight stocks hitting new highs/lows
- [ ] Add visual indicator for stocks needing attention
- [ ] Email notifications (via SMTP)
- [ ] Desktop notifications (for urgent sells)
- [ ] Save alert history

### 11. Code Quality & Structure
- [ ] Refactor into classes (Portfolio, Stock, Metrics)
- [ ] Separate concerns: data fetching, calculations, display
- [ ] Add type hints for better code documentation
- [ ] Add docstrings to all functions
- [ ] Create requirements.txt with pinned versions
- [ ] Add unit tests for calculations
- [ ] Create README.md with usage instructions

### 12. Data Export Options
- [ ] Keep existing CSV export but with timestamp
- [ ] Add JSON export option
- [ ] Add Excel export option (.xlsx)
- [ ] Add HTML report generation
- [ ] Add PDF report generation (optional, advanced)
- [ ] Email report option (implement from comment on line 113)

### 13. Database Integration (Optional)
- [ ] Implement SQLite database for historical data
- [ ] Create schema: portfolios, stocks, prices, transactions
- [ ] Add functions to query historical performance
- [ ] Add backup/restore functionality
- [ ] Add data migration scripts

## Future Enhancements (Nice to Have)

### Advanced Features
- [ ] Web dashboard (Flask/FastAPI) for visualization
- [ ] Real-time price updates (websocket feeds)
- [ ] Portfolio rebalancing suggestions
- [ ] Tax loss harvesting recommendations
- [ ] Dividend tracking and calendar
- [ ] Integration with broker APIs (live portfolio sync)
- [ ] Backtesting strategies
- [ ] Risk analysis (beta, correlation, Sharpe ratio)
- [ ] Compare portfolio performance to benchmarks (S&P 500, etc.)

### API & Integration
- [ ] Support multiple data sources (Alpha Vantage, IEX Cloud, etc.)
- [ ] Add cryptocurrency support
- [ ] Add forex/currency tracking
- [ ] Add commodity tracking (gold, oil, etc.)

### Visualization
- [ ] Generate performance charts (matplotlib/plotly)
- [ ] Pie chart for portfolio allocation
- [ ] Line chart for historical performance
- [ ] Candlestick charts for individual stocks
- [ ] Heat map for correlation matrix

## Bug Fixes
- [ ] Fix color reset bug on line 62: `bkgrn_colors.RESET` should be `colors.RESET`

## Documentation
- [ ] Create comprehensive README.md
- [ ] Add installation instructions
- [ ] Add usage examples
- [ ] Document configuration file format
- [ ] Add troubleshooting section
- [ ] Create CHANGELOG.md for version tracking

## Testing Checklist
- [ ] Test with invalid stock symbols
- [ ] Test with no internet connection
- [ ] Test with large portfolios (100+ stocks)
- [ ] Test with different timezones
- [ ] Test CSV export/import
- [ ] Test all command-line arguments
- [ ] Test color output in different terminals

## Notes
- Keep the terminal-based approach - it's clean and functional
- Maintain backward compatibility when adding features
- Consider performance impact of new features (API rate limits)
- yfinance has rate limits - implement caching and error handling
