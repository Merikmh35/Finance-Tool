# Multi Asset Valuation Tool

A Python program that pulls historical data from Yahoo Finance and analyzes stocks, ETFs, and bonds.

The tool provides:
- Annualized returns
- Risk metrics (volatility and max drawdown)
- Monte Carlo portfolio simulation
- A standalone valuation score (0–100) per ticker  
  - Each asset is scored on its own history  
  - Scores do NOT depend on what other tickers are entered  

Internet access is required.

---

## Requirements

- Windows 10 or newer
- Python 3.10 or newer
- Internet connection

---

## Installation (Windows)

### 1. Install Python
Download Python from:
https://www.python.org

During installation:
- Check **Add Python to PATH**

Verify installation:
```bash
python --version
