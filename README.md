# Multi Asset Valuation Tool

A Python program that pulls historical data from Yahoo Finance and analyzes **stocks, ETFs, and bonds**.

The tool provides:
- Annualized returns
- Risk metrics (volatility and max drawdown)
- Monte Carlo portfolio simulation
- A standalone valuation score (0–100) per ticker

Important characteristics:
- Each ticker is scored **independently** based on its own history
- Scores do **not** change based on what other tickers are entered
- Internet access is required

---

## Requirements

- Windows 10 or newer
- Python 3.10 or newer
- Internet connection

---

## Files in this repository

- `finance_sim.py` – main program
- `requirements.txt` – required Python libraries
- `README.md` – instructions (this file)

---

## Step 1: Install Python (Windows)

1. Go to:
   https://www.python.org/downloads/
2. Download Python 3.10 or newer
3. Run the installer
4. IMPORTANT: check the box:
   - **Add Python to PATH**
5. Finish installation

Verify Python is installed:
1. Open Command Prompt
2. Run:
   ```bash
   python --version

You should see something like:

Python 3.11.x

If `python` is not recognized, reinstall Python and make sure **Add Python to PATH** is checked.

---

## Step 2: Install Visual Studio Code (recommended)

VS Code is the easiest and most reliable way to run this program.

1. Download VS Code:
   [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Install it with default options

---

## Step 3: Install the Python extension in VS Code

1. Open VS Code
2. Click **Extensions** on the left sidebar
3. Search for **Python**
4. Install **Python by Microsoft**

---

## Step 4: Get the code

### Option A: Clone from GitHub

1. Open VS Code
2. Press:

```
  Ctrl + Shift + `
```

To open the terminal
3. Run:

   ```bash
   git clone https://github.com/Merikmh35/finance-tool.git
   cd finance-tool
   ```

### Option B: Download ZIP

1. Click **Code → Download ZIP** on GitHub
2. Extract the folder somewhere easy, for example:

   ```
   C:\Users\YourName\Documents\finance-tool
   ```
3. Open VS Code
4. Click **File → Open Folder**
5. Select the `finance-tool` folder

---

## Step 5: Select the Python interpreter (very important)

1. In VS Code press:

   ```
   Ctrl + Shift + P
   ```
2. Type:

   ```
   Python: Select Interpreter
   ```
3. Choose your installed Python version (3.10+)

If you skip this step, the program may fail to find installed libraries.

---

## Step 6: Install required libraries

1. In VS Code:

   * Click **Terminal → New Terminal**
2. Run:

   ```bash
   pip install -r requirements.txt
   ```

If that fails, use:

```bash
python -m pip install -r requirements.txt
```

---

## Step 7: Run the program (VS Code)

### Recommended method

In the VS Code terminal:

```bash
python finance_sim.py
```

### Alternate method

1. Open `finance_sim.py`
2. Click the **Run Python File** button (top right)

If you get missing module errors:

* Recheck the interpreter
* Reinstall dependencies using:

  ```bash
  python -m pip install -r requirements.txt
  ```

---

## Running without VS Code (Command Prompt)

1. Open Command Prompt
2. Navigate to the folder:

   ```bash
   cd C:\Users\YourName\Documents\finance-tool
   ```
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```
4. Run:

   ```bash
   python finance_sim.py
   ```

---

## Using the program

When prompted, enter tickers separated by commas.

Examples:

```
NVDA,GOOGL,AAPL
VFV.TO,SPY,TLT
META,TSLA
```

Menu options:

1. Annual returns
2. Risk metrics
3. Monte Carlo simulation
4. Plot Monte Carlo paths
5. Valuation scores
6. Exit

Notes:

* Monte Carlo simulations may take a few seconds
* Close plot windows to return to the menu

---

## Valuation score explanation (0–100)

The valuation score is based on historical performance and risk:

* Annualized return
* Risk-adjusted return (Sharpe style)
* Volatility
* Maximum drawdown (crash risk)

For stocks only:

* A valuation penalty using P/E and P/B

Important notes:

* ETFs and bonds do not rely on stock valuation ratios
* Scores are **absolute**, not relative to the input list
* A score of 100 means very strong by the model
* A score of 0 means very weak risk-adjusted performance

---

## Updating the program (GitHub users)

To get the latest version:

```bash
git pull
```

To update libraries:

```bash
pip install -r requirements.txt --upgrade
```

---

## Common issues

### "pip is not recognized"

Use:

```bash
python -m pip install -r requirements.txt
```

### "No module named yfinance / pandas / numpy"

You installed libraries for a different Python version.
Fix:

1. Select interpreter again in VS Code
2. Run:

   ```bash
   python -m pip install -r requirements.txt
   ```

### Ticker not found or empty results

* Check spelling
* Add exchange suffix if needed (example: `.TO` for Canada)
* Yahoo Finance may occasionally fail temporarily

---

## Disclaimer

This tool is for educational and analytical purposes only.
It is not financial advice.
