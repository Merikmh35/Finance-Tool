# pip install yfinance pandas numpy matplotlib if needed
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# list of tickers
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
n_assets = len(tickers)

# fetch historical prices
data = yf.download(tickers, period="3y", progress=False)

# handle yf format robustly
if isinstance(data.columns, pd.MultiIndex):
    # MultiIndex: pick 'Adj Close' if it exists, else 'Close'
    if "Adj Close" in data.columns.levels[0]:
        df = data["Adj Close"].dropna(how="any")
    elif "Close" in data.columns.levels[0]:
        df = data["Close"].dropna(how="any")
    else:
        raise RuntimeError("Unexpected yf.download MultiIndex format")
else:
    # single-level: use tickers if present, else 'Close'
    if set(tickers).issubset(data.columns):
        df = data[tickers].dropna(how="any")
    elif "Close" in data.columns:
        df = data["Close"].dropna(how="any")
    else:
        raise RuntimeError("Unexpected yf.download single-level format")

# daily log returns
logr = np.log(df).diff().dropna()
S0 = df.iloc[-1].values.astype(float)  # latest prices

# simulation parameters
T = 1.0      # 1 year
steps = 252  # trading days
paths = 5000 # number of simulated paths
dt = T / steps
seed = 123

# annualized mean and covariance
mu = logr.mean().values * 252
Sigma = logr.cov().values * 252

# Cholesky decomposition for correlated random numbers
rng = np.random.default_rng(seed)
L = np.linalg.cholesky(Sigma)

# generate random paths
Z = rng.standard_normal((steps, paths, n_assets))
corr = Z @ L.T * np.sqrt(dt)

drift = (mu - 0.5 * np.diag(Sigma)) * dt
log_inc = drift + corr

# simulate prices
log_prices = np.cumsum(log_inc, axis=0)
log_prices = np.vstack([np.zeros((1, paths, n_assets)), log_prices])
prices = S0 * np.exp(log_prices)

# equal-weight portfolio
w = np.repeat(1 / n_assets, n_assets)
portfolio = (prices * w).sum(axis=2)

# final stats
ST = portfolio[-1]
p5, p50, p95 = np.percentile(ST, [5, 50, 95])
print(f"Start: {np.sum(S0*w):.2f} | 5%={p5:.2f} median={p50:.2f} 95%={p95:.2f}")

# plot sample paths
plt.figure(figsize=(9, 4))
sample_paths = min(200, paths)
for i in rng.choice(paths, size=sample_paths, replace=False):
    plt.plot(portfolio[:, i], alpha=0.35, linewidth=0.6)
plt.axhline(np.sum(S0*w), linestyle="--", color="black")
plt.title("Sample Portfolio Paths")
plt.xlabel("Day")
plt.ylabel("Portfolio Value")
plt.tight_layout()
plt.show()