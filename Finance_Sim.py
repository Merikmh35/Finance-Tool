# pip install yfinance pandas numpy matplotlib

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

YEARS_HISTORY = 5
RISK_FREE_RATE = 0.03
TRADING_DAYS = 252


# -----------------------------
# Data (no shared dropna, ticker scores stay consistent)
# -----------------------------
def get_prices(tickers, years=YEARS_HISTORY):
    data = yf.download(
        tickers,
        period=f"{years}y",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if isinstance(tickers, list) and len(tickers) == 1:
        return data["Close"].to_frame(tickers[0])

    prices = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.levels[0]:
                try:
                    prices[t] = data[t]["Close"]
                except Exception:
                    pass
        out = pd.DataFrame(prices)
        if out.shape[1] == 0:
            raise RuntimeError("No valid tickers returned price data.")
        return out

    if "Close" in data.columns:
        return data["Close"]

    raise RuntimeError("Price data not found.")


# -----------------------------
# Math helpers (work with Series or DataFrame)
# -----------------------------
def annual_return(prices):
    if isinstance(prices, pd.Series):
        years = len(prices) / TRADING_DAYS
        return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1
    years = len(prices) / TRADING_DAYS
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1


def annual_volatility(prices):
    if isinstance(prices, pd.Series):
        log_ret = np.log(prices / prices.shift(1)).dropna()
        return float(log_ret.std() * np.sqrt(TRADING_DAYS))
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(prices):
    if isinstance(prices, pd.Series):
        cum = prices / prices.iloc[0]
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())
    cum = prices / prices.iloc[0]
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def sharpe_ratio(ret, vol):
    vol = float(vol)
    if vol <= 0:
        return 0.0
    return float((ret - RISK_FREE_RATE) / vol)


# -----------------------------
# Asset type (best effort)
# -----------------------------
def detect_type(ticker):
    try:
        info = yf.Ticker(ticker).info
        qt = info.get("quoteType", "").upper()
        if qt in ["ETF", "BOND"]:
            return qt.title()
        return "Stock"
    except Exception:
        return "Stock"


# -----------------------------
# Absolute scoring helpers (no list comparisons)
# -----------------------------
def _sigmoid(x):
    x = float(x)
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def _clamp01(x):
    return float(max(0.0, min(1.0, x)))


def component_scores(ret, vol, dd, sharpe):
    # Absolute targets
    # Return: 8% neutral, 14% strong
    ret_s = _sigmoid((ret - 0.08) / 0.05)

    # Sharpe: 0.4 neutral, 1.0 strong
    sharpe_s = _sigmoid((sharpe - 0.40) / 0.35)

    # Volatility: 25% neutral, lower is better
    vol_s = _sigmoid((0.25 - vol) / 0.08)

    # Drawdown: 45% neutral, lower is better
    dd_abs = abs(dd)
    dd_s = _sigmoid((0.45 - dd_abs) / 0.15)

    return {
        "ret_s": _clamp01(ret_s),
        "sharpe_s": _clamp01(sharpe_s),
        "vol_s": _clamp01(vol_s),
        "dd_s": _clamp01(dd_s),
    }


def stock_valuation_penalty(pe, pb):
    # Bounded penalty in [0, 0.35]
    penalties = []

    if isinstance(pe, (int, float)) and pe > 0:
        p = np.log1p(pe / 35.0) / np.log1p(6.0)
        penalties.append(min(float(p), 1.0))

    if isinstance(pb, (int, float)) and pb > 0:
        p = np.log1p(pb / 6.0) / np.log1p(6.0)
        penalties.append(min(float(p), 1.0))

    if not penalties:
        return 0.0

    avg = float(sum(penalties) / len(penalties))
    return min(0.35, 0.35 * avg)


# -----------------------------
# Valuation table (consistent per ticker)
# -----------------------------
def valuation_table(tickers):
    prices = get_prices(tickers)
    rows = []

    for t in prices.columns:
        p = prices[t].dropna()

        if len(p) < TRADING_DAYS * 2:
            # not enough data to score meaningfully
            rows.append({
                "Type": detect_type(t),
                "Score": 0.0,
                "Annual Return %": "-",
                "Volatility %": "-",
                "Max Drawdown %": "-",
                "Sharpe": "-",
                "P/E": "-",
                "P/B": "-",
            })
            continue

        ret = float(annual_return(p))
        vol = float(annual_volatility(p))
        dd = float(max_drawdown(p))
        sharpe = float(sharpe_ratio(ret, vol))

        try:
            info = yf.Ticker(t).info
        except Exception:
            info = {}

        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        asset_type = detect_type(t)

        comps = component_scores(ret, vol, dd, sharpe)

        base = (
            0.30 * comps["ret_s"] +
            0.25 * comps["sharpe_s"] +
            0.20 * comps["vol_s"] +
            0.25 * comps["dd_s"]
        )

        penalty = 0.0
        if asset_type == "Stock":
            penalty = stock_valuation_penalty(pe, pb)

        score = 100.0 * base * (1.0 - penalty)
        score = float(np.clip(score, 0.0, 100.0))

        rows.append({
            "Type": asset_type,
            "Score": round(score, 2),
            "Annual Return %": round(ret * 100, 2),
            "Volatility %": round(vol * 100, 2),
            "Max Drawdown %": round(dd * 100, 2),
            "Sharpe": round(sharpe, 2),
            "P/E": round(pe, 2) if isinstance(pe, (int, float)) else "-",
            "P/B": round(pb, 2) if isinstance(pb, (int, float)) else "-",
        })

    df = pd.DataFrame(rows, index=prices.columns).sort_values("Score", ascending=False)

    print("\nValuation Overview (0-100, standalone scoring)\n")
    print(df)
    return df


# -----------------------------
# Monte Carlo
# -----------------------------
def monte_carlo(prices, paths=5000, seed=42):
    # Use overlap for simulation only (portfolio needs aligned dates)
    aligned = prices.dropna(how="any")
    if aligned.shape[0] < TRADING_DAYS:
        raise RuntimeError("Not enough overlapping data for portfolio simulation.")

    log_ret = np.log(aligned / aligned.shift(1)).dropna()
    mu = log_ret.mean().values * TRADING_DAYS
    cov = log_ret.cov().values * TRADING_DAYS

    n_assets = aligned.shape[1]
    weights = np.repeat(1 / n_assets, n_assets)

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)

    Z = rng.standard_normal((TRADING_DAYS, paths, n_assets))
    shocks = Z @ L.T / np.sqrt(TRADING_DAYS)

    drift = (mu - 0.5 * np.diag(cov)) / TRADING_DAYS
    log_paths = np.cumsum(drift + shocks, axis=0)

    S0 = aligned.iloc[-1].values
    prices_sim = S0 * np.exp(log_paths)

    portfolio = (prices_sim * weights).sum(axis=2)
    return portfolio


def plot_paths(portfolio):
    plt.figure(figsize=(9, 4))
    total_paths = portfolio.shape[1]
    plot_n = min(150, total_paths)

    idx = np.random.choice(total_paths, plot_n, replace=False)
    for i in idx:
        plt.plot(portfolio[:, i], alpha=0.3, linewidth=0.6)

    plt.title("Monte Carlo Portfolio Paths")
    plt.xlabel("Days")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
    plt.close()


# -----------------------------
# Menu
# -----------------------------
def menu():
    print("\nMulti Asset Valuation Tool\n")

    raw = input("Enter tickers (Eg: NVDA,VFV.TO,GOOGL): ")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    if not tickers:
        print("No valid tickers.")
        return

    prices = get_prices(tickers)
    portfolio = None

    while True:
        print("\nChoose an option")
        print("1 - Annual returns")
        print("2 - Risk metrics")
        print("3 - Monte Carlo simulation")
        print("4 - Plot Monte Carlo paths")
        print("5 - Valuation scores (standalone)")
        print("6 - Exit")

        choice = input("Choice: ").strip()

        if choice == "1":
            aligned = prices.dropna(how="any")
            ret = annual_return(aligned) * 100
            print("\nAnnual Returns % (portfolio-aligned)\n")
            print(ret.round(2))

        elif choice == "2":
            aligned = prices.dropna(how="any")
            vol = annual_volatility(aligned) * 100
            dd = max_drawdown(aligned) * 100
            print("\nRisk Metrics (portfolio-aligned)\n")
            print(pd.DataFrame({
                "Volatility %": vol.round(2),
                "Max Drawdown %": dd.round(2)
            }))

        elif choice == "3":
            portfolio = monte_carlo(prices)
            print("\nMonte Carlo simulation complete.")

        elif choice == "4":
            if portfolio is None:
                portfolio = monte_carlo(prices)
            plot_paths(portfolio)

        elif choice == "5":
            valuation_table(tickers)

        elif choice == "6":
            print("Done.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    menu()