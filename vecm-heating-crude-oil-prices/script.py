import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


def fetch_data(symbols, start="2015-01-01", end="2024-12-31"):
    """
    Fetches historical daily closing prices from Yahoo Finance for specified symbols.

    Args:
        symbols (list): List of ticker symbols to download (e.g., ['HO=F', 'CL=F']).
        start (str): Start date for the data in 'YYYY-MM-DD' format.
        end (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with columns ['HeatingOil', 'CrudeOil'] and date index.
    """
    df = yf.download(symbols, start=start, end=end)['Close'].dropna()
    df.columns = ['HeatingOil', 'CrudeOil']
    return df


def plot_series(df, filename="cointegration_series.png"):
    """
    Plots the input time series and saves the figure to a file.

    Args:
        df (pd.DataFrame): DataFrame with time series to plot.
        filename (str): Name of the output PNG file.
    """
    df.plot(title="Heating Oil vs Crude Oil Prices")
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def adf_summary(df):
    """
    Performs Augmented Dickey-Fuller (ADF) tests on both levels and first differences
    of each column in the DataFrame and prints the results.

    Args:
        df (pd.DataFrame): DataFrame containing time series to test.
    """
    for col in df.columns:
        result = adfuller(df[col])
        print(f"{col} ADF: {result[0]:.4f}, p={result[1]:.4f}")
        result_diff = adfuller(df[col].diff().dropna())
        print(f"{col} ΔADF: {result_diff[0]:.4f}, p={result_diff[1]:.4f}")


def johansen_test(df):
    """
    Runs the Johansen cointegration test on the input data.

    Args:
        df (pd.DataFrame): DataFrame containing non-stationary I(1) time series.

    Returns:
        coint_johansen: The result object containing test statistics and eigenvectors.
    """
    result = coint_johansen(df, det_order=0, k_ar_diff=1)
    print("Trace Statistic:", result.lr1)
    print("Critical Values (90%, 95%, 99%):\n", result.cvt)
    return result


def fit_vecm(df, lags=1, rank=1):
    """
    Fits a Vector Error Correction Model (VECM) to the data.

    Args:
        df (pd.DataFrame): Time series data to model.
        lags (int): Number of lags to use in the VECM.
        rank (int): Cointegration rank determined from Johansen test.

    Returns:
        VECMResults: Fitted VECM model results object.
    """
    model = VECM(df, k_ar_diff=lags, coint_rank=rank)
    res = model.fit()
    print(res.summary())
    return res


def forecast_vecm(res, df, steps=12):
    """
    Forecasts future values using the fitted VECM and plots results.

    Args:
        res (VECMResults): Fitted VECM results object.
        df (pd.DataFrame): Original time series data.
        steps (int): Number of periods to forecast forward.
    """
    forecast = res.predict(steps=steps)
    future_idx = pd.date_range(df.index[-1], periods=steps, freq='ME')
    forecast_df = pd.DataFrame(forecast, columns=df.columns, index=future_idx)
    
    df[-100:].plot(figsize=(10, 5), label='Historical')
    forecast_df.plot(ax=plt.gca(), style='--')
    plt.title("12-Month VECM Forecast")
    plt.savefig("vecm_forecast.png")
    plt.show()


def run_var_irf_fevd(df_diff, lags=1, horizon=12):
    """
    Fits a VAR model on differenced data, plots impulse response functions (IRFs)
    and forecast error variance decomposition (FEVD).

    Args:
        df_diff (pd.DataFrame): First-differenced time series data.
        lags (int): Number of lags for the VAR model.
        horizon (int): Forecast horizon for IRF and FEVD plots.
    """
    model = VAR(df_diff)
    res = model.fit(maxlags=lags)
    print(res.summary())

    irf = res.irf(horizon)
    irf.plot(orth=False)
    plt.suptitle("Impulse Response (Standard)")
    plt.savefig("var_irf_standard.png")
    plt.show()

    irf.plot(orth=True)
    plt.suptitle("Impulse Response (Orthogonalized)")
    plt.savefig("var_irf_orthogonalized.png")
    plt.show()

    fevd = res.fevd(horizon)
    fevd.plot()
    plt.suptitle("Forecast Error Variance Decomposition (FEVD)")
    plt.savefig("var_fevd.png")
    plt.show()


def main():
    """
    Runs the full cointegration analysis pipeline:
    - Downloads data
    - Performs stationarity and cointegration tests
    - Fits and forecasts VECM
    - Plots IRFs and FEVD from a VAR on differenced data
    """
    symbols = ['HO=F', 'CL=F']  # Heating Oil and Crude Oil
    df = fetch_data(symbols)
    plot_series(df)
    adf_summary(df)

    johansen_test(df)
    vecm_res = fit_vecm(df, lags=1, rank=1)
    forecast_vecm(vecm_res, df)

    df_diff = df.diff().dropna()
    run_var_irf_fevd(df_diff, lags=1, horizon=12)


if __name__ == "__main__":
    main()