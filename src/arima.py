"""ARIMA baseline and the ARIMA + tree-model hybrid feature.

Per series (item x market x price type) an ARIMA(7,1,0) is fitted on the daily log-price series using data
up to `fit_end` only (differencing once, then an AR(7) by conditional least squares, which is how ARIMA(p,1,0) is
estimated by the CSS method). The 7-day-ahead forecast of the cumulative log change (`arima_fc7`) is then computed for
every report day with the fitted coefficients (pure algebra, no refitting, no future data):

    d_hat(t+h) = sum_k phi_k * d(t+h-k),   forecasts replace unknown differences,  fc7 = sum_{h=1..7} d_hat(t+h)

It is used (a) directly as an "existing approach" baseline score and (b) as an extra feature for Random Forest
(the lecturer's ARIMA + RF hybrid idea).
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

import prep

ORDER = (7, 1, 0)


def _daily_log_series(p):
    d = p.set_index("date")
    return {c: np.log(d[c].ffill()) for c in prep.price_columns(p)}


def fit_params(p, fit_end):
    """AR coefficients on the differenced log price for each series, fitted on dates < fit_end."""
    phis = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c, x in _daily_log_series(p).items():
            d = x[x.index < fit_end].dropna().diff().dropna().values      # difference once (the "I" in ARIMA)
            phis[c] = AutoReg(d, lags=ORDER[0], trend="n").fit().params   # AR(7) on differences by least squares
    return phis


def fc7_table(p, phis):
    """Rolling 7-day-ahead forecast of the cumulative log change at every day, per series."""
    out = {}
    k = ORDER[0]
    for c, x in _daily_log_series(p).items():
        d = x.diff().fillna(0.0).values
        n = len(d)
        # last k differences at each t (columns: d_t, d_{t-1}, ... d_{t-k+1})
        hist = np.column_stack([np.r_[np.zeros(j), d[:n - j]] for j in range(k)])
        phi = phis[c]
        total = np.zeros(n)
        for _ in range(7):
            nxt = hist @ phi
            total += nxt
            hist = np.column_stack([nxt, hist[:, :-1]])
        out[c] = pd.Series(total, index=x.index)
    return out


def add_arima_feature(L, p, phis):
    """Attach `arima_fc7` to the long table (match each row's series and date)."""
    table = fc7_table(p, phis)
    vals = np.empty(len(L))
    for key, idx in L.groupby(["price_type", "market", "item"]).groups.items():
        col = "_".join(key)
        vals[L.index.get_indexer(idx)] = table[col].reindex(L.loc[idx, "date"]).values
    out = L.copy()
    out["arima_fc7"] = vals
    return out


def add_features_arima(df):
    """Engineered features + the ARIMA forecast (hybrid input)."""
    X = prep.add_features(df)
    X["arima_fc7"] = df["arima_fc7"].values
    return X
