"""Data preparation for the Sri Lanka vegetable price-spike early-warning model.

Source: CBSL Daily Price Reports (parsed to CSV by the dataset author) + Open-Meteo rainfall.

Task (classification): for an item in a Colombo-area market on a report day t,
will its price RISE by >= 10% within the next 7 days?

Key decisions (each checked against the data, see notebook):
  * The file has 1,339 daily rows but only 733 distinct CBSL reports; the other rows are exact copies
    of the previous report (the `is_interpolated` flag is False everywhere, so it cannot be trusted).
    -> only genuine report days are used as examples.
  * Wide table (one column per item x market x price type) -> long table (one row per series per day).
  * Every feature uses information available on day t only (no future leakage).
  * Time-ordered splits: train 2023-24, validation 2025, test 2026 (untouched until the end).
"""
import numpy as np
import pandas as pd

HORIZON = 7           # days ahead
SPIKE = 0.10          # +10% price rise
ANCHORS = ["price_now", "price_7d_ago", "price_14d_ago", "price_28d_ago"]
RAW_FEATURES = ["item", "market", "price_type"] + ANCHORS + ["rain_ne_7d_mm", "rain_pol_7d_mm", "month"]
CATEGORICAL = ["item", "market", "price_type"]
NUMERIC = ["lp", "r7", "r14", "r28", "prev7", "gap4", "vol4", "rain_ne_7d_mm", "rain_pol_7d_mm", "month"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "spike"

TRAIN_END = pd.Timestamp("2025-01-01")   # train: target window ends before this
VAL_END = pd.Timestamp("2026-01-01")     # validation: 2025; test: 2026 onwards


def load_wide(csv_path):
    """Read the raw CSV, parse dates and the report date embedded in `source_pdf`."""
    p = pd.read_csv(csv_path)
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p = p.sort_values("date").reset_index(drop=True)
    p["pdf_date"] = pd.to_datetime(p["source_pdf"].str.extract(r"(\d{8})")[0], format="%Y%m%d")
    return p


def price_columns(p, max_missing=0.5):
    cols = [c for c in p.columns if c.startswith(("wholesale_", "retail_"))]
    return [c for c in cols if p[c].isna().mean() <= max_missing]


def build_long(p, max_missing=0.5):
    """One row per (report day, series). Features are anchor prices at t, t-7, t-14, t-28 (known at t)."""
    is_report = p["date"] == p["pdf_date"]               # genuine report days only
    d = p.set_index("date")
    rain = d[["rainfall_nuwara_eliya_mm", "rainfall_polonnaruwa_mm"]].rolling(7, min_periods=7).sum()
    rain.columns = ["rain_ne_7d_mm", "rain_pol_7d_mm"]
    rep_dates = p.loc[is_report, "date"]
    parts = []
    for c in price_columns(p, max_missing):
        price_type, market, item = c.split("_", 2)
        s = d[c].ffill()                                  # last published price as of each day
        f = pd.DataFrame({
            "price_now": s, "price_7d_ago": s.shift(7), "price_14d_ago": s.shift(14), "price_28d_ago": s.shift(28),
            "future_price": s.shift(-HORIZON),
        })
        f = f.join(rain)
        f["item"], f["market"], f["price_type"] = item, market, price_type
        parts.append(f.loc[f.index.intersection(rep_dates)])
    L = pd.concat(parts).rename_axis("date").reset_index()
    L["month"] = L["date"].dt.month
    L = L.dropna(subset=ANCHORS + ["future_price", "rain_ne_7d_mm", "rain_pol_7d_mm"]).reset_index(drop=True)
    L["change_7d"] = L["future_price"] / L["price_now"] - 1
    L[TARGET] = (L["change_7d"] >= SPIKE).astype(int)
    return L.sort_values(["date", "item", "market", "price_type"]).reset_index(drop=True)


def add_features(df):
    """Raw inputs -> engineered features. Used inside the pickled pipeline, so the app only needs raw inputs."""
    out = df.copy()
    a = np.log(out[ANCHORS].astype(float).clip(lower=1e-6))
    out["lp"] = a["price_now"]
    out["r7"] = a["price_now"] - a["price_7d_ago"]          # change over last week
    out["r14"] = a["price_now"] - a["price_14d_ago"]
    out["r28"] = a["price_now"] - a["price_28d_ago"]
    out["prev7"] = a["price_7d_ago"] - a["price_14d_ago"]   # the week before
    out["gap4"] = a["price_now"] - a.mean(axis=1)           # distance from the 4-anchor average
    out["vol4"] = a.std(axis=1)                             # recent volatility
    return out[FEATURES]


# ---------------------------------------------------------------- time-ordered splits
def split(L):
    """train: 2023-24 | validation: 2025 | test: 2026. A row's target looks 7 days ahead, so each
    split keeps only rows whose whole target window lies inside it (no overlap across the boundary)."""
    end = L["date"] + pd.Timedelta(days=HORIZON)
    train = L[end < TRAIN_END]
    val = L[(L["date"] >= TRAIN_END) & (end < VAL_END)]
    test = L[L["date"] >= VAL_END]
    return train, val, test


def tuning_folds(L):
    """Expanding-window folds used ONLY for hyper-parameter search (never touches 2026)."""
    end = L["date"] + pd.Timedelta(days=HORIZON)
    folds = []
    for fit_end, v_start, v_end in [("2024-01-01", "2024-01-01", "2025-01-01"), ("2025-01-01", "2025-01-01", "2026-01-01")]:
        fe, vs, ve = map(pd.Timestamp, (fit_end, v_start, v_end))
        fit = L[end < fe]
        val = L[(L["date"] >= vs) & (end < ve)]
        folds.append((fit, val))
    return folds
