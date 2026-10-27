"""Build and save the deployable bundle (one pickle) used by the Streamlit app."""
import pickle
from pathlib import Path

import prep

ROOT = Path(__file__).resolve().parent.parent


def latest_anchors(L):
    """Most recent real anchor prices per series, used to pre-fill the app with realistic values."""
    last = L.sort_values("date").groupby(["item", "market", "price_type"]).tail(1)
    return {(r.item, r.market, r.price_type): {
        "date": str(r.date.date()), "price_now": float(r.price_now), "price_7d_ago": float(r.price_7d_ago),
        "price_14d_ago": float(r.price_14d_ago), "price_28d_ago": float(r.price_28d_ago),
        "rain_ne_7d_mm": float(r.rain_ne_7d_mm), "rain_pol_7d_mm": float(r.rain_pol_7d_mm)} for r in last.itertuples()}


def save_bundle(fitted, name, params, threshold, L, report=None, importance=None, path=None):
    """`fitted` is any object with predict_proba(raw_df) (a Pipeline or SoftVote)."""
    bundle = {
        "pipeline": fitted, "model_name": name, "params": params, "threshold": float(threshold),
        "raw_features": prep.RAW_FEATURES, "items": sorted(L["item"].unique()),
        "markets": sorted(L["market"].unique()), "price_types": sorted(L["price_type"].unique()),
        "latest": latest_anchors(L), "report": report, "shap_importance": importance,
        "spike_threshold": prep.SPIKE, "horizon_days": prep.HORIZON,
        "item_spike_rate": L[L["date"] < prep.VAL_END].groupby("item")[prep.TARGET].mean().round(4).to_dict(),
        "note": "Needs src/prep.py and src/models.py importable (custom code inside the pipeline).",
    }
    path = Path(path or ROOT / "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return bundle, path
