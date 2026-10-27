"""Group-level checks on the 2026 test year: does the deployed model work equally well for every market, price type
and item group? Also partial dependence (PDP) curves for the most influential numeric features.
Run: python src/fairness.py   (writes results/fairness_*.csv and results/pdp.png)"""
import pickle, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, prep
from sklearn.metrics import roc_auc_score

def main():
    raw = prep.load_wide(ROOT / "data/price_dataset.csv"); L = prep.build_long(raw)
    train, val, test = prep.split(L); trval = pd.concat([train, val])
    B = pickle.load(open(ROOT / "model.pkl", "rb")); pipe, thr = B["pipeline"], B["threshold"]
    t = test.copy(); t["p"] = pipe.predict_proba(t[prep.RAW_FEATURES])[:, 1]; t["alert"] = t.p >= thr
    rate = trval.groupby("item").spike.mean(); volatile = set(rate[rate >= 0.05].index)
    t["item_group"] = np.where(t.item.isin(volatile), "volatile items", "stable staples (rice etc.)")
    def summarise(g):
        out = {"rows": len(g), "spike rate": g.spike.mean(), "alert rate": g.alert.mean(),
               "recall": (g.alert & (g.spike == 1)).sum() / max(g.spike.sum(), 1),
               "precision": (g.alert & (g.spike == 1)).sum() / max(g.alert.sum(), 1)}
        out["AUC"] = roc_auc_score(g.spike, g.p) if 0 < g.spike.mean() < 1 else np.nan
        return pd.Series(out)
    res = {}
    for col in ["market", "price_type", "item_group"]:
        res[col] = t.groupby(col).apply(summarise)
        res[col].to_csv(ROOT / "results" / f"fairness_{col}.csv")
    # per-item table
    it = t.groupby("item").apply(summarise).sort_values("spike rate", ascending=False); it.to_csv(ROOT / "results" / "fairness_item.csv")
    # PDP on engineered features
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    X = pipe["feat"].transform(train.sample(4000, random_state=0)[prep.RAW_FEATURES]); rest = pipe[1:]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for a, f in zip(ax, ["vol4", "r7", "lp"]):
        grid = np.quantile(X[f], np.linspace(0.02, 0.98, 25)); ys = []
        for v in grid:
            Z = X.copy(); Z[f] = v; ys.append(rest.predict_proba(Z)[:, 1].mean())
        a.plot(grid, ys, color="#1f77b4"); a.set(xlabel=f, ylabel="mean P(spike)", title=f"PDP: {f}")
    plt.tight_layout(); plt.savefig(ROOT / "results" / "pdp.png", dpi=130)
    return res, it

if __name__ == "__main__":
    res, it = main()
    for k, v in res.items(): print("\n", k); print(v.round(3).to_string())
    print("\nper item (top and bottom)"); print(pd.concat([it.head(6), it.tail(4)]).round(3).to_string())
