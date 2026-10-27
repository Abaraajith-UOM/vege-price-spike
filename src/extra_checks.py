"""Supplementary check: how much of the pooled AUC is just item identity? Run: python src/extra_checks.py"""
import sys, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, prep, models, evaluate
from sklearn.metrics import roc_auc_score, average_precision_score
raw = prep.load_wide("data/price_dataset.csv"); L = prep.build_long(raw)
train, val, test = prep.split(L); trval = pd.concat([train, val]); R = prep.RAW_FEATURES
best = {n: evaluate.load_best(n) for n in ["LGBM", "XGB"]}
P = {}
for n in ["LGBM", "XGB"]:
    P[n] = models.make_model(n, best[n]).fit(trval[R], trval.spike).predict_proba(test[R])[:, 1]
P["LR"] = models.make_model("LR", {"C": 1.0, "class_weight": "balanced"}).fit(trval[R], trval.spike).predict_proba(test[R])[:, 1]
P["item base rate only"] = test.item.map(trval.groupby("item").spike.mean()).values
ser = trval.groupby(["item", "market", "price_type"]).spike.mean()
P["series base rate only"] = test.set_index(["item", "market", "price_type"]).index.map(ser).values.astype(float)
F = prep.add_features(test); P["vol4 only"] = F.vol4.values; P["item rate x vol4"] = P["item base rate only"] * F.vol4.values
def within(df, p, min_n=20):
    out = [roc_auc_score(g.spike, g.p) for _, g in df.assign(p=p).groupby("item")
           if g.spike.sum() >= min_n and (g.spike == 0).sum() >= min_n]
    return np.mean(out)
veg = trval.groupby("item").spike.mean(); veg = veg[veg >= 0.05].index; m = test.item.isin(veg).values
rows = {k: {"pooled AUC": roc_auc_score(test.spike, np.asarray(p, float)), "PR-AUC": average_precision_score(test.spike, np.asarray(p, float)),
            "volatile-items AUC": roc_auc_score(test.spike[m], np.asarray(p, float)[m]), "within-item AUC": within(test, np.asarray(p, float))} for k, p in P.items()}
print(pd.DataFrame(rows).T.round(3).to_string())
