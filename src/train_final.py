"""One-command retraining (the MLOps step): rebuild features from the CSV, refit the chosen model with the saved
tuned parameters, score it on the newest period, and rewrite model.pkl.

usage:  python src/train_final.py [MODEL]      # MODEL in RF|XGB|LGBM|CAT (default: LGBM)
Set --all to fit on ALL data (what you would deploy) instead of holding the last year out for a quick check.
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import evaluate
import export
import models
import prep

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="LGBM", choices=["RF", "XGB", "LGBM", "CAT"])
ap.add_argument("--all", action="store_true", help="fit on all data (deployment) instead of holding out 2026")
ap.add_argument("--threshold", type=float, default=None, help="alert threshold (default: read from results/threshold.txt or 0.30)")
a = ap.parse_args()

raw = prep.load_wide(ROOT / "data" / "price_dataset.csv")
L = prep.build_long(raw)
train, val, test = prep.split(L)
params = evaluate.load_best(a.model)
fit_df = L if a.all else pd.concat([train, val])
pipe = models.make_model(a.model, params).fit(fit_df[prep.RAW_FEATURES], fit_df[prep.TARGET])
if not a.all and len(test):
    p = pipe.predict_proba(test[prep.RAW_FEATURES])[:, 1]
    print(f"newest-period check ({test.date.min().date()} to {test.date.max().date()}): "
          f"AUC={roc_auc_score(test.spike, p):.3f}  PR-AUC={average_precision_score(test.spike, p):.3f}")
thr_file = ROOT / "results" / "threshold.txt"
thr = a.threshold if a.threshold is not None else (float(thr_file.read_text()) if thr_file.exists() else 0.30)
_, path = export.save_bundle(pipe, a.model, params, thr, L, path=ROOT / "model.pkl")
print("saved", path, "| model", a.model, "| threshold", thr)
