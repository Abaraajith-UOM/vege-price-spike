"""Randomised hyper-parameter search with TIME-ORDERED validation (never touches the 2026 test set).

usage:  python src/tune.py RF|XGB|LGBM|CAT|defaults [n_iter]

Folds (expanding window):  fit on 2023 -> validate 2024;   fit on 2023-24 -> validate 2025.
Score = mean ROC-AUC across the two folds (PR-AUC also recorded).
"""
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler

import models
import prep

L = pd.read_pickle(ROOT / "results" / "long.pkl")
FOLDS = prep.tuning_folds(L)
T = prep.TARGET
RAW = prep.RAW_FEATURES


def score(name, params):
    roc, pr = [], []
    for fit, val in FOLDS:
        m = models.make_model(name, params).fit(fit[RAW], fit[T])
        p = m.predict_proba(val[RAW])[:, 1]
        roc.append(roc_auc_score(val[T], p)); pr.append(average_precision_score(val[T], p))
    return roc, pr


if __name__ == "__main__":
    which = sys.argv[1]
    t0 = time.time()
    if which == "defaults":
        rows = []
        for name, params in models.DEFAULTS.items():
            roc, pr = score(name, params)
            rows.append(dict(model=name, auc_2024=roc[0], auc_2025=roc[1], mean_auc=np.mean(roc), mean_pr_auc=np.mean(pr)))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / "tune_defaults.csv", index=False)
    else:
        n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else models.N_ITER[which]
        rows = []
        for i, p in enumerate(ParameterSampler(models.SPACES[which], n_iter, random_state=0)):
            roc, pr = score(which, p)
            rows.append({**p, "mean_auc": np.mean(roc), "std_auc": np.std(roc), "mean_pr_auc": np.mean(pr),
                         "auc_2024": roc[0], "auc_2025": roc[1]})
            print(f"{which} {i + 1}/{n_iter} auc={np.mean(roc):.4f} pr={np.mean(pr):.4f}  ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).sort_values("mean_auc", ascending=False).to_csv(ROOT / "results" / f"tune_{which}.csv", index=False)
    print(which, "done in", round(time.time() - t0), "s")
