"""Small helpers used by the notebook: load tuned parameters, fit/predict, choose a threshold, score."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

import models
import prep

ROOT = Path(__file__).resolve().parent.parent
RAW, T = prep.RAW_FEATURES, prep.TARGET


def _clean(v):
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else (int(v) if float(v).is_integer() and abs(v) > 1 else float(v))
    return v


PARAM_COLS = {k: list(v) for k, v in models.SPACES.items()}


def load_best(name):
    """Best row of the saved search -> clean parameter dict."""
    row = pd.read_csv(ROOT / "results" / f"tune_{name}.csv").iloc[0]
    return {k: _clean(row[k]) for k in PARAM_COLS[name]}


def fit_predict(name, params, fit_df, pred_df, hybrid=False):
    m = models.make_hybrid(params) if hybrid else models.make_model(name, params)
    cols = RAW + (["arima_fc7"] if hybrid else [])
    m.fit(fit_df[cols], fit_df[T])
    return m, m.predict_proba(pred_df[cols])[:, 1]


def best_threshold(y, p):
    """Threshold that maximises F1 (chosen on VALIDATION predictions only)."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    return float(th[np.argmax(f1)])


def score_row(y, p, thr=None):
    r = {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p)}
    if thr is not None:
        yh = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
        r.update(threshold=thr, accuracy=accuracy_score(y, yh), bal_acc=balanced_accuracy_score(y, yh), recall=recall_score(y, yh), precision=precision_score(y, yh), F1=f1_score(y, yh),
                 TP=tp, FP=fp, FN=fn, TN=tn)
    return r


def date_bootstrap(y, probs, dates, n_boot=300, seed=0):
    """Block bootstrap over REPORT DAYS (all series on a day move together, so rows are not independent).
    Returns {name: (auc_low, auc_high)} and, for each pair vs the first model, the CI of the AUC difference."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); dates = np.asarray(dates)
    days = np.unique(dates)
    idx_by_day = {d: np.where(dates == d)[0] for d in days}
    names = list(probs)
    aucs = {n: [] for n in names}
    for _ in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in pick])
        if y[idx].min() == y[idx].max():
            continue
        for n in names:
            aucs[n].append(roc_auc_score(y[idx], np.asarray(probs[n])[idx]))
    ci = {n: tuple(np.percentile(a, [2.5, 97.5])) for n, a in aucs.items()}
    diff = {n: tuple(np.percentile(np.array(aucs[n]) - np.array(aucs[names[0]]), [2.5, 97.5])) for n in names[1:]}
    return ci, diff
