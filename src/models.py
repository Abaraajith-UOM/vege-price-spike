"""Model factories. Every model is ONE sklearn Pipeline that starts from the RAW user inputs:
raw inputs -> add_features (log-ratios etc.) -> imputation / encoding (/ scaling) -> classifier.
That makes pickling and the front end trivial: the app only passes raw values."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import prep

SEED = 42


def _features():
    return FunctionTransformer(prep.add_features, validate=False)


def _encode(scale=False):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), prep.NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), prep.CATEGORICAL),
    ])


def make_model(name, params=None):
    """name in {LR, RF, XGB, LGBM, CAT}. `params` are the hyper-parameters (NOT library defaults)."""
    params = dict(params or {})
    if name == "LR":
        clf = LogisticRegression(max_iter=2000, **params)
        return Pipeline([("feat", _features()), ("enc", _encode(scale=True)), ("clf", clf)])
    if name == "RF":
        clf = RandomForestClassifier(n_jobs=2, random_state=SEED, **params)
    elif name == "XGB":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_jobs=2, random_state=SEED, eval_metric="logloss", tree_method="hist", **params)
    elif name == "LGBM":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_jobs=2, random_state=SEED, verbose=-1, subsample_freq=1, **params)
    elif name == "CAT":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(random_seed=SEED, thread_count=2, verbose=0, cat_features=prep.CATEGORICAL,
                                 allow_writing_files=False, **params)
        return Pipeline([("feat", _features()), ("clf", clf)])      # CatBoost encodes categories natively
    else:
        raise ValueError(name)
    return Pipeline([("feat", _features()), ("enc", _encode()), ("clf", clf)])


# Library defaults, used ONLY to show how much tuning changes (the assignment forbids submitting defaults).
DEFAULTS = {"RF": {}, "XGB": {}, "LGBM": {"subsample": 1.0}, "CAT": {}}

SPACES = {
    "RF": {"n_estimators": [200, 300, 400], "max_depth": [6, 8, 10, 14, 20], "min_samples_leaf": [10, 25, 50, 100, 200],
           "max_features": ["sqrt", 0.3, 0.5], "class_weight": ["balanced_subsample", None]},
    "XGB": {"n_estimators": [200, 400, 600], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 4, 6, 8],
            "min_child_weight": [1, 5, 10, 25], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [1, 5, 10]},
    "LGBM": {"n_estimators": [200, 400, 600], "learning_rate": [0.03, 0.05, 0.1], "num_leaves": [15, 31, 63],
             "max_depth": [-1, 6, 10], "min_child_samples": [20, 50, 100, 200], "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0, 1, 5, 10]},
    "CAT": {"iterations": [300, 500, 800], "depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1], "l2_leaf_reg": [1, 3, 5, 10]},
}
N_ITER = {"RF": 15, "XGB": 25, "LGBM": 25, "CAT": 10}


def make_hybrid(params):
    """ARIMA + Random Forest hybrid: the RF also sees the ARIMA(7,1,0) 7-day forecast (`arima_fc7`)."""
    import arima
    enc = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), prep.NUMERIC + ["arima_fc7"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), prep.CATEGORICAL),
    ])
    clf = RandomForestClassifier(n_jobs=2, random_state=SEED, **params)
    return Pipeline([("feat", FunctionTransformer(arima.add_features_arima, validate=False)), ("enc", enc), ("clf", clf)])


class SoftVote:
    """Soft-voting ensemble of fitted pipelines (the lecturer's 'voting=soft' idea): average the probabilities."""

    def __init__(self, pipes):
        self.pipes = list(pipes)

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.pipes], axis=0)
        return np.column_stack([1 - p, p])


class Stack:
    """Stacking ensemble (level-0: tuned RF, XGBoost, LightGBM, CatBoost; level-1: logistic regression on their
    out-of-fold probabilities). Out-of-fold predictions are TIME-ORDERED: for each year Y after the first, the base
    models are fitted on rows whose target window ends before 1 Jan Y and predict year Y, so the meta-learner never
    sees a prediction made with future data. Finally the base models are refitted on all the fitting data.
    fit(df) needs the `date` and `spike` columns; predict_proba(X) needs only the raw features (app-safe)."""

    def __init__(self, params, names=("RF", "XGB", "LGBM", "CAT")):
        self.params, self.names = params, list(names)

    def _base(self, n):
        return make_model(n, self.params[n])

    def fit(self, df):
        raw = prep.RAW_FEATURES
        end = df["date"] + np.timedelta64(prep.HORIZON, "D")
        years = sorted(df["date"].dt.year.unique())
        oof, ys = [], []
        for y in years[1:]:
            start, stop = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
            fit = df[end < start]
            blk = df[(df["date"] >= start) & (end < stop)]
            if len(fit) == 0 or len(blk) == 0:
                continue
            oof.append(np.column_stack([self._base(n).fit(fit[raw], fit[prep.TARGET]).predict_proba(blk[raw])[:, 1]
                                        for n in self.names]))
            ys.append(blk[prep.TARGET].values)
        Z, y_ = np.vstack(oof), np.concatenate(ys)
        self.meta = LogisticRegression(C=1.0, max_iter=1000).fit(self._logit(Z), y_)
        self.oof_rows = len(y_)
        self.base = [self._base(n).fit(df[raw], df[prep.TARGET]) for n in self.names]
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def predict_proba(self, X):
        Z = np.column_stack([m.predict_proba(X)[:, 1] for m in self.base])
        p = self.meta.predict_proba(self._logit(Z))[:, 1]
        return np.column_stack([1 - p, p])
