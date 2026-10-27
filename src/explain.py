"""SHAP and LIME helpers that work for every model in the project (one-hot models and CatBoost).

SHAP values are additive, so the one-hot columns of a categorical feature (item, market, price_type) are summed
back into ONE contribution per original feature. That makes the plots readable and the totals still exact.
"""
import numpy as np
import pandas as pd
import shap

import prep


def _positive_class(sv):
    """Normalise the different shapes shap returns for a binary classifier to (n_rows, n_features)."""
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv


def shap_by_feature(pipe, raw_df):
    """Return (SHAP values per ORIGINAL feature [DataFrame], engineered feature table, base value in log-odds)."""
    X = pipe["feat"].transform(raw_df)                       # engineered features (DataFrame)
    clf = pipe[-1]
    if clf.__class__.__name__ == "CatBoostClassifier":
        from catboost import Pool
        pool = Pool(X, cat_features=prep.CATEGORICAL)
        sv = clf.get_feature_importance(pool, type="ShapValues")
        base = float(sv[0, -1])
        agg = pd.DataFrame(sv[:, :-1], columns=list(X.columns), index=X.index)
        return agg, X, base
    enc = pipe["enc"]
    Xt = enc.transform(X)
    names = list(enc.get_feature_names_out())
    explainer = shap.TreeExplainer(clf)
    sv = _positive_class(explainer.shap_values(Xt))
    base = explainer.expected_value
    base = float(np.ravel(base)[-1])
    cols = {}
    for j, n in enumerate(names):
        orig = n.split("__", 1)[1]
        for c in prep.CATEGORICAL:                           # "item_Beans" -> "item"
            if orig.startswith(c + "_"):
                orig = c
        cols.setdefault(orig, np.zeros(len(X)))
        cols[orig] = cols[orig] + sv[:, j]
    agg = pd.DataFrame(cols, index=X.index)[list(X.columns)]
    return agg, X, base


class LimeHelper:
    """LIME on the engineered feature table. Categorical columns are label-encoded for LIME and decoded
    back to text before calling the model, so every model type (including CatBoost) works the same way."""

    def __init__(self, pipe, train_raw, seed=0):
        from lime.lime_tabular import LimeTabularExplainer
        self.pipe = pipe
        self.rest = pipe[1:]                                  # everything after the feature step
        X = pipe["feat"].transform(train_raw)
        self.cats = {c: sorted(X[c].unique()) for c in prep.CATEGORICAL}
        self.cols = list(X.columns)
        Xl = self.encode(X)
        self.explainer = LimeTabularExplainer(
            Xl.values, feature_names=self.cols, class_names=["no spike", "spike"], mode="classification",
            categorical_features=[self.cols.index(c) for c in prep.CATEGORICAL],
            categorical_names={self.cols.index(c): self.cats[c] for c in prep.CATEGORICAL},
            discretize_continuous=True, random_state=seed)

    def encode(self, X):
        Xl = X.copy()
        for c, vals in self.cats.items():
            Xl[c] = Xl[c].map({v: i for i, v in enumerate(vals)}).fillna(0).astype(int)
        return Xl[self.cols].astype(float)

    def decode(self, arr):
        df = pd.DataFrame(arr, columns=self.cols)
        for c, vals in self.cats.items():
            idx = np.clip(np.rint(df[c].values).astype(int), 0, len(vals) - 1)
            df[c] = np.array(vals)[idx]
        return df

    def predict_fn(self, arr):
        return self.rest.predict_proba(self.decode(arr))

    def explain(self, raw_row, num_features=8, num_samples=3000):
        X = self.pipe["feat"].transform(raw_row)
        row = self.encode(X).values[0]
        return self.explainer.explain_instance(row, self.predict_fn, num_features=num_features, num_samples=num_samples)
