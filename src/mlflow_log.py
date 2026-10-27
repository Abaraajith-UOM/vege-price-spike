"""MLflow experiment tracking (lecture 9). Logs every tuning candidate as a child run and the deployed model as the
final run (parameters, validation and test metrics, threshold, model.pkl artifact).
Run:  python src/mlflow_log.py      then:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
import mlflow, pandas as pd, models

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("vegetable-price-spike")
NAMES = {"RF": "Random Forest", "XGB": "XGBoost", "LGBM": "LightGBM", "CAT": "CatBoost"}

with mlflow.start_run(run_name="tuning-search"):
    for key in models.SPACES:
        df = pd.read_csv(ROOT / "results" / f"tune_{key}.csv")
        params = [c for c in df.columns if c in models.SPACES[key]]
        for i, r in df.iterrows():
            with mlflow.start_run(run_name=f"{key}-{i}", nested=True):
                mlflow.set_tags({"stage": "tuning", "model": NAMES[key]})
                mlflow.log_params({k: ("None" if pd.isna(r[k]) else r[k]) for k in params})
                mlflow.log_metrics({"val_auc_mean": r.mean_auc, "val_auc_std": r.std_auc, "val_pr_auc_mean": r.mean_pr_auc,
                                    "auc_2024": r.auc_2024, "auc_2025": r.auc_2025})

B = pickle.load(open(ROOT / "model.pkl", "rb"))
with mlflow.start_run(run_name="deployed-model"):
    mlflow.set_tags({"stage": "deployed", "model": B["model_name"], "target": "price rise >= 10% within 7 days"})
    mlflow.log_params({**B["params"], "alert_threshold": B["threshold"]})
    for k, v in B["report"][B["model_name"]].items():
        mlflow.log_metric(f"test_{k}", v)
    mlflow.log_artifact(str(ROOT / "model.pkl"))
    mlflow.log_artifact(str(ROOT / "results" / "pdp.png"))
print("logged. View with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
