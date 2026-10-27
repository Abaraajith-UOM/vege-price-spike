Repository: https://github.com/Abaraajith-UOM/vege-price-spike

# Early Warning of Weekly Vegetable Price Spikes in Sri Lankan Markets
Applied Machine Learning, Final Assignment (Part 2). Abaraajith, index 269073E.

**Question:** will a vegetable's price in a Colombo-area market rise by 10% or more within the next 7 days?
Binary classification on tabular data (no deep learning, vision or RL).

## Data
- Vegetable prices: *Srilanka-vegetable-prices* (Hugging Face, Rusiru-erandaka), from public Central Bank of Sri Lanka daily price reports. **Licence not stated by the uploader.**
- Rainfall: Open-Meteo (Nuwara Eliya, Polonnaruwa), already inside the CSV.
- Audit finding: 1,339 daily rows but only 733 real reports, so only report days are used (119 series, 24 items).

## Method in one paragraph
Time-ordered split: train 2023-24, validation 2025, test 2026 (opened once). Features are scale-free log-ratios of price now vs 7/14/28 days ago, 4-week volatility, rainfall, month, plus item/market/price type. Models: Random Forest (bagging), XGBoost, LightGBM, CatBoost (boosting), a soft vote, a stacking ensemble (LR meta-learner on time-ordered out-of-fold predictions), logistic regression baseline, all tuned by random search with expanding-window folds; plus ARIMA(7,1,0) baseline and an ARIMA+RF hybrid. SHAP and LIME explain the selected model.

## Headline results (test 2026)
| Model | AUC |
|---|---|
| LightGBM (deployed, chosen on validation) | 0.771 |
| XGBoost / Soft vote / Stacking / Logistic regression | 0.770 / 0.770 / 0.769 / 0.771 |
| CatBoost / ARIMA+RF / Random Forest | 0.762 / 0.766 / 0.754 |
| ARIMA alone / mean-reversion rule | 0.625 / 0.578 |

Honest caveats: boosting only ties logistic regression; an item-base-rate lookup alone gets 0.723, so within-item skill is modest (about 0.65); precision at the alert threshold is about 32%. See notebook sections 7 and 9.

## Run (Windows PowerShell, Python 3.11)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py          # the app (loads model.pkl)
jupyter notebook AML_Final_Assignment.ipynb
```
Retrain everything the app uses: `python src/train_final.py LGBM` (reads `results/threshold.txt`).
Re-tune: `python src/tune.py LGBM 25`.

## Layout
`app.py` front end · `model.pkl` saved model bundle · `src/` prep, models, tune, arima, evaluate, explain, export, train_final, extra_checks · `results/` tuning logs · `data/` input CSV.

## Pickle note
`model.pkl` references code in `src/`, so run the app from this folder. Only unpickle files you trust.

## Front-end prompt used (GenAI)
"Build a small Streamlit app that loads model.pkl, lets the user pick item, market and price type, prefills recent prices and rainfall from real data, runs predict_proba on the raw inputs, and shows the probability and an alert vs the saved threshold, plus a tab with test results and SHAP importance."

## Dataset and how it was obtained
`data/price_dataset.csv` is a copy of the public *Srilanka-vegetable-prices* dataset (Hugging Face, uploader Rusiru-erandaka). It compiles Central Bank of Sri Lanka daily price reports (the file records each row's source PDF) with Open-Meteo rainfall for Nuwara Eliya and Polonnaruwa. No licence is stated by the uploader; if you cannot redistribute it, delete the file and download the original from Hugging Face. The data contain no personal information.

## Map to the marking scheme
| Criterion | Where |
|---|---|
| Problem, data source, features, target, size, preprocessing, ethics | Report sections 1 to 3; notebook sections 1 to 2 |
| New algorithm and why (LightGBM, CatBoost, stacking vs standard models) | Report section 4; notebook section 5 |
| Split, hyper-parameters, metrics, results | Report sections 3 to 5; notebook sections 5 to 7 |
| Explainability (SHAP, LIME, PDP, domain check) | Report section 6; notebook sections 8 and 9b; app "Why this prediction?" |
| Limitations, data quality, bias/fairness, ethics | Report section 8; notebook sections 9 and 9b (`src/fairness.py`) |
| Front end with input, prediction and explanation | `streamlit run app.py` |

## MLOps (lecture 9)
- **Experiment tracking (MLflow):** `python src/mlflow_log.py` logs every tuning candidate and the deployed model (parameters, validation and test metrics, `model.pkl`). View with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- **Docker:** `docker build -t veg-app .` then `docker run -p 8501:8501 veg-app` serves the app at http://localhost:8501. The image uses `requirements.app.txt` (the app only, no notebook packages).
- **Retraining:** `python src/train_final.py LGBM` refits and rewrites `model.pkl`.
- **Deployment:** the Docker image can be pushed to any container host (Azure Container Apps, Hugging Face Spaces, Render, etc.). Add your live URL here once deployed: `<URL>`.
