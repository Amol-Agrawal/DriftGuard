# DriftGuard

End-to-end ML pipeline for clinical risk prediction under temporal distribution shift, with drift-aware retraining and a Streamlit monitoring dashboard.

Predicts cardiometabolic risk from raw clinical records, then keeps the models honest when the data distribution moves: feature engineering across 17 source datasets, GridSearchCV-tuned Decision Tree / SVM / MLP, cross-period evaluation, continual learning, and a dashboard for drift and performance monitoring.

## What this repository contains
| File | Purpose |
| --- | --- |
| `pipeline.py` | Full ML pipeline — loads the 17 CSVs in `csv/`, engineers features, builds Dataset 1 (pre-2021) and Dataset 2 (2021+), tunes + trains Decision Tree / SVM / MLP with `GridSearchCV`, performs cross-dataset evaluation, runs continual learning on D2, and writes every artifact to `./artifacts`. |
| `dashboard.py` | Streamlit dashboard that loads the artifacts and presents EDA & drift, model performance, bias/variance, feature importance, and continual-learning results. No training happens in the dashboard — it is purely a reader of the artifacts. |
| `requirements.txt` | Python dependencies. |
| `csv/` | The 17 raw Synthea-style CSVs (provided). |
| `artifacts/` | Produced by `pipeline.py` — processed datasets, trained models, metrics, plots. |

## Target variable (justification)
We predict a **composite cardiometabolic risk flag**. A patient is labelled `1` if **either**

1. their `conditions.csv` record contains any diagnosis of **Diabetes / Prediabetes / Hypertension / Hyperlipidemia / Coronary‑Ischemic Heart Disease / Heart Failure**, **or**
2. their **observations within the window** satisfy at least **2 of 5** standard clinical cardiometabolic thresholds:

   | Criterion | Rule (using max observation in window) |
   | --- | --- |
   | Hypertension | Systolic BP ≥ 140 mmHg |
   | Hyperglycaemia (A1c) | HbA1c ≥ 6.5 % |
   | Hyperglycaemia (fasting glucose) | Glucose ≥ 126 mg/dL |
   | Dyslipidaemia | LDL ≥ 160 mg/dL |
   | Obesity | BMI ≥ 30 |

### Why both sources?
The Synthea slice used here has `conditions.csv` records for **only ~108 / 2 825 patients** (≈ 2 % positive) — too sparse to train/validate three classifiers reliably. Adding value‑based criteria (using accepted clinical thresholds) recovers a medically defensible, workable positive rate of ~25–45 %. Arrhythmia / Stroke / COPD were dropped because they have 0 rows in this slice.

### Why the target does *not* leak into the features
- **Target** uses the **maximum** observed value of each lab / vital in the window (“did the patient ever cross the threshold?”).
- **Features** use the **mean and the latest** value per patient.

These are systematically different statistics — a patient with one spike and an otherwise normal mean satisfies the target but the `*_mean` / `*_last` features are near‑normal. The classifiers therefore need to *learn* the relationship from the feature aggregates, demographics, and utilisation — it is not trivially recoverable.

### Decision threshold tuning
Because the positive class is still the minority, for each model we pick the decision threshold that **maximises F1 on the D1 training set**. For continual‑learning models the threshold is re‑picked on `D1_train ∪ D2_train`. All reported precision, recall, F1 values use these tuned thresholds; ROC‑AUC is threshold‑independent.

## Temporal split
`2021-01-01` (UTC). Both Dataset 1 and Dataset 2 may contain the same patient — with a **different feature window** (pre-2021 vs 2021+ records) and possibly a **different label** if the patient was diagnosed during the 2021–2026 window. This is what produces the real distribution shift we want to study.

## Feature engineering
For every patient × window:

- **Demographics**: `AGE` (computed at window end), `GENDER`, `RACE`, `ETHNICITY`, `MARITAL`, `INCOME`, `HEALTHCARE_EXPENSES`, `HEALTHCARE_COVERAGE`.
- **Utilisation**: `encounter_count`, `medication_count`, plus counts per encounter class (`wellness`, `ambulatory`, `emergency`, `inpatient`, `urgentcare`).
- **Vitals / labs** — `mean` and `last` values per patient within the window, for: Body Height, Body Weight, BMI, Systolic BP, Diastolic BP, Heart rate, Respiratory rate, Glucose, HbA1c, Cholesterol, HDL, LDL, eGFR, Creatinine.

All numeric features go through `SimpleImputer(strategy="median")`; for **SVM and MLP** they are additionally `StandardScaler`-scaled **inside the sklearn `Pipeline`**, so CV folds never leak. Categoricals are median-imputed and one-hot encoded.

## Models & hyper-parameter tuning
Each model is a `Pipeline(preprocessor → classifier)` tuned with `GridSearchCV`, `StratifiedKFold(5)`, scoring `roc_auc`.

| Model | Grid |
| --- | --- |
| Decision Tree | `max_depth ∈ {3,5,8,12,None}`, `min_samples_leaf ∈ {1,5,10,20}`, `criterion ∈ {gini,entropy}` |
| SVM (RBF) | `C ∈ {0.1,1,10}`, `gamma ∈ {scale,0.01,0.1}` |
| MLP | `hidden_layer_sizes ∈ {(32,),(64,),(64,32)}`, `alpha ∈ {1e-4,1e-3,1e-2}`, `learning_rate_init ∈ {1e-3,1e-2}` |

`class_weight="balanced"` is used everywhere it is supported.

## Evaluation
- Train on **D1 train** → evaluate on both **D1 test** *and* **D2 test**.
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curve.
- Train-vs-test ROC-AUC gap is used to discuss bias/variance.
- Decision Tree: built-in `feature_importances_`; SVM & MLP: permutation importance on D1 test.

## Continual learning
- **MLP** — true warm-start fine-tuning via `partial_fit` on D2 train at 0.1× the tuned initial learning rate.
- **DT / SVM** — fine-tuning analogue: refit the tuned config on `D1_train ∪ D2_train` with a 2× `sample_weight` on D2 rows (emphasis on recent data).

Evaluated on D2 test and compared against the plain-D1 model.

## How to run

```bash
# one-time
python -m pip install -r requirements.txt

# 1. run the full training + evaluation + continual-learning pipeline
python pipeline.py
# -> writes ./artifacts/{processed,models,metrics,plots,meta.json}

# 2. launch the dashboard
streamlit run dashboard.py
```

The dashboard does **not** retrain anything — it only reads `./artifacts`, so it launches instantly.

## Reproducibility
All random seeds fixed to `42`. `GridSearchCV` uses a fixed `StratifiedKFold`. Rerunning `pipeline.py` will produce identical artifacts on the same CSV inputs.

## Credits

Built by Rishit Toshniwal, Shashank Nelanti, Amol Agrawal, and Rudram Pingale.
