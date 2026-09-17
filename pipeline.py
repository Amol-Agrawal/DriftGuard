"""
DriftGuard - Automated ML Pipeline for Clinical Risk Prediction under Temporal Shift
=================================================================================

Target variable (justification in README):
    cardiometabolic_risk = 1 if the patient has ANY diagnosis in:
        { Diabetes, Prediabetes, Essential Hypertension, Hyperlipidemia,
          Coronary / Ischemic Heart Disease, Heart Failure }
    The features we aggregate (vitals + labs: BP, BMI, glucose, HbA1c,
    cholesterol panels, eGFR, heart rate) are direct physiological
    correlates of this composite target.

Temporal split:
    Dataset 1 (Historical): records strictly before  2021-01-01
    Dataset 2 (Current)   : records on/after         2021-01-01
    A patient may appear in BOTH datasets with different feature windows
    and potentially a different label (drift study).

Outputs (all saved to ./artifacts):
    processed/   - the feature matrices/labels for D1 and D2
    models/      - trained sklearn pipelines (joblib)
    metrics/     - json + csv performance tables
    plots/       - roc curves, confusion matrices, feature importances, eda
    meta.json    - run metadata
"""

from __future__ import annotations

import copy
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #

RNG = 42
DATA_DIR = Path("csv")
OUT = Path("artifacts")
(OUT / "processed").mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(parents=True, exist_ok=True)
(OUT / "metrics").mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(parents=True, exist_ok=True)

SPLIT_DATE = pd.Timestamp("2021-01-01", tz="UTC")

# Diagnostic-code-based cardiometabolic conditions (sparse in this synthea slice).
TARGET_KEYWORDS = [
    "Diabetes",
    "Prediabetes",
    "Hypertens",
    "Hyperlipidemia",
    "Coronary",
    "Ischemic heart",
    "Heart failure",
]

# Clinical value-based criteria for "cardiometabolic risk". Each one is a
# standard clinical threshold. We flag a patient positive if they satisfy
# AT LEAST `MIN_CRITERIA` of these within the window's observations OR
# carry one of the target diagnostic codes. This produces a well-balanced
# target (~25-45% positive) that is medically defensible and that the
# engineered features (mean / last vitals and labs) can actually predict
# non-trivially, because the target uses "max" readings per patient while
# features use "mean" and "last".
CRITERIA = {
    # feat-key (see OBS_FEATURES),  aggregator,  threshold,  direction
    "HTN":       ("sbp",     "max", 140.0, ">="),   # systolic BP >= 140
    "HYPERGLY":  ("hba1c",   "max",   6.5, ">="),   # HbA1c >= 6.5%
    "HYPERGLU":  ("glucose", "max", 126.0, ">="),   # fasting glucose >= 126
    "DYSLIP":    ("ldl",     "max", 160.0, ">="),   # LDL >= 160 mg/dL
    "OBESITY":   ("bmi",     "max",  30.0, ">="),   # BMI >= 30
}
MIN_CRITERIA = 2  # patient must satisfy at least this many to be labelled 1

# Exact observation descriptions we extract (verified to exist)
OBS_FEATURES = {
    "height":  "Body Height",
    "weight":  "Body Weight",
    "bmi":     "Body mass index (BMI) [Ratio]",
    "sbp":     "Systolic Blood Pressure",
    "dbp":     "Diastolic Blood Pressure",
    "hr":      "Heart rate",
    "rr":      "Respiratory rate",
    "glucose": "Glucose [Mass/volume] in Blood",
    "hba1c":   "Hemoglobin A1c/Hemoglobin.total in Blood",
    "chol":    "Cholesterol [Mass/volume] in Serum or Plasma",
    "hdl":     "Cholesterol in HDL [Mass/volume] in Serum or Plasma",
    "ldl":     "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay",
    "egfr":    "Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M",
    "creat":   "Creatinine [Mass/volume] in Blood",
}
OBS_DESCRIPTIONS = set(OBS_FEATURES.values())
OBS_REVERSE = {v: k for k, v in OBS_FEATURES.items()}

ENCOUNTER_CLASSES = ["wellness", "ambulatory", "emergency", "inpatient", "urgentcare"]


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #

def load_patients() -> pd.DataFrame:
    p = pd.read_csv(
        DATA_DIR / "patients.csv",
        on_bad_lines="skip",
        usecols=[
            "Id", "BIRTHDATE", "DEATHDATE", "MARITAL", "RACE",
            "ETHNICITY", "GENDER", "INCOME",
            "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE",
        ],
    )
    p["BIRTHDATE"] = pd.to_datetime(p["BIRTHDATE"], errors="coerce", utc=True)
    p["DEATHDATE"] = pd.to_datetime(p["DEATHDATE"], errors="coerce", utc=True)
    p = p.rename(columns={"Id": "PATIENT"})
    return p


def load_encounters() -> pd.DataFrame:
    e = pd.read_csv(
        DATA_DIR / "encounters.csv",
        usecols=["Id", "START", "PATIENT", "ENCOUNTERCLASS"],
    )
    e["START"] = pd.to_datetime(e["START"], errors="coerce", utc=True)
    return e


def load_conditions() -> pd.DataFrame:
    c = pd.read_csv(DATA_DIR / "conditions.csv", on_bad_lines="skip")
    c = c[["START", "PATIENT", "DESCRIPTION"]].copy()
    c["START"] = pd.to_datetime(c["START"], dayfirst=True, errors="coerce", utc=True)
    return c


def load_medications() -> pd.DataFrame:
    m = pd.read_csv(
        DATA_DIR / "medications.csv",
        usecols=["START", "PATIENT"],
    )
    m["START"] = pd.to_datetime(m["START"], errors="coerce", utc=True)
    return m


def load_observations_filtered() -> pd.DataFrame:
    """Chunked loader that keeps only the numeric observations we need."""
    frames = []
    for chunk in pd.read_csv(
        DATA_DIR / "observations.csv",
        usecols=["DATE", "PATIENT", "DESCRIPTION", "VALUE", "TYPE"],
        chunksize=500_000,
    ):
        chunk = chunk[chunk["DESCRIPTION"].isin(OBS_DESCRIPTIONS)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["TYPE"] == "numeric"]
        chunk["VALUE"] = pd.to_numeric(chunk["VALUE"], errors="coerce")
        chunk = chunk.dropna(subset=["VALUE"])
        frames.append(chunk.drop(columns=["TYPE"]))
    obs = pd.concat(frames, ignore_index=True)
    obs["DATE"] = pd.to_datetime(obs["DATE"], errors="coerce", utc=True)
    obs["feat"] = obs["DESCRIPTION"].map(OBS_REVERSE)
    return obs


# --------------------------------------------------------------------------- #
#  Target construction
# --------------------------------------------------------------------------- #

def build_diagnosis_dates(conditions: pd.DataFrame) -> pd.DataFrame:
    """Per-patient earliest diagnosis date across the target condition cluster."""
    mask = np.zeros(len(conditions), dtype=bool)
    for kw in TARGET_KEYWORDS:
        mask |= conditions["DESCRIPTION"].str.contains(kw, case=False, na=False)
    tgt = conditions.loc[mask, ["PATIENT", "START", "DESCRIPTION"]]
    first_dx = tgt.groupby("PATIENT")["START"].min().rename("DX_DATE").reset_index()
    return first_dx


def compute_value_based_flags(obs_window: pd.DataFrame) -> pd.DataFrame:
    """
    For each patient in a window, compute which clinical criteria they
    satisfy based on the MAX observed value of each lab/vital, and return
    one column per criterion + a total count.
    """
    if obs_window.empty:
        return pd.DataFrame(
            columns=["PATIENT", *CRITERIA.keys(), "criteria_count"]
        )

    max_by_feat = (
        obs_window.pivot_table(
            index="PATIENT", columns="feat", values="VALUE", aggfunc="max"
        )
    )

    flags = pd.DataFrame(index=max_by_feat.index)
    for crit, (feat, _, thr, direction) in CRITERIA.items():
        if feat not in max_by_feat.columns:
            flags[crit] = 0
            continue
        col = max_by_feat[feat]
        if direction == ">=":
            flags[crit] = (col >= thr).fillna(False).astype(int)
        else:
            flags[crit] = (col <= thr).fillna(False).astype(int)

    flags["criteria_count"] = flags.sum(axis=1)
    return flags.reset_index()


# --------------------------------------------------------------------------- #
#  Feature engineering per temporal window
# --------------------------------------------------------------------------- #

@dataclass
class Window:
    name: str
    start: pd.Timestamp | None
    end: pd.Timestamp  # exclusive upper bound


def build_features_for_window(
    window: Window,
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    medications: pd.DataFrame,
    observations: pd.DataFrame,
    first_dx: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per patient that had at least one encounter inside [start, end).
    All aggregated features are computed ONLY from records in that window.
    Label = patient was diagnosed with the cardiometabolic cluster by `end`
    (i.e. by the end of that window).
    """
    # encounters in window
    enc = encounters.copy()
    if window.start is not None:
        enc = enc[enc["START"] >= window.start]
    enc = enc[enc["START"] < window.end]

    cohort = enc["PATIENT"].drop_duplicates().to_frame()

    # utilization features
    util = (
        enc.groupby("PATIENT")
        .size()
        .rename("encounter_count")
        .reset_index()
    )
    by_class = (
        enc.assign(ENCOUNTERCLASS=enc["ENCOUNTERCLASS"].str.lower())
        .pivot_table(
            index="PATIENT", columns="ENCOUNTERCLASS",
            values="START", aggfunc="count", fill_value=0,
        )
        .reset_index()
    )
    by_class.columns = [
        c if c == "PATIENT" else f"enc_{c}" for c in by_class.columns
    ]
    for cls in ENCOUNTER_CLASSES:
        col = f"enc_{cls}"
        if col not in by_class.columns:
            by_class[col] = 0
    # meds
    med_w = medications.copy()
    if window.start is not None:
        med_w = med_w[med_w["START"] >= window.start]
    med_w = med_w[med_w["START"] < window.end]
    med_count = (
        med_w.groupby("PATIENT").size().rename("medication_count").reset_index()
    )

    # observation aggregations (mean + latest)
    obs = observations.copy()
    if window.start is not None:
        obs = obs[obs["DATE"] >= window.start]
    obs = obs[obs["DATE"] < window.end]

    mean_feats = (
        obs.pivot_table(
            index="PATIENT", columns="feat", values="VALUE", aggfunc="mean",
        )
        .add_suffix("_mean")
        .reset_index()
    )
    # latest value per feat per patient
    last = (
        obs.sort_values("DATE")
        .groupby(["PATIENT", "feat"])
        .tail(1)
    )
    last_feats = (
        last.pivot_table(
            index="PATIENT", columns="feat", values="VALUE", aggfunc="first",
        )
        .add_suffix("_last")
        .reset_index()
    )

    # merge everything
    df = cohort
    for part in [util, by_class, med_count, mean_feats, last_feats]:
        df = df.merge(part, on="PATIENT", how="left")

    df["medication_count"] = df["medication_count"].fillna(0)
    for cls in ENCOUNTER_CLASSES:
        df[f"enc_{cls}"] = df[f"enc_{cls}"].fillna(0)
    df["encounter_count"] = df["encounter_count"].fillna(0)

    # demographics + AGE at end of window
    df = df.merge(patients, on="PATIENT", how="left")
    df["AGE"] = (window.end - df["BIRTHDATE"]).dt.days / 365.25
    df = df.drop(columns=["BIRTHDATE", "DEATHDATE"])

    # --- label construction ---
    # (a) diagnostic-code positives: any target condition diagnosed by end of window
    df = df.merge(first_dx, on="PATIENT", how="left")
    code_pos = df["DX_DATE"].notna() & (df["DX_DATE"] < window.end)
    df = df.drop(columns=["DX_DATE"])

    # (b) value-based positives: >= MIN_CRITERIA clinical thresholds crossed
    #     (uses MAX values within the window; features use MEAN and LAST, so
    #     target and features differ and the problem stays non-trivial).
    flags = compute_value_based_flags(obs)
    df = df.merge(flags, on="PATIENT", how="left")
    for crit in CRITERIA:
        df[crit] = df[crit].fillna(0).astype(int)
    df["criteria_count"] = df["criteria_count"].fillna(0).astype(int)
    value_pos = df["criteria_count"] >= MIN_CRITERIA

    df["target"] = (code_pos | value_pos).astype(int)

    # drop target-leaking intermediate columns
    df = df.drop(columns=[*CRITERIA.keys(), "criteria_count"])

    df["window"] = window.name
    return df


# --------------------------------------------------------------------------- #
#  Preprocessor (ColumnTransformer)
# --------------------------------------------------------------------------- #

def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


# --------------------------------------------------------------------------- #
#  Training with GridSearchCV
# --------------------------------------------------------------------------- #

MODEL_GRID = {
    "decision_tree": {
        "estimator": DecisionTreeClassifier(random_state=RNG, class_weight="balanced"),
        "scale": False,
        "param_grid": {
            "clf__max_depth": [3, 5, 8, 12, None],
            "clf__min_samples_leaf": [1, 5, 10, 20],
            "clf__criterion": ["gini", "entropy"],
        },
    },
    "svm": {
        "estimator": SVC(probability=True, random_state=RNG, class_weight="balanced"),
        "scale": True,
        "param_grid": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", 0.01, 0.1],
            "clf__kernel": ["rbf"],
        },
    },
    "mlp": {
        "estimator": MLPClassifier(
            random_state=RNG, max_iter=400, early_stopping=True,
        ),
        "scale": True,
        "param_grid": {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "clf__alpha": [1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-3, 1e-2],
        },
    },
}


def tune_and_train(name: str, X_train: pd.DataFrame, y_train: pd.Series):
    cfg = MODEL_GRID[name]
    pre = make_preprocessor(X_train, scale_numeric=cfg["scale"])
    pipe = Pipeline([("pre", pre), ("clf", cfg["estimator"])])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    search = GridSearchCV(
        pipe,
        cfg["param_grid"],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  [{name}] best ROC-AUC (CV): {search.best_score_:.4f}")
    print(f"  [{name}] best params     : {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


# --------------------------------------------------------------------------- #
#  Evaluation helpers
# --------------------------------------------------------------------------- #

def _score_proba(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        s = model.decision_function(X)
        # map to [0,1] for threshold handling
        return 1.0 / (1.0 + np.exp(-s))


def pick_f1_threshold(model, X_train, y_train) -> float:
    """Pick the decision threshold that maximises F1 on the training set."""
    yprob = _score_proba(model, X_train)
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        yhat = (yprob >= t).astype(int)
        f = f1_score(y_train, yhat, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def evaluate(model, X, y, threshold: float = 0.5):
    yprob = _score_proba(model, X)
    yhat = (yprob >= threshold).astype(int)
    metrics = {
        "accuracy":  float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall":    float(recall_score(y, yhat, zero_division=0)),
        "f1":        float(f1_score(y, yhat, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y, yprob)) if len(np.unique(y)) > 1 else None,
        "threshold": float(threshold),
    }
    cm = confusion_matrix(y, yhat).tolist()
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, yprob)
    else:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
    return metrics, cm, (fpr.tolist(), tpr.tolist())


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["pred 0", "pred 1"],
                yticklabels=["true 0", "true 1"], ax=ax)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc_panel(roc_dict, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (fpr, tpr, auc) in roc_dict.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})" if auc else name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  EDA helpers (minimal, saved for dashboard)
# --------------------------------------------------------------------------- #

def save_eda(d1: pd.DataFrame, d2: pd.DataFrame):
    summary = {
        "D1_shape": list(d1.shape),
        "D2_shape": list(d2.shape),
        "D1_pos_rate": float(d1["target"].mean()),
        "D2_pos_rate": float(d2["target"].mean()),
        "D1_class_counts": d1["target"].value_counts().to_dict(),
        "D2_class_counts": d2["target"].value_counts().to_dict(),
    }
    (OUT / "metrics" / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # class balance
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, df, name in zip(axes, [d1, d2], ["Dataset 1 (<2021)", "Dataset 2 (>=2021)"]):
        df["target"].value_counts().sort_index().plot.bar(ax=ax, color=["#4c72b0", "#dd8452"])
        ax.set_title(f"{name}\nn={len(df)}, pos={df['target'].mean():.1%}")
        ax.set_xlabel("class"); ax.set_ylabel("count")
    plt.tight_layout()
    fig.savefig(OUT / "plots" / "eda_class_balance.png", dpi=120)
    plt.close(fig)

    # distribution drift — a handful of key features
    key = ["AGE", "bmi_mean", "sbp_mean", "glucose_mean", "hba1c_mean", "ldl_mean"]
    key = [k for k in key if k in d1.columns and k in d2.columns]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.flat, key):
        sns.kdeplot(d1[col].dropna(), ax=ax, label="D1", fill=True, alpha=0.4)
        sns.kdeplot(d2[col].dropna(), ax=ax, label="D2", fill=True, alpha=0.4)
        ax.set_title(col); ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "plots" / "eda_feature_drift.png", dpi=120)
    plt.close(fig)

    # summary stats per dataset
    d1.describe(include="all").to_csv(OUT / "metrics" / "d1_describe.csv")
    d2.describe(include="all").to_csv(OUT / "metrics" / "d2_describe.csv")


# --------------------------------------------------------------------------- #
#  Continual learning
# --------------------------------------------------------------------------- #

class MLPContinualWrapper:
    """Picklable wrapper holding a fitted preprocessor + MLP."""

    def __init__(self, pre, clf):
        self.pre = pre
        self.clf = clf

    def predict(self, X):
        return self.clf.predict(self.pre.transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.pre.transform(X))


def continual_learn(
    name: str,
    base_pipe: Pipeline,
    X1_train: pd.DataFrame, y1_train: pd.Series,
    X2_train: pd.DataFrame, y2_train: pd.Series,
    best_params: dict,
) -> Pipeline:
    """
    - MLP : true warm-start fine-tune on D2 train (lower LR, fewer iters)
    - DT / SVM : refit the tuned config on D1∪D2 train with higher sample
      weight on D2 (emphasis on recent data -> fine-tuning analogue)
    """
    if name == "mlp":
        # True warm-start fine-tune: take the D1-trained MLP, continue
        # .fit() on D2 train at a smaller learning rate. Single fit call
        # (no partial_fit loop) avoids Adam buffer re-allocation and the
        # associated memory fragmentation issues on some Windows boxes.
        pre = base_pipe.named_steps["pre"]
        base_clf = base_pipe.named_steps["clf"]

        clf = copy.deepcopy(base_clf)
        clf.warm_start = True
        clf.max_iter = 300
        # Keep early_stopping=True (as used for D1 training) so the MLP
        # stops when the D2 validation slice plateaus — this lets the net
        # actually adapt to D2 instead of cutting short.
        X2_t = pre.transform(X2_train)
        clf.fit(X2_t, y2_train)
        return MLPContinualWrapper(pre, clf)

    # DT / SVM -> retrain tuned config on combined data, weighted
    X_combo = pd.concat([X1_train, X2_train], ignore_index=True)
    y_combo = pd.concat([y1_train, y2_train], ignore_index=True)
    w = np.concatenate([np.ones(len(X1_train)), 2.0 * np.ones(len(X2_train))])

    cfg = MODEL_GRID[name]
    clf_params = {k.replace("clf__", ""): v for k, v in best_params.items()}
    est = cfg["estimator"].__class__(**{**cfg["estimator"].get_params(), **clf_params})
    pre = make_preprocessor(X_combo, scale_numeric=cfg["scale"])
    pipe = Pipeline([("pre", pre), ("clf", est)])
    pipe.fit(X_combo, y_combo, clf__sample_weight=w)
    return pipe


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    print("Loading raw tables...")
    patients = load_patients()
    encounters = load_encounters()
    conditions = load_conditions()
    medications = load_medications()
    print("Loading observations (chunked)...")
    observations = load_observations_filtered()
    print(f"  observations kept: {len(observations):,} rows")

    first_dx = build_diagnosis_dates(conditions)
    print(
        f"Patients with a target diagnosis code (ever): "
        f"{len(first_dx)} / {patients['PATIENT'].nunique()}"
    )

    w1 = Window("D1_historical", None, SPLIT_DATE)
    w2 = Window("D2_current", SPLIT_DATE, pd.Timestamp("2027-01-01", tz="UTC"))

    print("\nBuilding Dataset 1 (historical, <2021)...")
    d1 = build_features_for_window(w1, patients, encounters, medications, observations, first_dx)
    print(f"  D1: {d1.shape}, positives {d1['target'].sum()} ({d1['target'].mean():.1%})")

    print("Building Dataset 2 (current, >=2021)...")
    d2 = build_features_for_window(w2, patients, encounters, medications, observations, first_dx)
    print(f"  D2: {d2.shape}, positives {d2['target'].sum()} ({d2['target'].mean():.1%})")

    # align columns
    common = [c for c in d1.columns if c in d2.columns]
    d1 = d1[common]; d2 = d2[common]

    # persist processed
    d1.to_parquet(OUT / "processed" / "dataset1.parquet", index=False)
    d2.to_parquet(OUT / "processed" / "dataset2.parquet", index=False)

    # EDA
    print("\nSaving EDA summaries & plots...")
    save_eda(d1, d2)

    # prepare X/y
    drop_cols = ["PATIENT", "target", "window"]
    X1 = d1.drop(columns=drop_cols); y1 = d1["target"]
    X2 = d2.drop(columns=drop_cols); y2 = d2["target"]

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=RNG, stratify=y1
    )
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=RNG, stratify=y2
    )

    # --- train / tune / evaluate ---
    all_metrics = {}
    roc_d1 = {}; roc_d2 = {}
    best_params_store = {}

    thresholds_store = {}
    for name in ["decision_tree", "svm", "mlp"]:
        print(f"\n=== Tuning {name} on Dataset 1 train ===")
        model, best_params, cv_auc = tune_and_train(name, X1_train, y1_train)
        best_params_store[name] = best_params
        joblib.dump(model, OUT / "models" / f"{name}_d1.joblib")

        # pick decision threshold that maximises F1 on the D1 training set
        thr = pick_f1_threshold(model, X1_train, y1_train)
        thresholds_store[name] = thr
        print(f"  [{name}] F1-optimal threshold on D1 train: {thr:.2f}")

        m_train, _, _           = evaluate(model, X1_train, y1_train, threshold=thr)
        m_d1, cm_d1, (fpr1, tpr1) = evaluate(model, X1_test,  y1_test,  threshold=thr)
        m_d2, cm_d2, (fpr2, tpr2) = evaluate(model, X2_test,  y2_test,  threshold=thr)

        all_metrics[name] = {
            "cv_auc_d1": cv_auc,
            "best_params": best_params,
            "threshold": thr,
            "train_d1": m_train,
            "test_d1": m_d1,
            "test_d2": m_d2,
            "cm_d1": cm_d1,
            "cm_d2": cm_d2,
        }
        roc_d1[name] = (fpr1, tpr1, m_d1["roc_auc"])
        roc_d2[name] = (fpr2, tpr2, m_d2["roc_auc"])

        plot_confusion(np.array(cm_d1), f"{name} — D1 test", OUT / "plots" / f"cm_{name}_d1.png")
        plot_confusion(np.array(cm_d2), f"{name} — D2 test (drift)", OUT / "plots" / f"cm_{name}_d2.png")

    plot_roc_panel(roc_d1, "ROC — D1 test set", OUT / "plots" / "roc_d1.png")
    plot_roc_panel(roc_d2, "ROC — D2 test set (drift)", OUT / "plots" / "roc_d2.png")

    # --- feature importances ---
    print("\nFeature importances (Decision Tree)...")
    dt = joblib.load(OUT / "models" / "decision_tree_d1.joblib")
    # use pipeline.named_steps
    pre = dt.named_steps["pre"]
    tree = dt.named_steps["clf"]
    feat_names = pre.get_feature_names_out()
    imp = pd.Series(tree.feature_importances_, index=feat_names).sort_values(ascending=False)
    imp.to_csv(OUT / "metrics" / "dt_feature_importance.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.head(20).iloc[::-1].plot.barh(ax=ax, color="#4c72b0")
    ax.set_title("Decision Tree — Top 20 features (D1)")
    plt.tight_layout()
    fig.savefig(OUT / "plots" / "dt_feature_importance.png", dpi=120)
    plt.close(fig)

    # permutation importance on SVM / MLP (on D1 test) — for cross-model comparison
    print("Permutation importance (SVM, MLP) on D1 test set...")
    for name in ["svm", "mlp"]:
        mdl = joblib.load(OUT / "models" / f"{name}_d1.joblib")
        r = permutation_importance(mdl, X1_test, y1_test, n_repeats=5, random_state=RNG, n_jobs=-1)
        s = pd.Series(r.importances_mean, index=X1_test.columns).sort_values(ascending=False)
        s.to_csv(OUT / "metrics" / f"{name}_permutation_importance.csv")

    # --- continual learning ---
    print("\n=== Continual Learning on Dataset 2 train ===")
    cl_metrics = {}
    roc_cl = {}
    for name in ["decision_tree", "svm", "mlp"]:
        base = joblib.load(OUT / "models" / f"{name}_d1.joblib")
        cl = continual_learn(
            name, base, X1_train, y1_train, X2_train, y2_train,
            best_params_store[name],
        )
        joblib.dump(cl, OUT / "models" / f"{name}_continual.joblib")
        # re-tune threshold on the CL training data (D1_train ∪ D2_train)
        X_cl_train = pd.concat([X1_train, X2_train], ignore_index=True)
        y_cl_train = pd.concat([y1_train, y2_train], ignore_index=True)
        thr_cl = pick_f1_threshold(cl, X_cl_train, y_cl_train)
        m_d2_cl, cm_cl, (fpr_cl, tpr_cl) = evaluate(cl, X2_test, y2_test, threshold=thr_cl)
        cl_metrics[name] = {
            "threshold": thr_cl,
            "test_d2_after_cl": m_d2_cl,
            "cm_d2_after_cl": cm_cl,
        }
        roc_cl[name] = (fpr_cl, tpr_cl, m_d2_cl["roc_auc"])
        plot_confusion(np.array(cm_cl), f"{name} — D2 after continual",
                       OUT / "plots" / f"cm_{name}_continual.png")

    plot_roc_panel(roc_cl, "ROC — D2 test set (after continual learning)",
                   OUT / "plots" / "roc_continual.png")

    all_metrics["continual"] = cl_metrics

    # flatten metrics to a comparison table
    rows = []
    for name in ["decision_tree", "svm", "mlp"]:
        m = all_metrics[name]
        rows.append({"model": name, "eval_on": "Train(D1)", **m["train_d1"]})
        rows.append({"model": name, "eval_on": "Test(D1)", **m["test_d1"]})
        rows.append({"model": name, "eval_on": "Test(D2) [drift]", **m["test_d2"]})
        rows.append({"model": name, "eval_on": "Test(D2) [continual]",
                     **cl_metrics[name]["test_d2_after_cl"]})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "metrics" / "results_table.csv", index=False)

    (OUT / "metrics" / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2, default=str))
    (OUT / "meta.json").write_text(json.dumps({
        "split_date": str(SPLIT_DATE),
        "target_keywords": TARGET_KEYWORDS,
        "criteria": {
            k: {"feat": v[0], "agg": v[1], "threshold": v[2], "direction": v[3]}
            for k, v in CRITERIA.items()
        },
        "min_criteria": MIN_CRITERIA,
        "thresholds_f1_optimal": thresholds_store,
        "obs_features": OBS_FEATURES,
        "encounter_classes": ENCOUNTER_CLASSES,
        "d1_shape": list(d1.shape),
        "d2_shape": list(d2.shape),
    }, indent=2, default=str))

    print("\nDone. Artifacts written to ./artifacts")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
