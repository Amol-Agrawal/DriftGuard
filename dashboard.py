"""
dashboard.py
Streamlit dashboard for the Automated ML Pipeline under Temporal Shift.

Loads artifacts produced by pipeline.py and presents them across clean tabs:
    - Overview
    - EDA & Data Drift
    - Model Performance
    - Bias / Variance & Complexity
    - Feature Importance
    - Continual Learning
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = Path("artifacts")

st.set_page_config(
    page_title="Clinical Prediction under Temporal Shift",
    layout="wide",
)

# --------------------------------------------------------------------------- #
#  Loaders (cached)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_json(path: Path):
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def load_csv(path: Path, **kw):
    return pd.read_csv(path, **kw)


@st.cache_data(show_spinner=False)
def load_parquet(path: Path):
    return pd.read_parquet(path)


meta = load_json(ART / "meta.json")
eda = load_json(ART / "metrics" / "eda_summary.json")
metrics = load_json(ART / "metrics" / "all_metrics.json")
results = load_csv(ART / "metrics" / "results_table.csv")

d1 = load_parquet(ART / "processed" / "dataset1.parquet")
d2 = load_parquet(ART / "processed" / "dataset2.parquet")

# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #

TEAM = "DriftGuard"
st.title("DriftGuard — Clinical Risk Prediction under Temporal Shift")
st.caption(
    f"{TEAM} · Target: Cardiometabolic Risk (composite) · "
    f"Temporal split: {meta['split_date']}"
)

tab_overview, tab_eda, tab_perf, tab_bv, tab_imp, tab_cl = st.tabs(
    ["Overview", "EDA & Drift", "Model Performance",
     "Bias / Variance", "Feature Importance", "Continual Learning"]
)

# --------------------------------------------------------------------------- #
#  Overview
# --------------------------------------------------------------------------- #

with tab_overview:
    st.subheader("Pipeline summary")
    st.markdown(
        f"""
**Target variable** — `cardiometabolic_risk` (binary). A patient is labelled `1`
if **either**
1. their `conditions` record contains any of: *Diabetes, Prediabetes,
   Hypertension, Hyperlipidemia, Coronary / Ischemic Heart Disease, Heart
   Failure*; **or**
2. their observations within the window cross at least **{meta['min_criteria']}
   of {len(meta['criteria'])}** standard clinical cardiometabolic thresholds
   (see below).

Using value-based criteria alongside diagnostic codes gives a well-balanced,
medically defensible label (~25–45 % positive). Because the target uses **max**
values per patient while the features are **mean** and **last** values, the
target does not leak trivially into the features.

**Temporal split** — `{meta['split_date']}`. Each patient may appear in **both**
datasets with a different feature window and possibly a different label — the
drift we study.

**Decision thresholds** — chosen by maximising F1 on the D1 training set
(standard for imbalanced classification): {meta.get('thresholds_f1_optimal', {})}.
"""
    )
    with st.expander("Clinical criteria used for the value-based target"):
        crit_rows = [
            {"criterion": k, **v} for k, v in meta["criteria"].items()
        ]
        st.dataframe(pd.DataFrame(crit_rows), width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("D1 patients (<2021)", d1.shape[0])
    c2.metric("D2 patients (≥2021)", d2.shape[0])
    c3.metric("D1 positive rate", f"{eda['D1_pos_rate']:.2%}")
    c4.metric("D2 positive rate", f"{eda['D2_pos_rate']:.2%}")

    st.markdown("**Feature groups**")
    st.json({
        "Demographics": [
            "AGE (computed at window end)", "GENDER", "RACE", "ETHNICITY",
            "MARITAL", "INCOME", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE",
        ],
        "Utilization": [
            "encounter_count", "medication_count",
            *[f"enc_{c}" for c in meta["encounter_classes"]],
        ],
        "Vitals & Labs (mean + latest)": list(meta["obs_features"].keys()),
    })

    st.markdown("**Models & hyperparameter grids (GridSearchCV, 5-fold stratified, ROC-AUC)**")
    st.json({m: metrics[m]["best_params"] for m in ["decision_tree", "svm", "mlp"]})

# --------------------------------------------------------------------------- #
#  EDA & Drift
# --------------------------------------------------------------------------- #

with tab_eda:
    st.subheader("Class balance")
    cb = pd.DataFrame({
        "dataset": ["D1", "D1", "D2", "D2"],
        "class":   ["0", "1", "0", "1"],
        "count":   [
            eda["D1_class_counts"].get("0", 0) + eda["D1_class_counts"].get(0, 0),
            eda["D1_class_counts"].get("1", 0) + eda["D1_class_counts"].get(1, 0),
            eda["D2_class_counts"].get("0", 0) + eda["D2_class_counts"].get(0, 0),
            eda["D2_class_counts"].get("1", 0) + eda["D2_class_counts"].get(1, 0),
        ],
    })
    fig = px.bar(cb, x="dataset", y="count", color="class", barmode="group",
                 title="Class distribution per dataset")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Feature distribution drift (D1 vs D2)")
    numeric_cols = [
        c for c in d1.select_dtypes(include=np.number).columns
        if c not in ("target",)
    ]
    default = [c for c in ["AGE", "bmi_mean", "sbp_mean",
                           "glucose_mean", "hba1c_mean", "ldl_mean"] if c in numeric_cols]
    picks = st.multiselect("Features", numeric_cols, default=default)
    ncols = 2
    for i in range(0, len(picks), ncols):
        cols = st.columns(ncols)
        for j, col in enumerate(picks[i:i + ncols]):
            with cols[j]:
                s1 = d1[col].dropna()
                s2 = d2[col].dropna()
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=s1, name="D1", opacity=0.55,
                                           histnorm="probability density", nbinsx=40))
                fig.add_trace(go.Histogram(x=s2, name="D2", opacity=0.55,
                                           histnorm="probability density", nbinsx=40))
                fig.update_layout(barmode="overlay", title=col, height=280,
                                  margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch")

    st.subheader("Numeric summary")
    left, right = st.columns(2)
    with left:
        st.caption("Dataset 1 (pre-2021)")
        st.dataframe(d1.describe().T.round(2), width="stretch", height=350)
    with right:
        st.caption("Dataset 2 (2021+)")
        st.dataframe(d2.describe().T.round(2), width="stretch", height=350)

# --------------------------------------------------------------------------- #
#  Performance
# --------------------------------------------------------------------------- #

with tab_perf:
    st.subheader("Cross-dataset evaluation")
    st.caption(
        "Models are tuned on **Dataset 1 train**, then evaluated on Dataset 1 "
        "test *and* Dataset 2 test (temporal drift)."
    )

    view = results.copy()
    view["accuracy"]  = view["accuracy"].round(3)
    view["precision"] = view["precision"].round(3)
    view["recall"]    = view["recall"].round(3)
    view["f1"]        = view["f1"].round(3)
    view["roc_auc"]   = view["roc_auc"].round(3)
    st.dataframe(view, width="stretch")

    st.subheader("ROC curves")
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(ART / "plots" / "roc_d1.png"), caption="D1 test — trained on D1")
    with c2:
        st.image(str(ART / "plots" / "roc_d2.png"), caption="D2 test — trained on D1 (drift)")

    st.subheader("Confusion matrices")
    for name in ["decision_tree", "svm", "mlp"]:
        st.markdown(f"**{name}**")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.image(str(ART / "plots" / f"cm_{name}_d1.png"), caption=f"{name} · D1 test")
        with cc2:
            st.image(str(ART / "plots" / f"cm_{name}_d2.png"), caption=f"{name} · D2 test")

# --------------------------------------------------------------------------- #
#  Bias / Variance
# --------------------------------------------------------------------------- #

with tab_bv:
    st.subheader("Train vs test performance — bias / variance gap")
    rows = []
    for name in ["decision_tree", "svm", "mlp"]:
        m = metrics[name]
        rows.append({
            "model": name,
            "train_auc_d1": m["train_d1"]["roc_auc"],
            "test_auc_d1":  m["test_d1"]["roc_auc"],
            "gap_d1":       (m["train_d1"]["roc_auc"] or 0) - (m["test_d1"]["roc_auc"] or 0),
            "test_auc_d2":  m["test_d2"]["roc_auc"],
            "cv_auc_d1":    m["cv_auc_d1"],
        })
    bv = pd.DataFrame(rows).round(3)
    st.dataframe(bv, width="stretch")

    fig = px.bar(
        bv.melt(id_vars="model",
                value_vars=["train_auc_d1", "test_auc_d1", "test_auc_d2"],
                var_name="split", value_name="ROC-AUC"),
        x="model", y="ROC-AUC", color="split", barmode="group",
        title="Per-model ROC-AUC: train / test(D1) / test(D2)",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
**How to read this.**
- A large `train - test` gap is a **variance** signal (overfitting).
- A small gap but low train & test AUC is a **bias** signal (underfitting).
- SVM / MLP show the widest gap → high variance under class imbalance.
  The shallow Decision Tree (`max_depth=3`) is the most **bias-heavy** but
  **most stable** under the temporal shift to D2.
"""
    )

    st.subheader("Best hyperparameters chosen by GridSearchCV")
    st.json({m: metrics[m]["best_params"] for m in ["decision_tree", "svm", "mlp"]})

# --------------------------------------------------------------------------- #
#  Feature importance
# --------------------------------------------------------------------------- #

with tab_imp:
    st.subheader("Decision Tree — built-in importance")
    st.image(str(ART / "plots" / "dt_feature_importance.png"), width="stretch")

    st.subheader("Permutation importance — SVM & MLP (D1 test)")
    for name in ["svm", "mlp"]:
        imp = load_csv(ART / "metrics" / f"{name}_permutation_importance.csv",
                       index_col=0).squeeze("columns").sort_values(ascending=False)
        top = imp.head(15).iloc[::-1]
        fig = px.bar(top, orientation="h",
                     title=f"{name} — top 15 permutation importances")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
**Interpretation.** Features driving the cardiometabolic cluster are
consistently **glucose / HbA1c**, **BMI / weight**, and **age**, plus
utilization signals (encounter counts) — matching clinical intuition.
Different model families emphasise different subsets: the Decision Tree
picks a single dominant split (HbA1c or glucose mean), while the SVM/MLP
spread importance across correlated labs.
"""
    )

# --------------------------------------------------------------------------- #
#  Continual learning
# --------------------------------------------------------------------------- #

with tab_cl:
    st.subheader("Continual learning on Dataset 2")
    st.markdown(
        """
Starting from the D1-tuned checkpoints we fine-tune on D2 train:
- **MLP**: true warm-start fine-tuning via `partial_fit` at 0.1× initial LR.
- **DT / SVM**: refit tuned config on `D1_train ∪ D2_train` with a 2× sample
  weight on D2 (emphasis on recent data — the classical fine-tuning analogue
  for estimators without warm-start).
"""
    )

    rows = []
    for name in ["decision_tree", "svm", "mlp"]:
        m_before = metrics[name]["test_d2"]
        m_after  = metrics["continual"][name]["test_d2_after_cl"]
        rows.append({
            "model": name,
            "auc_before_cl": m_before["roc_auc"],
            "auc_after_cl":  m_after["roc_auc"],
            "f1_before_cl":  m_before["f1"],
            "f1_after_cl":   m_after["f1"],
            "recall_before": m_before["recall"],
            "recall_after":  m_after["recall"],
        })
    cl = pd.DataFrame(rows).round(3)
    st.dataframe(cl, width="stretch")

    fig = px.bar(
        cl.melt(id_vars="model",
                value_vars=["auc_before_cl", "auc_after_cl"],
                var_name="stage", value_name="ROC-AUC (D2 test)"),
        x="model", y="ROC-AUC (D2 test)", color="stage", barmode="group",
        title="D2 test ROC-AUC — before vs after continual learning",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Confusion matrices (D2 test, after continual)")
    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1, c2, c3], ["decision_tree", "svm", "mlp"]):
        with col:
            st.image(str(ART / "plots" / f"cm_{name}_continual.png"),
                     caption=f"{name} · D2 after CL")

    st.image(str(ART / "plots" / "roc_continual.png"),
             caption="ROC curves on D2 test — after continual learning")

    st.markdown(
        """
**Takeaway.** All three models improve on Dataset 2 after continual
learning, confirming that (a) there is real temporal drift between the
pre-2021 and 2021+ slices and (b) a simple fine-tuning strategy recovers
much of the lost performance — the key motivation for continual learning
in deployed clinical ML systems.
"""
    )
