"""Phase 10: threshold tuning, referral capacity, and prevalence.

Phase 6 ranked models by cost at the default 0.5 threshold. Under the stated
500:1 cost ratio that comparison is inconsistent: the cost-minimising rule is
"refer if P(CVD) > FP / (FP + FN)" = 300 / 150,300, about 0.2%, which refers
almost everyone. This script makes that explicit and then asks the more useful
question: with a fixed cardiology referral capacity, which model misses the
fewest sick patients?

Models are fit on the training split only. Under a capacity rule no threshold
has to be tuned: the clinic simply sees the N highest-risk patients, so the
comparison is purely about how well each model ranks risk. Ties in coarse
scores (KNN, decision tree) are broken at random so every model refers
exactly N patients.

Run:  python phase10_thresholds.py           (expects cardio_train.csv here)
Out:  results/threshold_analysis.csv, results/capacity_analysis.csv,
      results/capacity_fairness.csv, figures/fig27_capacity_curve.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from cvd_data import FEATURE_COLS, SCALE_COLS, load_clean

RANDOM_STATE = 42
FN_COST, FP_COST = 150_000, 300
PER = 10_000
CAPACITIES = [0.20, 0.30, 0.40, 0.50, 0.60]
OUT = Path("results")
FIG = Path("figures")


def build_models() -> dict:
    scale = ColumnTransformer([("s", StandardScaler(), SCALE_COLS)], remainder="passthrough")
    return {
        "Logistic Regression": make_pipeline(scale, LogisticRegression(max_iter=1000)),
        "KNN": make_pipeline(scale, KNeighborsClassifier(n_neighbors=11)),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
        "SVM (RBF)": make_pipeline(scale, SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    }


def cost_per_10k(y, refer, prevalence=None):
    """Expected cost per 10,000 patients. If prevalence is given, reweight to it."""
    y, refer = np.asarray(y), np.asarray(refer).astype(bool)
    fnr = ((~refer) & (y == 1)).sum() / (y == 1).sum()
    fpr = (refer & (y == 0)).sum() / (y == 0).sum()
    p = y.mean() if prevalence is None else prevalence
    fn, fp = fnr * p * PER, fpr * (1 - p) * PER
    return fn, fp, fn * FN_COST + fp * FP_COST


def top_k(scores, share, seed=RANDOM_STATE):
    """Boolean mask selecting exactly round(share * n) highest scores, ties broken at random."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(scores)), -np.asarray(scores)))
    refer = np.zeros(len(scores), dtype=bool)
    refer[order[: int(round(share * len(scores)))]] = True
    return refer


def main():
    OUT.mkdir(exist_ok=True)
    df = load_clean("cardio_train.csv")
    X, y = df[FEATURE_COLS], df["cardio"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    prev = y_te.mean()
    print(f"Clean rows {len(df):,} | test {len(y_te):,} | CVD prevalence {prev:.1%}")

    # SVM on a 10K subsample, as in phase 5/6, to keep runtime reasonable
    svm_idx = X_tr.sample(n=10_000, random_state=RANDOM_STATE).index
    test_p = {}
    for name, model in build_models().items():
        Xa, ya = (X_tr.loc[svm_idx], y_tr.loc[svm_idx]) if name.startswith("SVM") else (X_tr, y_tr)
        test_p[name] = clone(model).fit(Xa, ya).predict_proba(X_te)[:, 1]
        print(f"  {name:20s} test AUC {roc_auc_score(y_te, test_p[name]):.3f}")

    # 1) default threshold vs cost-optimal threshold vs trivial policies
    t_star = FP_COST / (FP_COST + FN_COST)
    rows = []
    for policy, refer in [("Refer nobody", np.zeros(len(y_te))), ("Refer everybody", np.ones(len(y_te)))]:
        fn, fp, c = cost_per_10k(y_te, refer)
        rows.append({"Model": policy, "Rule": "–", "Referred": refer.mean(), "Missed/10K": fn, "False refs/10K": fp, "Cost/10K ($M)": c / 1e6})
    for name, p in test_p.items():
        for rule, t in [("p > 0.5", 0.5), (f"p > {t_star:.4f} (cost-optimal)", t_star)]:
            refer = p > t
            fn, fp, c = cost_per_10k(y_te, refer)
            rows.append({"Model": name, "Rule": rule, "Referred": refer.mean(), "Missed/10K": fn, "False refs/10K": fp, "Cost/10K ($M)": c / 1e6})
    thr = pd.DataFrame(rows)
    thr.to_csv(OUT / "threshold_analysis.csv", index=False)
    print("\n== Default vs cost-optimal thresholds (test set, dataset prevalence)")
    print(thr.round(3).to_string(index=False))

    # 2) fixed referral capacity: refer exactly the top cap*N patients by predicted risk
    rows = []
    for cap in CAPACITIES:
        for name, p in test_p.items():
            refer = top_k(p, cap)
            recall = refer[y_te.to_numpy() == 1].mean()
            for label, pv in [("dataset", None), ("10% illustrative", 0.10)]:
                fn, fp, c = cost_per_10k(y_te, refer, pv)
                rows.append({"Capacity": cap, "Model": name, "Prevalence": label, "Actually referred": refer.mean(),
                             "Recall": recall, "Missed/10K": fn, "Cost/10K ($M)": c / 1e6})
    capdf = pd.DataFrame(rows)
    capdf.to_csv(OUT / "capacity_analysis.csv", index=False)
    piv = capdf[capdf.Prevalence == "dataset"].pivot(index="Model", columns="Capacity", values="Missed/10K").round(0)
    print("\n== Sick patients missed per 10K at fixed referral capacity (dataset prevalence)")
    print(piv.sort_values(0.4).to_string())

    # 3) fairness at the 40% capacity operating point: equal opportunity + selection-rate ratio
    rows = []
    g = X_te["gender"].to_numpy()  # 1 = female, 2 = male in this dataset
    for name, p in test_p.items():
        refer = top_k(p, 0.40)
        yy = y_te.to_numpy()
        rec_f, rec_m = refer[(g == 1) & (yy == 1)].mean(), refer[(g == 2) & (yy == 1)].mean()
        sel_f, sel_m = refer[g == 1].mean(), refer[g == 2].mean()
        rows.append({"Model": name, "Recall F": rec_f, "Recall M": rec_m, "Recall gap": abs(rec_f - rec_m),
                     "Referral rate F": sel_f, "Referral rate M": sel_m,
                     "Selection ratio (min/max)": min(sel_f, sel_m) / max(sel_f, sel_m)})
    fair = pd.DataFrame(rows)
    fair.to_csv(OUT / "capacity_fairness.csv", index=False)
    print("\n== Fairness at 40% capacity")
    print(fair.round(3).to_string(index=False))

    # figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = capdf[capdf.Prevalence == "dataset"]
    for name in test_p:
        s = d[d.Model == name]
        ax.plot(s.Capacity * 100, s.Recall * 100, marker="o", label=name)
    ax.set(xlabel="Referral capacity (% of patients cardiology can see)", ylabel="Sick patients caught (%)",
           title="Which model catches the most CVD at a fixed referral budget?")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig27_capacity_curve.png", dpi=150)


if __name__ == "__main__":
    main()
