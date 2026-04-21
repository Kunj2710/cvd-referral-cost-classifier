# CVD Project - Phase 5A: Classifier Showdown - Training & Metrics
# BU MET CS 577 | Spring 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

RANDOM_STATE  = 42
COLOR_NO_CVD  = '#2E86AB'
COLOR_CVD     = '#C73E1D'
COLOR_ACCENT1 = '#A23B72'
COLOR_ACCENT2 = '#F18F01'
COLOR_ACCENT3 = '#2BA84A'
COLORS_TARGET = [COLOR_NO_CVD, COLOR_CVD]
LABELS_TARGET = ['No CVD', 'Has CVD']

DATA_PATH = 'cardio_train.csv'

# inline cleaning pipeline
df = pd.read_csv(DATA_PATH, sep=';')
df.drop(columns=['id'], inplace=True)
df.drop_duplicates(inplace=True)
df['age'] = (df['age'] / 365.25).round(1)
df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
df = df[df['ap_hi'] > df['ap_lo']]
df = df[(df['height'] >= 100) & (df['height'] <= 220)]
df = df[(df['weight'] >= 30)  & (df['weight'] <= 200)]
df['bmi'] = (df['weight'] / (df['height'] / 100) ** 2).round(1)
df = df[df['bmi'] <= 60]
df['bmi_category'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,float('inf')],
                             labels=['Underweight','Normal','Overweight','Obese'])
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df.reset_index(drop=True, inplace=True)

FEATURE_COLS = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                'bmi', 'pulse_pressure']
SCALE_COLS   = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']

X = df[FEATURE_COLS]
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
X_test_scaled[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])

# SVM subsample
SVM_N = 10_000
idx_svm    = X_train_scaled.sample(n=SVM_N, random_state=RANDOM_STATE).index
X_train_svm = X_train_scaled.loc[idx_svm]
y_train_svm = y_train.loc[idx_svm]

CV_FOLDS = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

print("Phase 5A - Classifier Showdown")
print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(FEATURE_COLS)}")

# define all 6 models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'KNN':                 KNeighborsClassifier(n_neighbors=11),
    'Naive Bayes':         GaussianNB(),
    'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=12,
                                                   random_state=RANDOM_STATE, n_jobs=-1),
    'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
}

# train each model, collect all metrics
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # SVM uses subsample, all others use full train set
    if name == 'SVM (RBF)':
        X_tr, y_tr = X_train_svm, y_train_svm
        cv_X, cv_y = X_train_svm, y_train_svm
    else:
        X_tr, y_tr = X_train_scaled, y_train
        cv_X, cv_y = X_train_scaled, y_train

    # train and time it
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    # test set predictions
    y_pred      = model.predict(X_test_scaled)
    y_prob      = model.predict_proba(X_test_scaled)[:, 1]

    # 10-fold cross validation on train set
    cv_scores = cross_val_score(model, cv_X, cv_y,
                                cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)

    # all metrics
    results[name] = {
        'accuracy':   accuracy_score(y_test, y_pred),
        'f1':         f1_score(y_test, y_pred),
        'precision':  precision_score(y_test, y_pred),
        'recall':     recall_score(y_test, y_pred),
        'auc':        roc_auc_score(y_test, y_prob),
        'cv_mean':    cv_scores.mean(),
        'cv_std':     cv_scores.std(),
        'cv_scores':  cv_scores,
        'train_time': train_time,
        'y_pred':     y_pred,
        'y_prob':     y_prob,
        'model':      model
    }

    print(f"  Accuracy : {results[name]['accuracy']:.4f}")
    print(f"  F1       : {results[name]['f1']:.4f}")
    print(f"  Recall   : {results[name]['recall']:.4f}")
    print(f"  AUC      : {results[name]['auc']:.4f}")
    print(f"  CV       : {results[name]['cv_mean']:.4f} +/- {results[name]['cv_std']:.4f}")
    print(f"  Time     : {train_time:.1f}s")

# results dataframe - sorted by AUC
results_df = pd.DataFrame({
    name: {
        'Accuracy':  f"{v['accuracy']:.4f}",
        'F1':        f"{v['f1']:.4f}",
        'Precision': f"{v['precision']:.4f}",
        'Recall':    f"{v['recall']:.4f}",
        'AUC':       f"{v['auc']:.4f}",
        'CV Mean':   f"{v['cv_mean']:.4f}",
        'CV Std':    f"+/-{v['cv_std']:.4f}",
        'Time (s)':  f"{v['train_time']:.1f}"
    }
    for name, v in results.items()
}).T

results_df_sorted = results_df.sort_values('AUC', ascending=False)
print("\nFinal Leaderboard (sorted by AUC):")
print(results_df_sorted.to_string())

# fig 12: model comparison bar chart
metrics_to_plot = ['accuracy', 'f1', 'recall', 'auc']
metric_labels   = ['Accuracy', 'F1', 'Recall', 'AUC']
model_names     = list(results.keys())
short_names     = ['LR', 'KNN', 'NB', 'DT', 'RF', 'SVM']
bar_colors      = [COLOR_NO_CVD, COLOR_ACCENT1, COLOR_ACCENT2,
                   COLOR_CVD, COLOR_ACCENT3, '#6B4226']

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Fig 12 - Model Comparison Across Metrics', fontsize=14, fontweight='bold')

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    vals = [results[n][metric] for n in model_names]
    bars = ax.bar(short_names, vals, color=bar_colors,
                  edgecolor='white', linewidth=0.8)
    ax.set_title(label)
    ax.set_ylabel('Score')
    ax.set_ylim(0.5, 0.85)
    ax.axhline(max(vals), color='gray', linestyle='--', linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f'{val:.3f}', ha='center', fontsize=7, rotation=45)

plt.tight_layout()
plt.savefig('fig12_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFig 12 saved.")

# fig 13: cross-validation box plots
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('Fig 13 - 10-Fold Cross-Validation Accuracy Distribution', fontsize=14, fontweight='bold')

cv_data = [results[n]['cv_scores'] for n in model_names]
bp = ax.boxplot(cv_data, labels=short_names, patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('CV Accuracy')
ax.set_ylim(0.5, 0.85)

for i, (name, scores) in enumerate(zip(model_names, cv_data)):
    ax.text(i + 1, scores.mean() + 0.002,
            f'{scores.mean():.3f}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('fig13_cv_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 13 saved.")

# fig 14: ROC curves all models overlaid
fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle('Fig 14 - ROC Curves (All 6 Models)', fontsize=14, fontweight='bold')

roc_colors = [COLOR_NO_CVD, COLOR_ACCENT1, COLOR_ACCENT2,
              COLOR_CVD, COLOR_ACCENT3, '#6B4226']

for (name, v), color in zip(results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, v['y_prob'])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC={v['auc']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Higher AUC = better discrimination')
ax.legend(fontsize=8, loc='lower right')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')

plt.tight_layout()
plt.savefig('fig14_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 14 saved.")

# fig 15: confusion matrices - all 6 in one grid
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Fig 15 - Confusion Matrices (All 6 Models)', fontsize=14, fontweight='bold')
axes = axes.flatten()

for ax, (name, v), short in zip(axes, results.items(), short_names):
    cm = confusion_matrix(y_test, v['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=LABELS_TARGET)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name}\nAcc={v["accuracy"]:.3f}  Recall={v["recall"]:.3f}')
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout()
plt.savefig('fig15_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 15 saved.")

print("\nPhase 5A complete.")
print("Key numbers for Phase 6 cost analysis:")
for name, v in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
    cm   = confusion_matrix(y_test, v['y_pred'])
    fn   = cm[1][0]
    fp   = cm[0][1]
    print(f"  {name:22s} AUC={v['auc']:.3f}  FN={fn:,}  FP={fp:,}  Recall={v['recall']:.3f}")

print("\nSave results dict - needed for Phase 5B and Phase 6.")
