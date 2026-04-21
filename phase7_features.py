# CVD Project - Phase 7: Minimal Feature Discovery
# BU MET CS 577 | Spring 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

RANDOM_STATE  = 42
COLOR_NO_CVD  = '#2E86AB'
COLOR_CVD     = '#C73E1D'
COLOR_ACCENT1 = '#A23B72'
COLOR_ACCENT2 = '#F18F01'
COLOR_ACCENT3 = '#2BA84A'

DATA_PATH = 'cardio_train.csv'

# inline cleaning
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

CV_FOLDS = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

# get feature order from permutation importance on RF
print("Getting feature importance order via permutation importance...")
rf_full = RandomForestClassifier(n_estimators=100, max_depth=12,
                                  random_state=RANDOM_STATE, n_jobs=-1)
rf_full.fit(X_train_scaled, y_train)
perm = permutation_importance(rf_full, X_test_scaled, y_test,
                               n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
perm_series   = pd.Series(perm.importances_mean, index=FEATURE_COLS)
feature_order = perm_series.sort_values(ascending=False).index.tolist()

full_acc = accuracy_score(y_test, rf_full.predict(X_test_scaled))
print(f"Full model (13 features) accuracy: {full_acc:.4f}")
print(f"Feature order (permutation): {feature_order}")

# sequential feature addition - add one feature at a time in importance order
print("\nSequential feature addition (RF):")
seq_results_rf  = []
seq_results_lr  = []
seq_results_knn = []

for n_feats in range(1, len(FEATURE_COLS) + 1):
    feat_subset = feature_order[:n_feats]

    rf_sub  = RandomForestClassifier(n_estimators=50, max_depth=10,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    lr_sub  = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    knn_sub = KNeighborsClassifier(n_neighbors=11)

    rf_sub.fit(X_train_scaled[feat_subset], y_train)
    lr_sub.fit(X_train_scaled[feat_subset], y_train)
    knn_sub.fit(X_train_scaled[feat_subset], y_train)

    rf_acc  = accuracy_score(y_test, rf_sub.predict(X_test_scaled[feat_subset]))
    lr_acc  = accuracy_score(y_test, lr_sub.predict(X_test_scaled[feat_subset]))
    knn_acc = accuracy_score(y_test, knn_sub.predict(X_test_scaled[feat_subset]))

    seq_results_rf.append(rf_acc)
    seq_results_lr.append(lr_acc)
    seq_results_knn.append(knn_acc)

    gap = full_acc - rf_acc
    print(f"  {n_feats:2d} features {str(feat_subset):80s} RF={rf_acc:.4f} gap={gap:.4f}")

# find plateau - where adding more features gives < 0.5% improvement
plateau_idx = None
for i in range(2, len(seq_results_rf)):
    improvement = seq_results_rf[i] - seq_results_rf[i-1]
    if improvement < 0.005 and seq_results_rf[i] >= full_acc * 0.97:
        plateau_idx = i
        break

if plateau_idx:
    plateau_features = feature_order[:plateau_idx + 1]
    plateau_acc      = seq_results_rf[plateau_idx]
    print(f"\nPlateau reached at {plateau_idx + 1} features")
    print(f"Features: {plateau_features}")
    print(f"Accuracy: {plateau_acc:.4f} vs full: {full_acc:.4f}")
    print(f"Gap from full model: {full_acc - plateau_acc:.4f} ({(full_acc-plateau_acc)/full_acc*100:.1f}%)")

# best 3-feature combination search
print("\nSearching best 3-feature combinations (RF)...")
top_features_pool = feature_order[:7]
best_3_acc  = 0
best_3_combo = None

combo_results = {}
for combo in combinations(top_features_pool, 3):
    rf_3 = RandomForestClassifier(n_estimators=50, max_depth=10,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    rf_3.fit(X_train_scaled[list(combo)], y_train)
    acc = accuracy_score(y_test, rf_3.predict(X_test_scaled[list(combo)]))
    combo_results[combo] = acc
    if acc > best_3_acc:
        best_3_acc   = acc
        best_3_combo = combo

print(f"Best 3-feature combo: {best_3_combo}")
print(f"Best 3-feature accuracy: {best_3_acc:.4f}")
print(f"Gap from 13-feature full model: {full_acc - best_3_acc:.4f} ({(full_acc-best_3_acc)/full_acc*100:.1f}%)")

# fig 22: accuracy vs number of features
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Fig 22 - Minimal Feature Discovery', fontsize=14, fontweight='bold')

ax = axes[0]
n_range = list(range(1, len(FEATURE_COLS) + 1))

ax.plot(n_range, seq_results_rf,  color=COLOR_ACCENT3, linewidth=2.5,
        marker='o', markersize=6, label='Random Forest')
ax.plot(n_range, seq_results_lr,  color=COLOR_NO_CVD, linewidth=2,
        marker='s', markersize=5, label='Logistic Regression')
ax.plot(n_range, seq_results_knn, color=COLOR_ACCENT1, linewidth=2,
        marker='^', markersize=5, label='KNN')

ax.axhline(full_acc, color='black', linestyle='--', linewidth=1.2,
           label=f'Full model ({full_acc:.3f})')
ax.axhline(full_acc * 0.97, color='gray', linestyle=':', linewidth=1,
           label='97% of full model')

if plateau_idx:
    ax.axvline(plateau_idx + 1, color=COLOR_CVD, linestyle='--', linewidth=1.5,
               label=f'Plateau at {plateau_idx+1} features')
    ax.scatter([plateau_idx + 1], [seq_results_rf[plateau_idx]],
               color=COLOR_CVD, s=120, zorder=5)

ax.set_xlabel('Number of Features')
ax.set_ylabel('Test Accuracy')
ax.set_title('Accuracy vs Number of Features\n(features added by permutation importance rank)')
ax.legend(fontsize=8)
ax.set_xlim(0.5, len(FEATURE_COLS) + 0.5)
ax.set_ylim(0.55, 0.80)

for i, feat in enumerate(feature_order[:6]):
    ax.annotate(feat, (i+1, seq_results_rf[i]),
                textcoords='offset points', xytext=(0, 8),
                fontsize=6, ha='center', color='gray')

# fig 22 panel 2: 3-feature combo heatmap
ax2 = axes[1]
top6 = feature_order[:6]
combo_matrix = pd.DataFrame(index=top6, columns=top6, dtype=float)

for combo, acc in combo_results.items():
    if len(combo) == 3:
        for i in range(len(combo)):
            for j in range(len(combo)):
                if i != j and combo[i] in top6 and combo[j] in top6:
                    current = combo_matrix.loc[combo[i], combo[j]]
                    if pd.isna(current) or acc > current:
                        combo_matrix.loc[combo[i], combo[j]] = acc

combo_matrix = combo_matrix.fillna(0).astype(float)

sns.heatmap(combo_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            center=combo_matrix.values[combo_matrix.values > 0].mean(),
            ax=ax2, linewidths=0.5, linecolor='white',
            annot_kws={'size': 8})
ax2.set_title(f'3-Feature Combo Accuracy Heatmap\nBest combo: {best_3_combo}\nacc={best_3_acc:.4f}')
ax2.set_xlabel('Feature 2')
ax2.set_ylabel('Feature 1')

plt.tight_layout()
plt.savefig('fig22_minimal_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFig 22 saved.")

# print final summary
print(f"\nPhase 7 complete - Minimal Feature Discovery")
print(f"Full model (13 features)    : {full_acc:.4f}")
print(f"Best 3-feature combo        : {best_3_combo}")
print(f"Best 3-feature accuracy     : {best_3_acc:.4f}")
print(f"Accuracy gap (13 vs 3 feat) : {full_acc - best_3_acc:.4f} ({(full_acc-best_3_acc)/full_acc*100:.2f}%)")
print(f"\nClinical meaning:")
print(f"A doctor with only {list(best_3_combo)} can achieve")
print(f"{best_3_acc:.1%} accuracy — only {(full_acc-best_3_acc)*100:.1f}% less than")
print(f"using all 13 features with laboratory tests.")
print(f"\nReady for Phase 8 - Fairness, Robustness, and Confidence.")
