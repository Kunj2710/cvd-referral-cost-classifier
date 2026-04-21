# CVD Project - Phase 5B: Classifier Visualizations
# BU MET CS 577 | Spring 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

RANDOM_STATE  = 42
COLOR_NO_CVD  = '#2E86AB'
COLOR_CVD     = '#C73E1D'
COLOR_ACCENT1 = '#A23B72'
COLOR_ACCENT2 = '#F18F01'
COLOR_ACCENT3 = '#2BA84A'
COLORS_TARGET = [COLOR_NO_CVD, COLOR_CVD]
LABELS_TARGET = ['No CVD', 'Has CVD']

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

SVM_N = 10_000
idx_svm     = X_train_scaled.sample(n=SVM_N, random_state=RANDOM_STATE).index
X_train_svm = X_train_scaled.loc[idx_svm]
y_train_svm = y_train.loc[idx_svm]

CV_FOLDS = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

# retrain all 6 models - same hyperparameters as 5A
print("Retraining all 6 models for Phase 5B visualizations...")
lr  = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
knn = KNeighborsClassifier(n_neighbors=11)
nb  = GaussianNB()
dt  = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
rf  = RandomForestClassifier(n_estimators=100, max_depth=12,
                              random_state=RANDOM_STATE, n_jobs=-1)
svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)

lr.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
nb.fit(X_train_scaled, y_train)
dt.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
svm.fit(X_train_svm, y_train_svm)
print("All models trained.")

# fig 16: decision boundaries via PCA 2D projection
print("\nBuilding decision boundaries (PCA 2D)...")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d  = pca.transform(X_test_scaled)

var_explained = pca.explained_variance_ratio_
print(f"PCA variance explained: {var_explained[0]:.1%} + {var_explained[1]:.1%} = {sum(var_explained):.1%}")

models_2d = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'KNN':                 KNeighborsClassifier(n_neighbors=11),
    'Naive Bayes':         GaussianNB(),
    'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
}

sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train_2d), 3000, replace=False)
X_2d_sample = X_train_2d[sample_idx]
y_sample    = y_train.values[sample_idx]

x_min, x_max = X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5
y_min, y_max = X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Fig 16 - Decision Boundaries (PCA 2D Projection)', fontsize=14, fontweight='bold')
axes = axes.flatten()

for ax, (name, model_2d) in zip(axes, models_2d.items()):
    model_2d.fit(X_2d_sample, y_sample)
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25,
                colors=[COLOR_NO_CVD, COLOR_CVD])
    ax.contour(xx, yy, Z, colors='white', linewidths=0.5, alpha=0.7)

    for cls, color, label in zip([0, 1], COLORS_TARGET, LABELS_TARGET):
        mask = y_sample == cls
        ax.scatter(X_2d_sample[mask, 0], X_2d_sample[mask, 1],
                   c=color, s=3, alpha=0.3, label=label)

    acc_2d = accuracy_score(y_sample, model_2d.predict(X_2d_sample))
    ax.set_title(f'{name}\n(2D acc={acc_2d:.3f})', fontsize=10)
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})')

patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLORS_TARGET, LABELS_TARGET)]
fig.legend(handles=patches, loc='lower center', ncol=2, fontsize=10, frameon=False)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig('fig16_decision_boundaries.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 16 saved.")

# fig 17: KNN optimal k - bias variance tradeoff
print("\nFinding optimal K for KNN...")

k_range  = range(1, 32, 2)
k_cv_scores = []

for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_k, X_train_scaled, y_train,
                             cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
    k_cv_scores.append(scores.mean())
    print(f"  k={k:2d}  CV accuracy={scores.mean():.4f}")

best_k   = list(k_range)[np.argmax(k_cv_scores)]
best_acc = max(k_cv_scores)
print(f"\nBest k={best_k} with CV accuracy={best_acc:.4f}")

fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle('Fig 17 - KNN Optimal K (Bias-Variance Tradeoff)', fontsize=14, fontweight='bold')

ax.plot(list(k_range), k_cv_scores, color=COLOR_ACCENT1,
        linewidth=2, marker='o', markersize=6)
ax.axvline(best_k, color=COLOR_CVD, linestyle='--', linewidth=1.5,
           label=f'Best k={best_k} (acc={best_acc:.4f})')
ax.axvline(11, color=COLOR_NO_CVD, linestyle=':', linewidth=1.5,
           label='k=11 used in project')

ax.fill_between(list(k_range)[:list(k_range).index(best_k)+1],
                k_cv_scores[:list(k_range).index(best_k)+1],
                min(k_cv_scores), alpha=0.08, color=COLOR_CVD, label='high variance zone (low k)')
ax.fill_between(list(k_range)[list(k_range).index(best_k):],
                k_cv_scores[list(k_range).index(best_k):],
                min(k_cv_scores), alpha=0.08, color=COLOR_NO_CVD, label='high bias zone (high k)')

ax.set_xlabel('Number of Neighbors (k)')
ax.set_ylabel('CV Accuracy')
ax.set_title('Low k = overfit | High k = underfit | Sweet spot in the middle')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('fig17_knn_optimal_k.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 17 saved.")

# fig 18: feature importance - RF gini vs permutation importance
print("\nCalculating feature importance...")

rf_importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

perm_result  = permutation_importance(rf, X_test_scaled, y_test,
                                       n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
perm_imp     = pd.Series(perm_result.importances_mean, index=FEATURE_COLS).sort_values(ascending=True)

lr_coef      = pd.Series(np.abs(lr.coef_[0]), index=FEATURE_COLS).sort_values(ascending=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Fig 18 - Feature Importance Analysis', fontsize=14, fontweight='bold')

# RF gini importance
colors_imp = [COLOR_CVD if v >= rf_importances.values[-3] else COLOR_NO_CVD
              for v in rf_importances.values]
axes[0].barh(rf_importances.index, rf_importances.values,
             color=colors_imp, edgecolor='white', linewidth=0.5)
axes[0].set_title('RF Gini Importance')
axes[0].set_xlabel('Importance Score')

# permutation importance
colors_perm = [COLOR_CVD if v >= perm_imp.values[-3] else COLOR_NO_CVD
               for v in perm_imp.values]
axes[1].barh(perm_imp.index, perm_imp.values,
             color=colors_perm, edgecolor='white', linewidth=0.5)
axes[1].set_title('RF Permutation Importance\n(more reliable than gini)')
axes[1].set_xlabel('Mean Accuracy Drop')

# LR coefficients (absolute values)
colors_lr = [COLOR_CVD if v >= lr_coef.values[-3] else COLOR_NO_CVD
             for v in lr_coef.values]
axes[2].barh(lr_coef.index, lr_coef.values,
             color=colors_lr, edgecolor='white', linewidth=0.5)
axes[2].set_title('LR Coefficient Magnitude\n(|coef| = feature influence)')
axes[2].set_xlabel('|Coefficient|')

plt.tight_layout()
plt.savefig('fig18_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 18 saved.")

print("\nTop 3 features by permutation importance:")
for feat, val in perm_imp.sort_values(ascending=False).head(3).items():
    print(f"  {feat:20s} {val:.4f}")

# fig 19: decision tree visualization - top 3 levels
print("\nBuilding decision tree visualization...")

dt_viz = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
dt_viz.fit(X_train_scaled, y_train)
dt_acc = accuracy_score(y_test, dt_viz.predict(X_test_scaled))

fig, ax = plt.subplots(figsize=(20, 8))
fig.suptitle(f'Fig 19 - Decision Tree (top 3 levels, acc={dt_acc:.3f})', fontsize=14, fontweight='bold')

plot_tree(dt_viz,
          feature_names=FEATURE_COLS,
          class_names=LABELS_TARGET,
          filled=True,
          rounded=True,
          fontsize=8,
          ax=ax,
          impurity=False,
          proportion=False)

plt.tight_layout()
plt.savefig('fig19_decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 19 saved.")

print("\nPhase 5B complete.")
print(f"PCA variance explained by 2 components: {sum(var_explained):.1%}")
print(f"Optimal KNN k: {best_k} (CV acc={best_acc:.4f})")
print(f"Top predictive feature: {perm_imp.idxmax()}")
print(f"Decision Tree top split feature: {FEATURE_COLS[dt_viz.tree_.feature[0]]}")
print("\nAll Phase 5 visualizations complete. Ready for Phase 6.")
