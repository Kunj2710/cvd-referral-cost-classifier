# CVD Project - Phase 4: Feature Preparation
# BU MET CS 577 | Spring 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE  = 42
COLOR_NO_CVD  = '#2E86AB'
COLOR_CVD     = '#C73E1D'
COLOR_ACCENT2 = '#F18F01'
COLORS_TARGET = [COLOR_NO_CVD, COLOR_CVD]
LABELS_TARGET = ['No CVD', 'Has CVD']

DATA_PATH = 'cardio_train.csv'

# inline cleaning - same pipeline as Phase 2 and 3
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
bmi_bins   = [0, 18.5, 25.0, 30.0, float('inf')]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df.reset_index(drop=True, inplace=True)

print(f"Clean dataset: {len(df):,} rows")

# define features and target
# dropping bmi_category - it's a categorical version of bmi, redundant for modeling
# keeping bmi (continuous) is better for all 6 models
FEATURE_COLS = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                'bmi', 'pulse_pressure']

X = df[FEATURE_COLS]
y = df['cardio']

print(f"Features: {list(X.columns)}")
print(f"Target  : cardio (0/1)")
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")

# train/test split - 80/20 stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain set : {X_train.shape[0]:,} rows ({X_train.shape[0]/len(df)*100:.0f}%)")
print(f"Test set  : {X_test.shape[0]:,} rows ({X_test.shape[0]/len(df)*100:.0f}%)")

# verify stratification held
train_cvd_rate = y_train.mean() * 100
test_cvd_rate  = y_test.mean()  * 100
print(f"\nCVD rate - train: {train_cvd_rate:.1f}% | test: {test_cvd_rate:.1f}%")

# scale features - fit ONLY on train, transform both
# columns that benefit from scaling
SCALE_COLS = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
X_test_scaled[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])

print(f"\nScaling applied to: {SCALE_COLS}")
print(f"Binary/ordinal columns left unscaled: gender, cholesterol, gluc, smoke, alco, active")

# verify scaling - scaled columns should have mean ~0 and std ~1 on train
print(f"\nPost-scaling verification (train set):")
for col in SCALE_COLS:
    mean = X_train_scaled[col].mean()
    std  = X_train_scaled[col].std()
    print(f"  {col:20s} mean: {mean:+.4f}  std: {std:.4f}")

# SVM subsample - SVM is O(n^2), too slow on 54K rows
SVM_SAMPLE_SIZE = 10_000
idx_svm = X_train_scaled.sample(n=SVM_SAMPLE_SIZE, random_state=RANDOM_STATE).index
X_train_svm = X_train_scaled.loc[idx_svm]
y_train_svm = y_train.loc[idx_svm]

svm_cvd_rate = y_train_svm.mean() * 100
print(f"\nSVM subsample: {SVM_SAMPLE_SIZE:,} rows")
print(f"SVM subsample CVD rate: {svm_cvd_rate:.1f}% (confirms stratification held)")

# figure 11 - feature preparation summary
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Phase 4 - Feature Preparation', fontsize=14, fontweight='bold')

# train vs test split visual
split_counts = [X_train.shape[0], X_test.shape[0]]
split_labels = [f'Train\n{X_train.shape[0]:,}\n(70%)', f'Test\n{X_test.shape[0]:,}\n(30%)']
axes[0].bar(split_labels, split_counts,
            color=[COLOR_NO_CVD, COLOR_ACCENT2], edgecolor='white', linewidth=0.8)
axes[0].set_title('Train / Test Split')
axes[0].set_ylabel('Row Count')
axes[0].set_ylim(0, max(split_counts) * 1.2)

# CVD balance in train vs test
balance_data = {
    'Train No CVD':  (y_train == 0).sum(),
    'Train Has CVD': (y_train == 1).sum(),
    'Test No CVD':   (y_test == 0).sum(),
    'Test Has CVD':  (y_test == 1).sum(),
}
bar_colors = [COLOR_NO_CVD, COLOR_CVD, COLOR_NO_CVD, COLOR_CVD]
bars = axes[1].bar(balance_data.keys(), balance_data.values(),
                   color=bar_colors, edgecolor='white', linewidth=0.8, alpha=0.85)
axes[1].set_title('Class Balance After Split')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', labelsize=7)
for bar, val in zip(bars, balance_data.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 100,
                 f'{val:,}', ha='center', fontsize=7)

# before vs after scaling for age (most intuitive example)
axes[2].hist(X_train['age'],        bins=40, alpha=0.5, color=COLOR_CVD,
             label='Before (years)', edgecolor='none')
axes[2].hist(X_train_scaled['age'], bins=40, alpha=0.7, color=COLOR_NO_CVD,
             label='After (z-score)', edgecolor='none')
axes[2].set_title('Scaling Effect on Age')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Count')
axes[2].legend()
ax2_twin = axes[2].twinx()
ax2_twin.set_yticks([])

plt.tight_layout()
plt.savefig('fig11_feature_preparation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFig 11 saved.")

# save everything needed for Phase 5
print("\nObjects ready for Phase 5:")
print(f"  X_train_scaled : {X_train_scaled.shape}")
print(f"  X_test_scaled  : {X_test_scaled.shape}")
print(f"  X_train_svm    : {X_train_svm.shape}  (SVM only)")
print(f"  y_train        : {y_train.shape}")
print(f"  y_test         : {y_test.shape}")
print(f"  y_train_svm    : {y_train_svm.shape}  (SVM only)")
print(f"  scaler         : fitted on train set only")
print(f"  FEATURE_COLS   : {FEATURE_COLS}")
print("\nPhase 4 done. Next: Phase 5 - Classifier Showdown.")
