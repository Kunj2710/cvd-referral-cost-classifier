# CVD Project - Phase 8: Fairness, Robustness, and Confidence
# BU MET CS 577 | Spring 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

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

# train all 6 models
print("Training all 6 models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'KNN':                 KNeighborsClassifier(n_neighbors=11),
    'Naive Bayes':         GaussianNB(),
    'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=12,
                                                   random_state=RANDOM_STATE, n_jobs=-1),
    'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
}

for name, model in models.items():
    X_tr = X_train_svm if name == 'SVM (RBF)' else X_train_scaled
    y_tr = y_train_svm if name == 'SVM (RBF)' else y_train
    model.fit(X_tr, y_tr)
print("Done.")

short_names = ['LR', 'KNN', 'NB', 'DT', 'RF', 'SVM']
bar_colors  = [COLOR_NO_CVD, COLOR_ACCENT1, COLOR_ACCENT2,
               COLOR_CVD, COLOR_ACCENT3, '#6B4226']

# PART 1: GENDER FAIRNESS AUDIT
print("\nPart 1 - Gender Fairness Audit")

X_test_with_meta = X_test_scaled.copy()
X_test_with_meta['cardio']       = y_test.values
X_test_with_meta['gender_label'] = X_test['gender'].values

female_mask = X_test_with_meta['gender_label'] == 1
male_mask   = X_test_with_meta['gender_label'] == 2

X_female = X_test_with_meta[female_mask][FEATURE_COLS]
y_female = X_test_with_meta[female_mask]['cardio']
X_male   = X_test_with_meta[male_mask][FEATURE_COLS]
y_male   = X_test_with_meta[male_mask]['cardio']

print(f"Female test patients: {len(X_female):,} | Male test patients: {len(X_male):,}")

fairness_results = {}
for name, model in models.items():
    y_pred_f = model.predict(X_female)
    y_pred_m = model.predict(X_male)

    acc_f  = accuracy_score(y_female, y_pred_f)
    acc_m  = accuracy_score(y_male, y_pred_m)
    rec_f  = recall_score(y_female, y_pred_f)
    rec_m  = recall_score(y_male, y_pred_m)

    fairness_results[name] = {
        'acc_female': acc_f, 'acc_male': acc_m, 'acc_gap': abs(acc_f - acc_m),
        'rec_female': rec_f, 'rec_male': rec_m, 'rec_gap': abs(rec_f - rec_m)
    }
    print(f"  {name:22s} acc F={acc_f:.3f} M={acc_m:.3f} gap={abs(acc_f-acc_m):.3f} | "
          f"recall F={rec_f:.3f} M={rec_m:.3f} gap={abs(rec_f-rec_m):.3f}")

# identical patient pairs test
print("\nIdentical patient pair test (gender swap):")
base_patient = {
    'age': 55.0, 'height': 165.0, 'weight': 75.0,
    'ap_hi': 140.0, 'ap_lo': 90.0, 'cholesterol': 2.0,
    'gluc': 1.0, 'smoke': 0.0, 'alco': 0.0, 'active': 1.0,
    'bmi': 27.5, 'pulse_pressure': 50.0
}

patient_female = pd.DataFrame([{**base_patient, 'gender': 1.0}])[FEATURE_COLS]
patient_male   = pd.DataFrame([{**base_patient, 'gender': 2.0}])[FEATURE_COLS]

patient_female_scaled = patient_female.copy()
patient_male_scaled   = patient_male.copy()
patient_female_scaled[SCALE_COLS] = scaler.transform(patient_female[SCALE_COLS])
patient_male_scaled[SCALE_COLS]   = scaler.transform(patient_male[SCALE_COLS])

print(f"Patient profile: age=55, ap_hi=140, ap_lo=90, cholesterol=2 (Above Normal)")
print(f"{'Model':22s}  Female pred  Female prob  Male pred  Male prob  Agrees?")
for name, model in models.items():
    pred_f = model.predict(patient_female_scaled)[0]
    pred_m = model.predict(patient_male_scaled)[0]
    prob_f = model.predict_proba(patient_female_scaled)[0][1]
    prob_m = model.predict_proba(patient_male_scaled)[0][1]
    agrees = 'YES' if pred_f == pred_m else 'NO - BIAS'
    print(f"  {name:22s}  {'CVD' if pred_f else 'No CVD':10s}  {prob_f:.3f}        "
          f"{'CVD' if pred_m else 'No CVD':9s}  {prob_m:.3f}     {agrees}")

# fig 23: fairness audit
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Fig 23 - Gender Fairness Audit', fontsize=14, fontweight='bold')

# accuracy by gender
x = np.arange(len(models))
w = 0.35
acc_f = [fairness_results[n]['acc_female'] for n in models]
acc_m = [fairness_results[n]['acc_male']   for n in models]
axes[0].bar(x - w/2, acc_f, w, color=COLOR_ACCENT1, label='Female', alpha=0.85)
axes[0].bar(x + w/2, acc_m, w, color=COLOR_NO_CVD,  label='Male',   alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names)
axes[0].set_title('Accuracy by Gender')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0.65, 0.80)
axes[0].legend()

# recall by gender - more important clinically
rec_f = [fairness_results[n]['rec_female'] for n in models]
rec_m = [fairness_results[n]['rec_male']   for n in models]
axes[1].bar(x - w/2, rec_f, w, color=COLOR_ACCENT1, label='Female', alpha=0.85)
axes[1].bar(x + w/2, rec_m, w, color=COLOR_NO_CVD,  label='Male',   alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(short_names)
axes[1].set_title('Recall by Gender\n(did we catch CVD patients?)')
axes[1].set_ylabel('Recall')
axes[1].set_ylim(0.5, 0.85)
axes[1].legend()

# fairness gap - recall gap per model
rec_gaps = [fairness_results[n]['rec_gap'] for n in models]
colors_gap = [COLOR_ACCENT3 if g < 0.05 else COLOR_CVD for g in rec_gaps]
axes[2].bar(short_names, rec_gaps, color=colors_gap, edgecolor='white', linewidth=0.8)
axes[2].axhline(0.05, color='gray', linestyle='--', linewidth=1, label='5% fairness threshold')
axes[2].set_title('Recall Gap Between Genders\n(green = fair, red = biased)')
axes[2].set_ylabel('|Recall Female - Recall Male|')
axes[2].legend(fontsize=8)
for bar, val in zip(axes[2].patches, rec_gaps):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.002,
                 f'{val:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('fig23_fairness_audit.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 23 saved.")

# PART 2: ROBUSTNESS TEST
print("\nPart 2 - Robustness Test (measurement noise)")

noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
robustness_results = {name: [] for name in models}

np.random.seed(RANDOM_STATE)
for noise in noise_levels:
    if noise == 0.0:
        X_noisy = X_test_scaled.copy()
    else:
        noise_matrix = np.random.normal(0, noise, X_test_scaled[SCALE_COLS].shape)
        X_noisy = X_test_scaled.copy()
        X_noisy[SCALE_COLS] = X_test_scaled[SCALE_COLS] + noise_matrix

    for name, model in models.items():
        acc = accuracy_score(y_test, model.predict(X_noisy))
        robustness_results[name].append(acc)

    print(f"  Noise={noise:.0%}: " +
          " | ".join([f"{s}={robustness_results[n][-1]:.3f}"
                      for n, s in zip(models, short_names)]))

# fig 24: robustness curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Fig 24 - Robustness Under Measurement Noise', fontsize=14, fontweight='bold')

noise_pct = [n * 100 for n in noise_levels]
for (name, accs), color, short in zip(robustness_results.items(), bar_colors, short_names):
    axes[0].plot(noise_pct, accs, color=color, linewidth=2,
                 marker='o', markersize=6, label=short)

axes[0].set_xlabel('Noise Level (%)')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy vs Noise Level\n(simulating measurement errors)')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0.60, 0.78)
axes[0].axvline(5, color='gray', linestyle=':', linewidth=1, label='realistic noise (~5%)')

# accuracy drop at 20% noise
acc_drop = {name: robustness_results[name][0] - robustness_results[name][-1]
            for name in models}
drop_colors = [COLOR_ACCENT3 if v < 0.02 else COLOR_CVD for v in acc_drop.values()]
bars = axes[1].bar(short_names, list(acc_drop.values()),
                   color=drop_colors, edgecolor='white', linewidth=0.8)
axes[1].set_title('Accuracy Drop at 20% Noise\n(green = robust, red = fragile)')
axes[1].set_ylabel('Accuracy Drop')
axes[1].axhline(0.02, color='gray', linestyle='--', linewidth=1, label='2% threshold')
axes[1].legend(fontsize=8)
for bar, val in zip(bars, acc_drop.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.001,
                 f'{val:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('fig24_robustness.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 24 saved.")

# PART 3: CONFIDENCE-BASED PREDICTIONS
print("\nPart 3 - Confidence-Based Predictions")

GREY_ZONE_LOW  = 0.40
GREY_ZONE_HIGH = 0.60

# use RF (best AUC) and KNN (best cost) for confidence analysis
rf  = models['Random Forest']
knn = models['KNN']

rf_probs  = rf.predict_proba(X_test_scaled)[:, 1]
knn_probs = knn.predict_proba(X_test_scaled)[:, 1]

for model_name, probs in [('Random Forest', rf_probs), ('KNN', knn_probs)]:
    confident_mask    = (probs < GREY_ZONE_LOW) | (probs > GREY_ZONE_HIGH)
    uncertain_mask    = ~confident_mask
    n_confident       = confident_mask.sum()
    n_uncertain       = uncertain_mask.sum()
    confident_acc     = accuracy_score(y_test[confident_mask],
                                       (probs[confident_mask] > 0.5).astype(int))
    overall_acc       = accuracy_score(y_test, (probs > 0.5).astype(int))

    print(f"\n  {model_name}:")
    print(f"    Confident predictions  : {n_confident:,} ({n_confident/len(y_test)*100:.1f}%)")
    print(f"    Uncertain (grey zone)  : {n_uncertain:,} ({n_uncertain/len(y_test)*100:.1f}%)")
    print(f"    Overall accuracy       : {overall_acc:.4f}")
    print(f"    Confident-only accuracy: {confident_acc:.4f}")
    print(f"    Accuracy gain from filtering: +{confident_acc-overall_acc:.4f}")

# fig 25: confidence analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Fig 25 - Confidence-Based Predictions', fontsize=14, fontweight='bold')

# probability distribution for RF
axes[0].hist(rf_probs[y_test == 0], bins=40, alpha=0.6,
             color=COLOR_NO_CVD, label='No CVD', edgecolor='none')
axes[0].hist(rf_probs[y_test == 1], bins=40, alpha=0.6,
             color=COLOR_CVD, label='Has CVD', edgecolor='none')
axes[0].axvspan(GREY_ZONE_LOW, GREY_ZONE_HIGH, alpha=0.15,
                color='gray', label=f'Grey zone ({GREY_ZONE_LOW}-{GREY_ZONE_HIGH})')
axes[0].axvline(0.5, color='black', linestyle='--', linewidth=1)
axes[0].set_title('RF Predicted Probability Distribution')
axes[0].set_xlabel('P(CVD)')
axes[0].set_ylabel('Count')
axes[0].legend(fontsize=8)

# confidence coverage vs accuracy tradeoff
thresholds   = np.arange(0.05, 0.45, 0.02)
coverages_rf = []
accs_rf      = []
for t in thresholds:
    mask = (rf_probs < (0.5 - t)) | (rf_probs > (0.5 + t))
    if mask.sum() > 100:
        coverages_rf.append(mask.mean() * 100)
        accs_rf.append(accuracy_score(y_test[mask], (rf_probs[mask] > 0.5).astype(int)))
    else:
        coverages_rf.append(np.nan)
        accs_rf.append(np.nan)

ax_twin = axes[1].twinx()
axes[1].plot(coverages_rf, accs_rf, color=COLOR_ACCENT3, linewidth=2, marker='o', markersize=4)
axes[1].set_xlabel('Coverage (% of patients predicted)')
axes[1].set_ylabel('Accuracy on predicted patients', color=COLOR_ACCENT3)
axes[1].set_title('RF Coverage vs Accuracy Tradeoff\n(less coverage = higher accuracy)')

# mark our chosen threshold
chosen_conf_mask = (rf_probs < GREY_ZONE_LOW) | (rf_probs > GREY_ZONE_HIGH)
chosen_coverage  = chosen_conf_mask.mean() * 100
chosen_acc       = accuracy_score(y_test[chosen_conf_mask],
                                   (rf_probs[chosen_conf_mask] > 0.5).astype(int))
axes[1].scatter([chosen_coverage], [chosen_acc], color=COLOR_CVD, s=100, zorder=5,
                label=f'Our threshold: {chosen_coverage:.0f}% coverage, {chosen_acc:.3f} acc')
axes[1].legend(fontsize=8)

# what the model says for each group
groups = ['Confident\nNo CVD', 'Grey Zone\n(Uncertain)', 'Confident\nHas CVD']
n_confident_no  = ((rf_probs < GREY_ZONE_LOW)).sum()
n_grey          = ((rf_probs >= GREY_ZONE_LOW) & (rf_probs <= GREY_ZONE_HIGH)).sum()
n_confident_cvd = ((rf_probs > GREY_ZONE_HIGH)).sum()
group_counts = [n_confident_no, n_grey, n_confident_cvd]
group_colors = [COLOR_NO_CVD, 'gray', COLOR_CVD]

axes[2].bar(groups, group_counts, color=group_colors, edgecolor='white', linewidth=0.8, alpha=0.85)
axes[2].set_title('Patient Groups by Confidence\n(grey zone = refer for more tests)')
axes[2].set_ylabel('Number of Patients')
for bar, val in zip(axes[2].patches, group_counts):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 50,
                 f'{val:,}\n({val/len(y_test)*100:.1f}%)',
                 ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('fig25_confidence.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 25 saved.")

print(f"\nPhase 8 complete.")
print(f"\nFairness summary:")
for name in models:
    print(f"  {name:22s} recall gap: {fairness_results[name]['rec_gap']:.3f}")

print(f"\nRobustness summary (acc drop at 20% noise):")
for name in models:
    drop = robustness_results[name][0] - robustness_results[name][-1]
    print(f"  {name:22s} drop: {drop:.4f}")

print(f"\nConfidence summary (RF, grey zone {GREY_ZONE_LOW}-{GREY_ZONE_HIGH}):")
print(f"  Confident predictions: {chosen_coverage:.1f}% of patients")
print(f"  Accuracy when confident: {chosen_acc:.4f}")
print(f"\nReady for Phase 9 - Risk Calculator and Final Conclusions.")
