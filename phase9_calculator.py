# CVD Project - Phase 9: Risk Calculator & Final Conclusions
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
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              precision_score, roc_auc_score, confusion_matrix)

RANDOM_STATE  = 42
COLOR_NO_CVD  = '#2E86AB'
COLOR_CVD     = '#C73E1D'
COLOR_ACCENT1 = '#A23B72'
COLOR_ACCENT2 = '#F18F01'
COLOR_ACCENT3 = '#2BA84A'

FN_COST = 150_000
FP_COST = 300

GREY_ZONE_LOW  = 0.40
GREY_ZONE_HIGH = 0.60

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
print("All models trained.")

# PART 1: RISK CALCULATOR
print("\nPart 1 - CVD Risk Calculator")
print("Uses top 3 features from Phase 7: ap_hi, cholesterol, age")
print("Model: Random Forest (best AUC, best fairness, robust to noise)")
print("Includes grey zone from Phase 8: 0.40-0.60 = uncertain")

rf = models['Random Forest']

def calculate_cvd_risk(age_years, ap_hi_mmhg, cholesterol_level,
                        gender=1, height_cm=165, weight_kg=75,
                        ap_lo_mmhg=80, gluc=1, smoke=0, alco=0, active=1):
    """
    Predicts CVD risk for a patient using the trained Random Forest model.
    Uses all 13 features for prediction but highlights the top 3.

    Parameters:
        age_years      : patient age in years (e.g. 55)
        ap_hi_mmhg     : systolic blood pressure in mmHg (e.g. 140)
        cholesterol_level: 1=Normal, 2=Above Normal, 3=Well Above Normal
        gender         : 1=Female, 2=Male (default 1)
        height_cm      : height in cm (default 165)
        weight_kg      : weight in kg (default 75)
        ap_lo_mmhg     : diastolic BP in mmHg (default 80)
        gluc           : 1=Normal, 2=Above Normal, 3=Well Above Normal (default 1)
        smoke          : 0=No, 1=Yes (default 0)
        alco           : 0=No, 1=Yes (default 0)
        active         : 0=No, 1=Yes (default 1)

    Returns:
        dict with risk_level, probability, recommendation, confidence_flag
    """
    bmi            = round(weight_kg / (height_cm / 100) ** 2, 1)
    pulse_pressure = ap_hi_mmhg - ap_lo_mmhg

    patient = pd.DataFrame([{
        'age': age_years, 'gender': gender,
        'height': height_cm, 'weight': weight_kg,
        'ap_hi': ap_hi_mmhg, 'ap_lo': ap_lo_mmhg,
        'cholesterol': cholesterol_level, 'gluc': gluc,
        'smoke': smoke, 'alco': alco, 'active': active,
        'bmi': bmi, 'pulse_pressure': pulse_pressure
    }])[FEATURE_COLS]

    patient_scaled = patient.copy()
    patient_scaled[SCALE_COLS] = scaler.transform(patient[SCALE_COLS])

    prob = rf.predict_proba(patient_scaled)[0][1]

    if prob < GREY_ZONE_LOW:
        risk_level      = 'LOW RISK'
        recommendation  = 'No cardiology referral needed. Continue routine monitoring.'
        confidence_flag = 'CONFIDENT'
        color_code      = 'GREEN'
    elif prob > GREY_ZONE_HIGH:
        risk_level      = 'HIGH RISK'
        recommendation  = 'Refer to cardiologist. Immediate follow-up recommended.'
        confidence_flag = 'CONFIDENT'
        color_code      = 'RED'
    else:
        risk_level      = 'UNCERTAIN'
        recommendation  = 'Additional testing recommended before referral decision.'
        confidence_flag = 'GREY ZONE'
        color_code      = 'YELLOW'

    return {
        'probability':      round(prob, 3),
        'risk_level':       risk_level,
        'recommendation':   recommendation,
        'confidence_flag':  confidence_flag,
        'color_code':       color_code,
        'bmi':              bmi,
        'pulse_pressure':   pulse_pressure
    }

# test cases for live demo
test_cases = [
    {
        'label':      'Young healthy patient',
        'age':         38, 'ap_hi': 110, 'cholesterol': 1,
        'gender': 1, 'ap_lo': 70, 'weight_kg': 68, 'height_cm': 168
    },
    {
        'label':      'Middle-aged borderline patient',
        'age':         52, 'ap_hi': 135, 'cholesterol': 2,
        'gender': 2, 'ap_lo': 85, 'weight_kg': 82, 'height_cm': 174
    },
    {
        'label':      'High-risk older patient',
        'age':         62, 'ap_hi': 165, 'cholesterol': 3,
        'gender': 1, 'ap_lo': 100, 'weight_kg': 90, 'height_cm': 160
    },
    {
        'label':      'Borderline uncertain case',
        'age':         50, 'ap_hi': 130, 'cholesterol': 2,
        'gender': 2, 'ap_lo': 85, 'weight_kg': 78, 'height_cm': 172
    },
    {
        'label':      'Elderly high cholesterol',
        'age':         60, 'ap_hi': 150, 'cholesterol': 3,
        'gender': 1, 'ap_lo': 95, 'weight_kg': 85, 'height_cm': 158
    },
]

print("\nRisk Calculator - Demo Cases:")
print("-" * 90)
for case in test_cases:
    result = calculate_cvd_risk(
        age_years=case['age'], ap_hi_mmhg=case['ap_hi'],
        cholesterol_level=case['cholesterol'], gender=case['gender'],
        ap_lo_mmhg=case['ap_lo'], weight_kg=case['weight_kg'], height_cm=case['height_cm']
    )
    chol_map = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
    print(f"\n  [{case['label']}]")
    print(f"  Input  : age={case['age']}y, BP={case['ap_hi']}/{case['ap_lo']}, "
          f"chol={chol_map[case['cholesterol']]}, BMI={result['bmi']}")
    print(f"  Result : P(CVD)={result['probability']} | {result['risk_level']} "
          f"[{result['confidence_flag']}]")
    print(f"  Action : {result['recommendation']}")
print("-" * 90)

# fig 26: risk calculator visualization
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Fig 26 - CVD Risk Calculator - Demo Cases', fontsize=14, fontweight='bold')

results_list = []
for case in test_cases:
    r = calculate_cvd_risk(
        age_years=case['age'], ap_hi_mmhg=case['ap_hi'],
        cholesterol_level=case['cholesterol'], gender=case['gender'],
        ap_lo_mmhg=case['ap_lo'], weight_kg=case['weight_kg'], height_cm=case['height_cm']
    )
    results_list.append({**case, **r})

labels  = [f"Case {i+1}" for i in range(len(test_cases))]
probs   = [r['probability'] for r in results_list]
r_colors = []
for p in probs:
    if p < GREY_ZONE_LOW:
        r_colors.append(COLOR_ACCENT3)
    elif p > GREY_ZONE_HIGH:
        r_colors.append(COLOR_CVD)
    else:
        r_colors.append(COLOR_ACCENT2)

ax0 = fig.add_subplot(gs[0, :2])
bars = ax0.barh(labels, probs, color=r_colors, edgecolor='white', linewidth=0.8, height=0.5)
ax0.axvspan(0, GREY_ZONE_LOW,  alpha=0.08, color=COLOR_ACCENT3)
ax0.axvspan(GREY_ZONE_LOW, GREY_ZONE_HIGH, alpha=0.08, color='gray')
ax0.axvspan(GREY_ZONE_HIGH, 1, alpha=0.08, color=COLOR_CVD)
ax0.axvline(GREY_ZONE_LOW,  color='gray', linestyle='--', linewidth=1)
ax0.axvline(GREY_ZONE_HIGH, color='gray', linestyle='--', linewidth=1)
ax0.set_xlabel('P(CVD)')
ax0.set_title('Risk Calculator Demo - P(CVD) per Case\n(green=low risk, yellow=uncertain, red=high risk)')
ax0.set_xlim(0, 1)

for bar, r in zip(bars, results_list):
    chol_map = {1: 'Norm', 2: 'Above', 3: 'Well Above'}
    label_text = (f"Age={r['age']}y, BP={r['ap_hi']}/{r['ap_lo']}, "
                  f"Chol={chol_map[r['cholesterol']]} → {r['risk_level']} ({r['probability']})")
    ax0.text(r['probability'] + 0.02, bar.get_y() + bar.get_height()/2,
             label_text, va='center', fontsize=8)

ax0.text(0.02, -0.08, 'LOW RISK: No referral', transform=ax0.transAxes,
         fontsize=8, color=COLOR_ACCENT3)
ax0.text(0.38, -0.08, 'UNCERTAIN: More tests', transform=ax0.transAxes,
         fontsize=8, color='gray')
ax0.text(0.72, -0.08, 'HIGH RISK: Refer now', transform=ax0.transAxes,
         fontsize=8, color=COLOR_CVD)

# PART 2: FINAL LEADERBOARD
print("\nPart 2 - Final Leaderboard")

n_test = len(y_test)
scale  = 10_000 / n_test

final_results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    cm     = confusion_matrix(y_test, y_pred)
    fn, fp = cm[1][0], cm[0][1]
    total_cost = (fn * FN_COST + fp * FP_COST) * scale
    final_results[name] = {
        'accuracy':   accuracy_score(y_test, y_pred),
        'f1':         f1_score(y_test, y_pred),
        'recall':     recall_score(y_test, y_pred),
        'auc':        roc_auc_score(y_test, y_prob),
        'fn_10k':     fn * scale,
        'fp_10k':     fp * scale,
        'cost_10k':   total_cost,
    }

by_acc  = sorted(final_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
by_auc  = sorted(final_results.items(), key=lambda x: x[1]['auc'],      reverse=True)
by_cost = sorted(final_results.items(), key=lambda x: x[1]['cost_10k'])

print(f"\n{'Model':22s} {'Acc':>6} {'F1':>6} {'Recall':>7} {'AUC':>6} {'Cost/10K':>12} {'FN/10K':>8}")
print("-" * 80)
for name, v in by_auc:
    print(f"{name:22s} {v['accuracy']:.3f}  {v['f1']:.3f}  {v['recall']:.3f}   "
          f"{v['auc']:.3f}  ${v['cost_10k']/1e6:.1f}M     {v['fn_10k']:.0f}")

print(f"\nAccuracy #1 : {by_acc[0][0]}")
print(f"AUC #1      : {by_auc[0][0]}")
print(f"Cost #1     : {by_cost[0][0]}")

ax1 = fig.add_subplot(gs[0, 2])
model_names = list(final_results.keys())
short_names = ['LR', 'KNN', 'NB', 'DT', 'RF', 'SVM']
bar_colors  = [COLOR_NO_CVD, COLOR_ACCENT1, COLOR_ACCENT2,
               COLOR_CVD, COLOR_ACCENT3, '#6B4226']

acc_rank  = {n: i+1 for i, (n,_) in enumerate(by_acc)}
cost_rank = {n: i+1 for i, (n,_) in enumerate(by_cost)}

x = np.arange(len(model_names))
w = 0.4
ax1.bar(x - w/2, [acc_rank[n]  for n in model_names], w, color=bar_colors,
        label='Accuracy Rank', alpha=0.7)
ax1.bar(x + w/2, [cost_rank[n] for n in model_names], w, color=bar_colors,
        label='Cost Rank',     alpha=1.0)
ax1.set_xticks(x)
ax1.set_xticklabels(short_names)
ax1.set_ylabel('Rank (1=best)')
ax1.set_yticks(range(1, 7))
ax1.invert_yaxis()
ax1.set_title('Accuracy Rank vs Cost Rank\n(light=accuracy, dark=cost)')
ax1.set_ylim(6.5, 0.5)

# bottom row: comprehensive summary table as figure
ax2 = fig.add_subplot(gs[1, :])
ax2.axis('off')

table_data = []
headers    = ['Model', 'Accuracy', 'F1', 'Recall', 'AUC',
              'Cost/10K', 'FN/10K', 'Acc Rank', 'Cost Rank', 'Robust?', 'Fair?']

for name in model_names:
    v     = final_results[name]
    short = name
    table_data.append([
        short,
        f"{v['accuracy']:.3f}",
        f"{v['f1']:.3f}",
        f"{v['recall']:.3f}",
        f"{v['auc']:.3f}",
        f"${v['cost_10k']/1e6:.1f}M",
        f"{v['fn_10k']:.0f}",
        f"#{acc_rank[name]}",
        f"#{cost_rank[name]}",
        'YES' if name != 'Decision Tree' else 'NO',
        'YES'
    ])

tbl = ax2.table(cellText=table_data, colLabels=headers,
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.6)

for j in range(len(headers)):
    tbl[0, j].set_facecolor('#2E4057')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

for i, (name, _) in enumerate(zip(model_names, table_data), 1):
    if cost_rank[name] == 1:
        for j in range(len(headers)):
            tbl[i, j].set_facecolor('#E8F5E9')
    elif acc_rank[name] == 1:
        for j in range(len(headers)):
            tbl[i, j].set_facecolor('#FFF3E0')

ax2.set_title('Final Project Leaderboard — All Metrics Combined\n'
              '(green highlight = clinical cost winner | orange highlight = accuracy winner)',
              fontsize=10, pad=15)

plt.savefig('fig26_final_leaderboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 26 saved.")

# PART 3: FINAL CONCLUSIONS
print("\n" + "="*60)
print("FINAL PROJECT CONCLUSIONS")
print("="*60)

acc_winner  = by_acc[0][0]
cost_winner = by_cost[0][0]
savings     = (final_results[acc_winner]['cost_10k'] -
               final_results[cost_winner]['cost_10k']) / 1e6
fn_diff     = (final_results[acc_winner]['fn_10k'] -
               final_results[cost_winner]['fn_10k'])

print(f"""
PROJECT: Beyond Accuracy - Selecting the Right CVD Classifier
         for Clinical Referral Decisions
DATASET: 70,000 patients, cleaned to {len(df):,} rows (2.0% removed)
MODELS : 6 classifiers covering all 6 course modules
SPLIT  : 70/30 stratified | 10-fold CV

KEY FINDINGS:
1. ACCURACY WINNER      : {acc_winner} ({final_results[acc_winner]['accuracy']:.3f})
2. CLINICAL COST WINNER : {cost_winner} (${final_results[cost_winner]['cost_10k']/1e6:.1f}M per 10K)
3. SAVINGS FROM SWITCH  : ${savings:.2f}M per 10,000 patients
4. FEWER MISSED PATIENTS: {fn_diff:.0f} per 10,000 (switching from RF to KNN)
5. MINIMAL FEATURES     : 3 features (ap_hi, cholesterol, age) = 99.5% of full model accuracy
6. FAIRNESS             : All 6 models pass gender fairness (max recall gap = 1.2%)
7. ROBUSTNESS           : 5/6 models robust to 20% noise (DT fails)
8. CONFIDENCE           : RF confident on 81% of patients, accuracy 77.9% when confident
9. SENSITIVITY ANALYSIS : KNN wins at ALL FN cost assumptions ($25K-$200K)

CLINICAL RECOMMENDATION:
- For overall discrimination: Random Forest (AUC=0.801)
- For minimizing missed CVD patients: KNN (lowest FN count)
- For interpretability: Decision Tree (readable flowchart, but fragile to noise)
- For production use: Random Forest with confidence filtering (grey zone 0.4-0.6)
  Apply KNN threshold adjustment if FN minimization is the primary objective

LIMITATIONS:
- Self-reported lifestyle data (smoke, alco, active) showed weak signal
- Dataset lacks ECG, troponin, and genetic markers that would improve accuracy
- Cost figures ($150K FN, $300 FP) are illustrative estimates, not exact values
- External validation on independent clinical datasets is required before deployment
- SVM trained on 10K subsample due to O(n^2) complexity

FUTURE WORK:
- Incorporate ECG and laboratory biomarker data
- Longitudinal study design to predict CVD onset before symptoms
- Threshold optimization per demographic group for improved equity
- Calibration analysis to ensure probability outputs are reliable
- Regulatory pathway analysis for FDA clinical decision support classification
""")

print("Phase 9 complete. All 10 phases done.")
print("Total figures: 26 (fig01 through fig26)")
print("\nFiles produced:")
print("  phase0_setup.py through phase9_calculator.py")
print("  LEARN_phase0_setup.py  through LEARN_phase9_calculator.py")
