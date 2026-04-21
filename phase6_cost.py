# CVD Project - Phase 6: Cost-Sensitive Clinical Evaluation
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

RANDOM_STATE  = 42
COLOR_NO_CVD  = '#2E86AB'
COLOR_CVD     = '#C73E1D'
COLOR_ACCENT1 = '#A23B72'
COLOR_ACCENT2 = '#F18F01'
COLOR_ACCENT3 = '#2BA84A'
COLORS_TARGET = [COLOR_NO_CVD, COLOR_CVD]
LABELS_TARGET = ['No CVD', 'Has CVD']

FN_COST = 150_000
FP_COST = 300

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

results = {}
for name, model in models.items():
    X_tr = X_train_svm if name == 'SVM (RBF)' else X_train_scaled
    y_tr = y_train_svm if name == 'SVM (RBF)' else y_train
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test_scaled)
    cm     = confusion_matrix(y_test, y_pred)
    fn, fp = cm[1][0], cm[0][1]
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'fn': fn, 'fp': fp,
        'recall': cm[1][1] / (cm[1][0] + cm[1][1]),
        'total_cost': fn * FN_COST + fp * FP_COST,
        'fn_cost': fn * FN_COST,
        'fp_cost': fp * FP_COST,
        'y_pred': y_pred
    }
    print(f"  {name:22s} FN={fn:,}  FP={fp:,}  cost=${results[name]['total_cost']/1e6:.2f}M")

# scale to per-10,000 patients for clean presentation
n_test     = len(y_test)
scale      = 10_000 / n_test

print(f"\nResults scaled to per 10,000 patients (test set = {n_test:,}):")

cost_data = {}
for name, v in results.items():
    fn_scaled   = v['fn'] * scale
    fp_scaled   = v['fp'] * scale
    cost_scaled = fn_scaled * FN_COST + fp_scaled * FP_COST
    cost_data[name] = {
        'accuracy':     v['accuracy'],
        'fn_per_10k':   fn_scaled,
        'fp_per_10k':   fp_scaled,
        'cost_per_10k': cost_scaled,
        'fn_cost_10k':  fn_scaled * FN_COST,
        'fp_cost_10k':  fp_scaled * FP_COST
    }
    print(f"  {name:22s} acc={v['accuracy']:.3f}  "
          f"FN={fn_scaled:.0f}  FP={fp_scaled:.0f}  "
          f"cost=${cost_scaled/1e6:.2f}M")

# sort by accuracy and by cost for the leaderboard flip
by_accuracy = sorted(cost_data.items(), key=lambda x: x[1]['accuracy'], reverse=True)
by_cost     = sorted(cost_data.items(), key=lambda x: x[1]['cost_per_10k'])

print("\nAccuracy leaderboard:")
for rank, (name, v) in enumerate(by_accuracy, 1):
    print(f"  #{rank} {name:22s} acc={v['accuracy']:.3f}")

print("\nClinical cost leaderboard (lower cost = safer):")
for rank, (name, v) in enumerate(by_cost, 1):
    print(f"  #{rank} {name:22s} cost=${v['cost_per_10k']/1e6:.2f}M per 10K patients")

# find the flip
acc_winner  = by_accuracy[0][0]
cost_winner = by_cost[0][0]
print(f"\nAccuracy winner : {acc_winner}")
print(f"Cost winner     : {cost_winner}")
if acc_winner != cost_winner:
    savings = by_accuracy[0][1]['cost_per_10k'] - by_cost[0][1]['cost_per_10k']
    print(f"Switching from {acc_winner} to {cost_winner} saves")
    print(f"  ${savings/1e6:.2f}M per 10,000 patients")
    print(f"  = ${savings/1e3:.0f}K per 1,000 patients")

# fig 20: the main cost analysis - two leaderboards side by side
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Fig 20 - Cost-Sensitive Clinical Evaluation', fontsize=15, fontweight='bold')

model_names  = list(cost_data.keys())
short_names  = ['LR', 'KNN', 'NB', 'DT', 'RF', 'SVM']
bar_colors   = [COLOR_NO_CVD, COLOR_ACCENT1, COLOR_ACCENT2,
                COLOR_CVD, COLOR_ACCENT3, '#6B4226']

# panel 1: accuracy ranking
ax0 = fig.add_subplot(gs[0, 0])
acc_vals  = [cost_data[n]['accuracy'] for n in model_names]
acc_ranks = sorted(range(len(acc_vals)), key=lambda i: acc_vals[i], reverse=True)
colors_acc = [bar_colors[i] for i in range(len(model_names))]
bars = ax0.bar(short_names, acc_vals, color=colors_acc, edgecolor='white', linewidth=0.8)
ax0.set_title('Accuracy Leaderboard', fontweight='bold')
ax0.set_ylabel('Accuracy')
ax0.set_ylim(0.68, 0.77)
for bar, val, rank in zip(bars, acc_vals,
                           [sorted(acc_vals, reverse=True).index(v)+1 for v in acc_vals]):
    ax0.text(bar.get_x() + bar.get_width()/2, val + 0.001,
             f'#{rank}\n{val:.3f}', ha='center', fontsize=8, fontweight='bold')

# panel 2: clinical cost ranking
ax1 = fig.add_subplot(gs[0, 1])
cost_vals = [cost_data[n]['cost_per_10k']/1e6 for n in model_names]
cost_rank_list = sorted(cost_vals)
bars = ax1.bar(short_names, cost_vals, color=colors_acc, edgecolor='white', linewidth=0.8)
ax1.set_title('Clinical Cost Leaderboard\n(lower = safer)', fontweight='bold')
ax1.set_ylabel('Total Cost per 10K patients ($M)')
for bar, val in zip(bars, cost_vals):
    rank = cost_rank_list.index(val) + 1
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3,
             f'#{rank}\n${val:.1f}M', ha='center', fontsize=8, fontweight='bold')

# panel 3: rank change arrows
ax2 = fig.add_subplot(gs[0, 2])
acc_order  = [n for n, _ in by_accuracy]
cost_order = [n for n, _ in by_cost]

for i, name in enumerate(model_names):
    acc_rank  = acc_order.index(name) + 1
    cost_rank = cost_order.index(name) + 1
    color     = bar_colors[i]
    ax2.annotate('', xy=(1, cost_rank), xytext=(0, acc_rank),
                 arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax2.text(-0.05, acc_rank, f'#{acc_rank} {short_names[i]}',
             ha='right', va='center', fontsize=9, color=color, fontweight='bold')
    ax2.text(1.05, cost_rank, f'#{cost_rank} {short_names[i]}',
             ha='left', va='center', fontsize=9, color=color, fontweight='bold')

ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(0.5, len(model_names) + 0.5)
ax2.invert_yaxis()
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Accuracy\nRanking', 'Cost\nRanking'], fontsize=10)
ax2.set_yticks([])
ax2.set_title('Rank Changes:\nAccuracy vs Cost', fontweight='bold')
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)

# panel 4: stacked bar - FN cost vs FP cost per model
ax3 = fig.add_subplot(gs[1, 0])
fn_costs = [cost_data[n]['fn_cost_10k']/1e6 for n in model_names]
fp_costs = [cost_data[n]['fp_cost_10k']/1e6 for n in model_names]
x = np.arange(len(short_names))
ax3.bar(x, fn_costs, color=COLOR_CVD,    label=f'FN cost (${FN_COST:,} each)', alpha=0.85)
ax3.bar(x, fp_costs, bottom=fn_costs, color=COLOR_ACCENT2,
        label=f'FP cost (${FP_COST:,} each)', alpha=0.85)
ax3.set_xticks(x)
ax3.set_xticklabels(short_names)
ax3.set_title('Cost Breakdown: FN vs FP\nper 10K patients', fontweight='bold')
ax3.set_ylabel('Cost ($M)')
ax3.legend(fontsize=8)

# panel 5: cost savings vs accuracy winner
ax4 = fig.add_subplot(gs[1, 1])
baseline_cost = cost_data[acc_winner]['cost_per_10k']
savings_vals  = [(baseline_cost - cost_data[n]['cost_per_10k'])/1e6 for n in model_names]
colors_savings = [COLOR_ACCENT3 if s > 0 else COLOR_CVD for s in savings_vals]
ax4.bar(short_names, savings_vals, color=colors_savings, edgecolor='white', linewidth=0.8)
ax4.axhline(0, color='black', linewidth=0.8)
ax4.set_title(f'Cost Savings vs Accuracy Winner ({acc_winner[:2]})\nper 10K patients',
              fontweight='bold')
ax4.set_ylabel('Savings ($M)  [green = cheaper]')
for i, (bar, val) in enumerate(zip(ax4.patches, savings_vals)):
    ax4.text(i, val + (0.1 if val >= 0 else -0.3),
             f'${val:.1f}M', ha='center', fontsize=8)

# panel 6: FN count comparison - this is the human cost
ax5 = fig.add_subplot(gs[1, 2])
fn_vals   = [cost_data[n]['fn_per_10k'] for n in model_names]
fn_sorted = sorted(fn_vals)
bars = ax5.bar(short_names, fn_vals, color=colors_acc, edgecolor='white', linewidth=0.8)
ax5.set_title('Missed CVD Patients per 10K\n(lower = fewer lives at risk)', fontweight='bold')
ax5.set_ylabel('False Negatives (missed patients)')
for bar, val in zip(bars, fn_vals):
    rank = fn_sorted.index(val) + 1
    ax5.text(bar.get_x() + bar.get_width()/2, val + 5,
             f'#{rank}\n{val:.0f}', ha='center', fontsize=8, fontweight='bold')

plt.savefig('fig20_cost_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 20 saved.")

# fig 21: sensitivity analysis - does the winner change at different FN costs?
print("\nRunning sensitivity analysis on FN cost assumption...")

fn_cost_scenarios = [25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 200_000]
sensitivity_results = {}

for fn_cost_scenario in fn_cost_scenarios:
    scenario_costs = {}
    for name, v in results.items():
        scenario_costs[name] = v['fn'] * fn_cost_scenario * scale + v['fp'] * FP_COST * scale
    sensitivity_results[fn_cost_scenario] = scenario_costs

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Fig 21 - Sensitivity Analysis: Does the Winner Change?', fontsize=14, fontweight='bold')

colors_map = dict(zip(model_names, bar_colors))

# panel 1: cost lines across FN scenarios
ax = axes[0]
for name in model_names:
    costs = [sensitivity_results[fn][name]/1e6 for fn in fn_cost_scenarios]
    ax.plot([f/1000 for f in fn_cost_scenarios], costs,
            color=colors_map[name], linewidth=2, marker='o', markersize=5,
            label=name)
ax.set_xlabel('False Negative Cost Assumption ($K)')
ax.set_ylabel('Total Cost per 10K patients ($M)')
ax.set_title('Total Cost at Different FN Cost Assumptions')
ax.legend(fontsize=7, loc='upper left')
ax.axvline(FN_COST/1000, color='black', linestyle='--', linewidth=1,
           label=f'Our assumption (${FN_COST/1000:.0f}K)')

# panel 2: winner at each FN cost scenario
ax2 = axes[1]
winners_at_cost = []
winner_costs    = []
for fn_cost_scenario in fn_cost_scenarios:
    scenario = sensitivity_results[fn_cost_scenario]
    winner   = min(scenario, key=scenario.get)
    winners_at_cost.append(winner)
    winner_costs.append(scenario[winner]/1e6)

winner_colors = [colors_map[w] for w in winners_at_cost]
bars = ax2.bar([f'{f/1000:.0f}K' for f in fn_cost_scenarios],
               winner_costs, color=winner_colors, edgecolor='white', linewidth=0.8)
ax2.set_xlabel('False Negative Cost Assumption')
ax2.set_ylabel('Winner Total Cost per 10K ($M)')
ax2.set_title('Cost Winner at Each FN Assumption\n(color = winning model)')

for bar, winner, cost in zip(bars, winners_at_cost, winner_costs):
    short = winner[:3] if winner != 'Logistic Regression' else 'LR'
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{short}\n${cost:.0f}M', ha='center', fontsize=8, fontweight='bold')

legend_patches = [plt.Rectangle((0,0),1,1, color=colors_map[n], label=n)
                  for n in set(winners_at_cost)]
ax2.legend(handles=legend_patches, fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('fig21_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 21 saved.")

print(f"\nSensitivity analysis summary:")
print(f"{'FN Cost':>12}  {'Winner':25}  {'Cost per 10K':>15}")
for fn_cost_scenario, winner in zip(fn_cost_scenarios, winners_at_cost):
    cost = sensitivity_results[fn_cost_scenario][winner]/1e6
    print(f"  ${fn_cost_scenario:>8,}  {winner:25}  ${cost:.2f}M")

print(f"\nPhase 6 complete.")
print(f"The accuracy winner is    : {acc_winner}")
print(f"The clinical cost winner  : {cost_winner}")
if acc_winner != cost_winner:
    savings = (cost_data[acc_winner]['cost_per_10k'] - cost_data[cost_winner]['cost_per_10k'])
    print(f"Switching saves           : ${savings/1e6:.2f}M per 10,000 patients")
    fn_diff = cost_data[acc_winner]['fn_per_10k'] - cost_data[cost_winner]['fn_per_10k']
    print(f"Fewer missed patients     : {fn_diff:.0f} per 10,000")
print(f"\nThis is the core finding of the entire project.")
