# CVD Project - Phase 3: Exploratory Data Analysis
# BU MET CS 577 | Spring 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE  = 42
COLOR_NO_CVD  = '#2E86AB'
COLOR_CVD     = '#C73E1D'
COLOR_ACCENT1 = '#A23B72'
COLOR_ACCENT2 = '#F18F01'
COLOR_ACCENT3 = '#2BA84A'
COLORS_TARGET = [COLOR_NO_CVD, COLOR_CVD]
LABELS_TARGET = ['No CVD', 'Has CVD']

DATA_PATH = 'cardio_train.csv'

# run cleaning inline so Phase 3 is self-contained
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

print(f"Clean dataset loaded: {len(df):,} rows, {df.shape[1]} columns")

cvd    = df[df['cardio'] == 1]
no_cvd = df[df['cardio'] == 0]

# fig 3: age analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 3 - Age vs CVD Status', fontsize=14, fontweight='bold')

for val, label, color in zip([0, 1], LABELS_TARGET, COLORS_TARGET):
    axes[0].hist(df[df['cardio'] == val]['age'], bins=35,
                 alpha=0.6, color=color, label=label, edgecolor='none')
axes[0].set_title('Age Distribution by CVD Status')
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].axvline(no_cvd['age'].median(), color=COLOR_NO_CVD, linestyle='--', linewidth=1.5,
                label=f'No CVD median: {no_cvd["age"].median():.1f}y')
axes[0].axvline(cvd['age'].median(),    color=COLOR_CVD,    linestyle='--', linewidth=1.5,
                label=f'CVD median: {cvd["age"].median():.1f}y')
axes[0].legend(fontsize=8)

axes[1].boxplot([no_cvd['age'], cvd['age']],
                labels=LABELS_TARGET,
                patch_artist=True,
                boxprops=dict(facecolor='white'),
                medianprops=dict(color='black', linewidth=2))
for patch, color in zip(axes[1].findobj(plt.matplotlib.patches.PathPatch), COLORS_TARGET):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_title('Age Box Plot by CVD Status')
axes[1].set_ylabel('Age (years)')

age_diff = cvd['age'].median() - no_cvd['age'].median()
axes[1].text(1.5, df['age'].quantile(0.95),
             f'median diff: {age_diff:.1f} years', fontsize=9, color='black')

plt.tight_layout()
plt.savefig('fig03_age_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 3 saved.")

# fig 4: blood pressure analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Fig 4 - Blood Pressure vs CVD Status', fontsize=14, fontweight='bold')

# systolic distribution
for val, label, color in zip([0, 1], LABELS_TARGET, COLORS_TARGET):
    axes[0].hist(df[df['cardio'] == val]['ap_hi'], bins=40,
                 alpha=0.6, color=color, label=label, edgecolor='none')
axes[0].set_title('Systolic BP by CVD Status')
axes[0].set_xlabel('Systolic BP (mmHg)')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].axvline(120, color='gray', linestyle=':', linewidth=1, label='normal <120')
axes[0].axvline(140, color='black', linestyle=':', linewidth=1, label='high >=140')
axes[0].legend(fontsize=7)

# diastolic distribution
for val, label, color in zip([0, 1], LABELS_TARGET, COLORS_TARGET):
    axes[1].hist(df[df['cardio'] == val]['ap_lo'], bins=40,
                 alpha=0.6, color=color, label=label, edgecolor='none')
axes[1].set_title('Diastolic BP by CVD Status')
axes[1].set_xlabel('Diastolic BP (mmHg)')
axes[1].set_ylabel('Count')
axes[1].legend()

# pulse pressure by CVD - the engineered feature
axes[2].boxplot([no_cvd['pulse_pressure'], cvd['pulse_pressure']],
                labels=LABELS_TARGET,
                patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
for patch, color in zip(axes[2].findobj(plt.matplotlib.patches.PathPatch), COLORS_TARGET):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[2].set_title('Pulse Pressure by CVD Status')
axes[2].set_ylabel('Pulse Pressure (mmHg)')
axes[2].axhline(60, color='gray', linestyle='--', linewidth=1, label='clinical threshold 60')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('fig04_bp_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 4 saved.")

# fig 5: BMI analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 5 - BMI vs CVD Status', fontsize=14, fontweight='bold')

# violin plot
parts = axes[0].violinplot([no_cvd['bmi'], cvd['bmi']],
                            positions=[1, 2], showmedians=True)
for pc, color in zip(parts['bodies'], COLORS_TARGET):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)
axes[0].set_title('BMI Distribution by CVD Status (Violin)')
axes[0].set_xticks([1, 2])
axes[0].set_xticklabels(LABELS_TARGET)
axes[0].set_ylabel('BMI')
axes[0].axhline(25, color='gray',  linestyle='--', linewidth=1, label='overweight 25')
axes[0].axhline(30, color='black', linestyle='--', linewidth=1, label='obese 30')
axes[0].legend(fontsize=8)

# BMI category CVD rate
bmi_cvd_rate = df.groupby('bmi_category', observed=True)['cardio'].mean() * 100
colors_bmi   = [COLOR_NO_CVD, COLOR_ACCENT1, COLOR_ACCENT2, COLOR_CVD]
bars = axes[1].bar(bmi_cvd_rate.index, bmi_cvd_rate.values,
                   color=colors_bmi, edgecolor='white', linewidth=0.8)
axes[1].set_title('CVD Rate by BMI Category')
axes[1].set_ylabel('CVD Rate (%)')
axes[1].set_ylim(0, 80)
for bar, val in zip(bars, bmi_cvd_rate.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 1,
                 f'{val:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('fig05_bmi_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 5 saved.")

# fig 6: categorical features
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle('Fig 6 - Categorical Features vs CVD Rate', fontsize=14, fontweight='bold')

cat_features = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']
cat_labels   = {
    'cholesterol': ['Normal', 'Above\nNormal', 'Well Above\nNormal'],
    'gluc':        ['Normal', 'Above\nNormal', 'Well Above\nNormal'],
    'smoke':       ['No', 'Yes'],
    'alco':        ['No', 'Yes'],
    'active':      ['No', 'Yes']
}

for ax, feat in zip(axes, cat_features):
    cvd_rate = df.groupby(feat)['cardio'].mean() * 100
    n_cats   = len(cvd_rate)
    colors   = [COLOR_NO_CVD, COLOR_ACCENT2, COLOR_CVD][:n_cats]
    bars     = ax.bar(range(n_cats), cvd_rate.values,
                      color=colors, edgecolor='white', linewidth=0.8)
    ax.set_title(feat.capitalize())
    ax.set_ylabel('CVD Rate (%)')
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(cat_labels[feat], fontsize=8)
    ax.set_ylim(0, 90)
    for bar, val in zip(bars, cvd_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.0f}%', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('fig06_categorical_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 6 saved.")

# fig 7: correlation heatmap
fig, ax = plt.subplots(figsize=(11, 9))
fig.suptitle('Fig 7 - Correlation Heatmap (All Features)', fontsize=14, fontweight='bold')

numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                'bmi', 'pulse_pressure', 'cardio']
corr = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 8})
ax.set_title('Lower Triangle Only (avoids redundancy)', fontsize=10)

plt.tight_layout()
plt.savefig('fig07_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 7 saved.")

# fig 8: gender analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Fig 8 - Gender Analysis (sets up Phase 8 fairness audit)', fontsize=14, fontweight='bold')

gender_map    = {1: 'Female', 2: 'Male'}
df['gender_label'] = df['gender'].map(gender_map)

# gender distribution
gender_counts = df['gender_label'].value_counts()
axes[0].bar(gender_counts.index, gender_counts.values,
            color=[COLOR_ACCENT1, COLOR_NO_CVD], edgecolor='white', linewidth=0.8)
axes[0].set_title('Gender Distribution')
axes[0].set_ylabel('Count')
for i, (label, val) in enumerate(gender_counts.items()):
    axes[0].text(i, val + 200, f'{val:,}\n({val/len(df)*100:.1f}%)',
                 ha='center', fontsize=9)

# CVD rate by gender
cvd_by_gender = df.groupby('gender_label')['cardio'].mean() * 100
axes[1].bar(cvd_by_gender.index, cvd_by_gender.values,
            color=[COLOR_ACCENT1, COLOR_NO_CVD], edgecolor='white', linewidth=0.8)
axes[1].set_title('CVD Rate by Gender')
axes[1].set_ylabel('CVD Rate (%)')
axes[1].set_ylim(0, 80)
for i, (label, val) in enumerate(cvd_by_gender.items()):
    axes[1].text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=10)

# age distribution by gender
for gender, color in zip(['Female', 'Male'], [COLOR_ACCENT1, COLOR_NO_CVD]):
    axes[2].hist(df[df['gender_label'] == gender]['age'], bins=30,
                 alpha=0.6, color=color, label=gender, edgecolor='none')
axes[2].set_title('Age Distribution by Gender')
axes[2].set_xlabel('Age (years)')
axes[2].set_ylabel('Count')
axes[2].legend()

plt.tight_layout()
plt.savefig('fig08_gender_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 8 saved.")

# fig 9: pairplot of top features
top_features = ['age', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure', 'cardio']
sample       = df[top_features].sample(n=2000, random_state=RANDOM_STATE)

palette = {0: COLOR_NO_CVD, 1: COLOR_CVD}
g = sns.pairplot(sample, hue='cardio', palette=palette,
                 plot_kws=dict(alpha=0.4, s=8),
                 diag_kind='kde')
g.figure.suptitle('Fig 9 - Pairplot of Top Features (2,000 sample)', y=1.02, fontsize=13)
for ax in g.axes.flatten():
    if ax:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=c, markersize=8, label=l)
           for c, l in zip(COLORS_TARGET, LABELS_TARGET)]
g.figure.legend(handles=handles, loc='upper right', fontsize=9)

plt.savefig('fig09_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 9 saved.")

# fig 10: CVD rate by age group - the clinical story in one chart
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('Fig 10 - CVD Rate Rises Sharply With Age', fontsize=14, fontweight='bold')

age_bins   = [29, 35, 40, 45, 50, 55, 60, 65]
age_labels = ['30-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

age_cvd_rate = df.groupby('age_group', observed=True)['cardio'].mean() * 100
age_counts   = df.groupby('age_group', observed=True)['cardio'].count()

colors_age = [COLOR_NO_CVD if v < 50 else COLOR_CVD for v in age_cvd_rate.values]
bars = ax.bar(age_labels, age_cvd_rate.values, color=colors_age,
              edgecolor='white', linewidth=0.8)
ax.axhline(50, color='gray', linestyle='--', linewidth=1, label='50% threshold')
ax.set_xlabel('Age Group')
ax.set_ylabel('CVD Rate (%)')
ax.set_ylim(0, 85)
ax.legend(fontsize=9)

for bar, val, n in zip(bars, age_cvd_rate.values, age_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.0f}%\n(n={n:,})', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('fig10_age_cvd_rate.png', dpi=150, bbox_inches='tight')
plt.show()
print("Fig 10 saved.")

print("\nPhase 3 complete - 8 figures saved (fig03 through fig10).")
print("Key findings:")
print(f"  CVD patients are on average older by {cvd['age'].median() - no_cvd['age'].median():.1f} years")
print(f"  CVD patients have higher systolic BP: {cvd['ap_hi'].mean():.0f} vs {no_cvd['ap_hi'].mean():.0f} mmHg")
print(f"  CVD patients have higher BMI: {cvd['bmi'].mean():.1f} vs {no_cvd['bmi'].mean():.1f}")
print(f"  CVD rate by cholesterol - Normal: {df[df['cholesterol']==1]['cardio'].mean()*100:.0f}%, Well Above Normal: {df[df['cholesterol']==3]['cardio'].mean()*100:.0f}%")
