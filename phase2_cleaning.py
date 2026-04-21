# CVD Project - Phase 2: Data Cleaning
# BU MET CS 677 | Spring 2026

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
COLORS_TARGET = [COLOR_NO_CVD, COLOR_CVD]
LABELS_TARGET = ['No CVD', 'Has CVD']

DATA_PATH = 'cardio_train.csv'

df = pd.read_csv(DATA_PATH, sep=';')
rows_original = len(df)
print(f"Starting rows: {rows_original:,}")

# step 1: drop id
df.drop(columns=['id'], inplace=True)
print(f"\nStep 1 - dropped id column")

# step 2: duplicates (check AFTER dropping id so identical patients are caught)
dupes = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"Step 2 - removed {dupes:,} duplicates | rows remaining: {len(df):,}")

# step 3: age days to years
df['age'] = (df['age'] / 365.25).round(1)
print(f"Step 3 - age converted to years | range: {df['age'].min()} to {df['age'].max()}")

# step 4: blood pressure - 3 layer filter
before = len(df)
df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 250)]
print(f"\nStep 4a - ap_hi range filter (70-250) | removed: {before - len(df):,} rows")

before = len(df)
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
print(f"Step 4b - ap_lo range filter (40-150) | removed: {before - len(df):,} rows")

before = len(df)
df = df[df['ap_hi'] > df['ap_lo']]
print(f"Step 4c - ap_hi > ap_lo filter | removed: {before - len(df):,} rows")

# step 5: height and weight
before = len(df)
df = df[(df['height'] >= 100) & (df['height'] <= 220)]
print(f"\nStep 5a - height filter (100-220cm) | removed: {before - len(df):,} rows")

before = len(df)
df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
print(f"Step 5b - weight filter (30-200kg) | removed: {before - len(df):,} rows")

# step 6: engineer BMI + cap at 60 (anything above is a data error)
df['bmi'] = (df['weight'] / (df['height'] / 100) ** 2).round(1)
before = len(df)
df = df[df['bmi'] <= 60]
print(f"\nStep 6a - BMI cap at 60 | removed: {before - len(df):,} rows")

bmi_bins   = [0, 18.5, 25.0, 30.0, float('inf')]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels)
print(f"Step 6b - BMI category added | range: {df['bmi'].min()} to {df['bmi'].max()}")

# step 7: pulse pressure
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
print(f"Step 7 - pulse pressure engineered | range: {df['pulse_pressure'].min()} to {df['pulse_pressure'].max()}")

df.reset_index(drop=True, inplace=True)

rows_removed = rows_original - len(df)
print(f"\nCleaning complete.")
print(f"  Original rows : {rows_original:,}")
print(f"  Rows removed  : {rows_removed:,} ({rows_removed/rows_original*100:.1f}%)")
print(f"  Clean rows    : {len(df):,}")
print(f"  Columns now   : {list(df.columns)}")

target_counts = df['cardio'].value_counts()
target_pct    = df['cardio'].value_counts(normalize=True) * 100
print(f"\nTarget balance after cleaning:")
for label, count, pct in zip(LABELS_TARGET, target_counts, target_pct):
    print(f"  {label}: {count:,} ({pct:.1f}%)")

# figure 2 - cleaning results
df_raw = pd.read_csv(DATA_PATH, sep=';')

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle('Phase 2 - Cleaning Results', fontsize=14, fontweight='bold')

# panel 1: rows kept
ax0 = fig.add_subplot(gs[0, 0])
bars = ax0.bar(['Original', 'After Cleaning'],
               [rows_original, len(df)],
               color=[COLOR_CVD, COLOR_NO_CVD],
               edgecolor='white', linewidth=0.8)
ax0.set_title('Rows Kept After Cleaning')
ax0.set_ylabel('Row Count')
ax0.set_ylim(0, rows_original * 1.1)
for bar, val in zip(bars, [rows_original, len(df)]):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:,}', ha='center', fontsize=9)

# panel 2: systolic BP - show raw outlier range on left, clean distribution on right
ax1 = fig.add_subplot(gs[0, 1])
# raw data in full range to show the outliers
ax1.hist(df_raw['ap_hi'].clip(upper=500), bins=80,
         alpha=0.5, color=COLOR_CVD, label='Before (clipped at 500)', edgecolor='none')
ax1.hist(df['ap_hi'], bins=80,
         alpha=0.8, color=COLOR_NO_CVD, label='After', edgecolor='none')
ax1.axvline(x=250, color='black', linestyle='--', linewidth=1, label='cutoff 250')
ax1.set_title('Systolic BP Before vs After')
ax1.set_xlabel('ap_hi (mmHg)')
ax1.set_ylabel('Count')
ax1.set_xlim(0, 500)
ax1.legend(fontsize=7)

# panel 3: diastolic BP
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(df_raw['ap_lo'].clip(lower=-100, upper=300), bins=80,
         alpha=0.5, color=COLOR_CVD, label='Before (clipped at 300)', edgecolor='none')
ax2.hist(df['ap_lo'], bins=80,
         alpha=0.8, color=COLOR_NO_CVD, label='After', edgecolor='none')
ax2.axvline(x=150, color='black', linestyle='--', linewidth=1, label='cutoff 150')
ax2.axvline(x=40,  color='gray',  linestyle='--', linewidth=1, label='cutoff 40')
ax2.set_title('Diastolic BP Before vs After')
ax2.set_xlabel('ap_lo (mmHg)')
ax2.set_ylabel('Count')
ax2.set_xlim(-100, 300)
ax2.legend(fontsize=7)

# panel 4: age distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(df['age'], bins=40, color=COLOR_NO_CVD, edgecolor='white', linewidth=0.5)
ax3.set_title('Age Distribution (years, clean)')
ax3.set_xlabel('Age (years)')
ax3.set_ylabel('Count')

# panel 5: BMI by CVD status
ax4 = fig.add_subplot(gs[1, 1])
for val, label, color in zip([0, 1], LABELS_TARGET, COLORS_TARGET):
    ax4.hist(df[df['cardio'] == val]['bmi'], bins=40,
             alpha=0.6, color=color, label=label, edgecolor='none')
ax4.set_title('BMI by CVD Status')
ax4.set_xlabel('BMI')
ax4.set_ylabel('Count')
ax4.set_xlim(10, 60)
ax4.legend()

# panel 6: BMI category breakdown
ax5 = fig.add_subplot(gs[1, 2])
bmi_cat_counts = df['bmi_category'].value_counts().reindex(bmi_labels)
colors_bmi = [COLOR_NO_CVD, COLOR_ACCENT1, COLOR_ACCENT2, COLOR_CVD]
ax5.bar(bmi_cat_counts.index, bmi_cat_counts.values,
        color=colors_bmi, edgecolor='white', linewidth=0.8)
ax5.set_title('BMI Category Breakdown')
ax5.set_ylabel('Count')
for i, v in enumerate(bmi_cat_counts.values):
    ax5.text(i, v + 100, f'{v:,}', ha='center', fontsize=8)

plt.savefig('fig02_cleaning_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFig 2 saved.")
print("Phase 2 done. Next: Phase 3 - EDA.")
