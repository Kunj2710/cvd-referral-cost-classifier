# CVD Project - Phase 1: Data Loading & First Inspection
# BU MET CS 577 | Spring 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE  = 42
COLOR_NO_CVD  = '#2E86AB'
COLOR_CVD     = '#C73E1D'
COLORS_TARGET = [COLOR_NO_CVD, COLOR_CVD]
LABELS_TARGET = ['No CVD', 'Has CVD']

DATA_PATH = 'cardio_train.csv'

df = pd.read_csv(DATA_PATH, sep=';')

print("Phase 1 - Data Loading & First Inspection")

# shape
print(f"\n[1] Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# column names
print(f"\n[2] Columns: {list(df.columns)}")

# data types
print(f"\n[3] Data types:\n{df.dtypes}")

# missing values
missing = df.isnull().sum()
print(f"\n[4] Missing values:\n{missing}")
print(f"    Total missing: {missing.sum()}")

# duplicates
dupes = df.duplicated().sum()
print(f"\n[5] Duplicate rows: {dupes:,}")

# target balance
target_counts = df['cardio'].value_counts()
target_pct    = df['cardio'].value_counts(normalize=True) * 100
print(f"\n[6] Target balance:")
for label, count, pct in zip(LABELS_TARGET, target_counts, target_pct):
    print(f"    {label}: {count:,} ({pct:.1f}%)")

# descriptive stats
print(f"\n[7] Descriptive statistics:")
print(df.describe().round(2).to_string())

# red flags
print(f"\n[8] Red flags:")

age_min = df['age'].min()
age_max = df['age'].max()
print(f"    age stored in days - min: {age_min}, max: {age_max}")
print(f"    that converts to {age_min/365.25:.1f} to {age_max/365.25:.1f} years")

ap_hi_issues = ((df['ap_hi'] < 0) | (df['ap_hi'] > 400)).sum()
ap_lo_issues = ((df['ap_lo'] < 0) | (df['ap_lo'] > 400)).sum()
print(f"    ap_hi out of range (<0 or >400): {ap_hi_issues:,} rows")
print(f"    ap_lo out of range (<0 or >400): {ap_lo_issues:,} rows")
print(f"    ap_hi max value: {df['ap_hi'].max()}, ap_lo min value: {df['ap_lo'].min()}")

bp_inverted = (df['ap_hi'] <= df['ap_lo']).sum()
print(f"    systolic <= diastolic (physically impossible): {bp_inverted:,} rows")

height_issues = ((df['height'] < 100) | (df['height'] > 220)).sum()
weight_issues = ((df['weight'] < 30)  | (df['weight'] > 200)).sum()
print(f"    height out of range: {height_issues:,} rows")
print(f"    weight out of range: {weight_issues:,} rows")

print(f"\n    rough total rows flagged: ~{dupes + ap_hi_issues + ap_lo_issues + bp_inverted:,}")

# figure 1 - raw data snapshot before any cleaning
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Phase 1 - Raw Data Snapshot (Before Cleaning)', fontsize=14, fontweight='bold')

# panel 1: target balance
axes[0].bar(LABELS_TARGET, target_counts.values, color=COLORS_TARGET, edgecolor='white', linewidth=0.8)
axes[0].set_title('Target Balance')
axes[0].set_ylabel('Patient Count')
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 200, f'{v:,}\n({target_pct.values[i]:.1f}%)', ha='center', fontsize=9)

# panel 2: age in raw days, years shown on top axis
axes[1].hist(df['age'], bins=40, color=COLOR_NO_CVD, edgecolor='white', linewidth=0.5)
axes[1].set_title('Age Distribution (raw, in days)')
axes[1].set_xlabel('Age (days)')
axes[1].set_ylabel('Count')
ax1_twin = axes[1].twiny()
ax1_twin.set_xlim(axes[1].get_xlim())
ax1_twin.set_xticks(axes[1].get_xticks())
ax1_twin.set_xticklabels([f'{x/365.25:.0f}y' for x in axes[1].get_xticks()], fontsize=7)
ax1_twin.set_xlabel('Age (years)', fontsize=9)

# panel 3: blood pressure scatter showing how bad the outliers are
axes[2].scatter(df['ap_hi'], df['ap_lo'], alpha=0.05, s=1, color=COLOR_CVD)
axes[2].axvline(x=250, color='black', linestyle='--', linewidth=1, label='cutoff 250')
axes[2].axhline(y=150, color='gray',  linestyle='--', linewidth=1, label='cutoff 150')
axes[2].set_title('Blood Pressure - Raw (outliers visible)')
axes[2].set_xlabel('Systolic (ap_hi)')
axes[2].set_ylabel('Diastolic (ap_lo)')
axes[2].legend(fontsize=8)
axes[2].set_xlim(-200, 500)
axes[2].set_ylim(-50, 400)

plt.tight_layout()
plt.savefig('fig01_raw_data_snapshot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFig 1 saved.")
print("Phase 1 done. Next: Phase 2 - Cleaning.")
