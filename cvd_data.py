"""Shared loading and cleaning, matching the inline cleaning in phases 4-9.

Kept in one place so the sanity checks can be tested (see tests/).
"""
import pandas as pd

FEATURE_COLS = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                'bmi', 'pulse_pressure']
SCALE_COLS = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']

# Plausible clinical ranges used by the sanity checks
RANGES = {'ap_hi': (70, 250), 'ap_lo': (40, 150), 'height': (100, 220), 'weight': (30, 200)}
BMI_MAX = 60


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['id'], errors='ignore').drop_duplicates()
    df = df.assign(age=(df['age'] / 365.25).round(1))
    for col, (lo, hi) in RANGES.items():
        df = df[(df[col] >= lo) & (df[col] <= hi)]
    df = df[df['ap_hi'] > df['ap_lo']]
    df = df.assign(bmi=(df['weight'] / (df['height'] / 100) ** 2).round(1))
    df = df[df['bmi'] <= BMI_MAX]
    df = df.assign(pulse_pressure=df['ap_hi'] - df['ap_lo'])
    return df.reset_index(drop=True)


def load_clean(path: str = 'cardio_train.csv') -> pd.DataFrame:
    return clean(pd.read_csv(path, sep=';'))
