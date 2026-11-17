"""The phase-1 sanity checks, written as assertions.

Synthetic rows are used so the tests run without the Kaggle file; if
cardio_train.csv is present, the real cleaned data is checked too.
"""
from pathlib import Path

import pandas as pd
import pytest

from cvd_data import BMI_MAX, RANGES, clean, load_clean

GOOD = dict(id=1, age=50 * 365, gender=1, height=165, weight=70, ap_hi=120, ap_lo=80,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1, cardio=0)


def _row(**kw):
    return {**GOOD, **kw}


@pytest.mark.parametrize("bad", [
    dict(ap_hi=16020),           # the 16,020 mmHg reading found in phase 1
    dict(height=250),            # 250 cm heights
    dict(ap_lo=-70),             # negative diastolic
    dict(ap_hi=80, ap_lo=120),   # diastolic above systolic
    dict(weight=10),
])
def test_impossible_values_removed(bad):
    out = clean(pd.DataFrame([_row(), _row(id=2, **bad)]))
    assert len(out) == 1


def test_duplicates_removed_ignoring_id():
    out = clean(pd.DataFrame([_row(id=1), _row(id=2)]))
    assert len(out) == 1


def test_age_converted_from_days():
    out = clean(pd.DataFrame([_row()]))
    assert 49 < out.loc[0, 'age'] < 51


real = Path('cardio_train.csv')


@pytest.mark.skipif(not real.exists(), reason="Kaggle file not downloaded")
def test_real_data_within_ranges():
    df = load_clean(str(real))
    for col, (lo, hi) in RANGES.items():
        assert df[col].between(lo, hi).all(), col
    assert (df['ap_hi'] > df['ap_lo']).all()
    assert (df['bmi'] <= BMI_MAX).all()
    assert df['cardio'].isin([0, 1]).all()
    assert len(df) > 60_000
