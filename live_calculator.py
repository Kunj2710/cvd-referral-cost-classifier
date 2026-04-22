# CVD Risk Calculator - Interactive Live Demo
# BU MET CS 577 | Spring 2026
# Run this during your presentation
# You will TYPE values and see the result instantly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # change to 'Qt5Agg' if TkAgg doesn't work on your machine
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE   = 42
COLOR_LOW      = '#2BA84A'
COLOR_GREY     = '#F18F01'
COLOR_HIGH     = '#C73E1D'
GREY_ZONE_LOW  = 0.40
GREY_ZONE_HIGH = 0.60

DATA_PATH = 'cardio_train.csv'

# train the model on startup
print("\n" + "="*55)
print("  CVD RISK CALCULATOR - BU MET CS 577")
print("  Loading model... please wait ~15 seconds")
print("="*55)

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
X_train_scaled[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])

rf = RandomForestClassifier(n_estimators=100, max_depth=12,
                             random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

print("  Model ready. Trained on 68,575 patients.")
print("="*55)


def get_input(prompt, type_fn, valid_range=None, valid_values=None):
    """Helper to get validated user input."""
    while True:
        try:
            val = type_fn(input(prompt))
            if valid_range and not (valid_range[0] <= val <= valid_range[1]):
                print(f"  Please enter a value between {valid_range[0]} and {valid_range[1]}")
                continue
            if valid_values and val not in valid_values:
                print(f"  Please enter one of: {valid_values}")
                continue
            return val
        except ValueError:
            print("  Invalid input. Please try again.")


def show_gauge(prob, risk, color, patient_name):
    """Show the visual risk gauge."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor('white')

    # colored background zones
    ax.barh(0, GREY_ZONE_LOW,                      left=0,              height=0.5,
            color=COLOR_LOW,  alpha=0.2, edgecolor='none')
    ax.barh(0, GREY_ZONE_HIGH - GREY_ZONE_LOW,     left=GREY_ZONE_LOW,  height=0.5,
            color=COLOR_GREY, alpha=0.2, edgecolor='none')
    ax.barh(0, 1 - GREY_ZONE_HIGH,                 left=GREY_ZONE_HIGH, height=0.5,
            color=COLOR_HIGH, alpha=0.2, edgecolor='none')

    # probability needle
    ax.scatter([prob], [0], color=color, s=500, zorder=5,
               edgecolors='white', linewidth=2.5)
    ax.axvline(prob, color=color, linewidth=2.5, linestyle='--', alpha=0.7)

    # probability text above needle
    ax.text(prob, 0.33, f'{prob:.1%}',
            ha='center', fontsize=14, fontweight='bold', color=color)

    # zone boundary lines
    ax.axvline(GREY_ZONE_LOW,  color='gray', linewidth=1.2, linestyle=':')
    ax.axvline(GREY_ZONE_HIGH, color='gray', linewidth=1.2, linestyle=':')

    # zone labels below
    ax.text(GREY_ZONE_LOW / 2,          -0.35, 'LOW RISK\n(No Referral)',
            ha='center', va='top', fontsize=10, color=COLOR_LOW,  fontweight='bold')
    ax.text((GREY_ZONE_LOW + GREY_ZONE_HIGH) / 2, -0.35, 'UNCERTAIN\n(More Tests)',
            ha='center', va='top', fontsize=10, color=COLOR_GREY, fontweight='bold')
    ax.text((GREY_ZONE_HIGH + 1) / 2,   -0.35, 'HIGH RISK\n(Refer Now)',
            ha='center', va='top', fontsize=10, color=COLOR_HIGH, fontweight='bold')

    # x-axis ticks
    ax.set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '50%', '60%', '80%', '100%'], fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.65, 0.55)
    ax.set_yticks([])
    ax.set_xlabel('Predicted CVD Probability', fontsize=11)
    ax.set_title(f'{patient_name}   —   {risk}',
                 fontsize=14, fontweight='bold', color=color, pad=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('last_result.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)


def run_calculator():
    """Main interactive loop."""

    while True:
        print("\n" + "─"*55)
        print("  ENTER PATIENT DETAILS")
        print("─"*55)

        # required inputs - the big three
        age   = get_input("  Age (years, 30-65)              : ", float, valid_range=(20, 80))
        ap_hi = get_input("  Systolic BP / ap_hi (mmHg, 70-250): ", float, valid_range=(70, 250))

        print("  Cholesterol level:")
        print("    1 = Normal")
        print("    2 = Above Normal")
        print("    3 = Well Above Normal")
        cholesterol = get_input("  Enter 1, 2, or 3               : ", int, valid_values=[1, 2, 3])

        # optional inputs with defaults
        print("\n  Optional details (press Enter to use defaults):")

        def get_optional(prompt, default, type_fn, valid_range=None):
            raw = input(f"  {prompt} [default={default}]: ").strip()
            if raw == '':
                return default
            try:
                val = type_fn(raw)
                if valid_range and not (valid_range[0] <= val <= valid_range[1]):
                    return default
                return val
            except ValueError:
                return default

        gender = get_optional("Gender (1=Female, 2=Male)", 1, int)
        ap_lo  = get_optional("Diastolic BP / ap_lo (mmHg)", 80, float, (40, 150))
        height = get_optional("Height (cm)", 165, float, (100, 220))
        weight = get_optional("Weight (kg)", 75, float, (30, 200))

        # make sure systolic > diastolic
        if ap_hi <= ap_lo:
            print(f"\n  Warning: systolic ({ap_hi}) must be > diastolic ({ap_lo})")
            print(f"  Setting diastolic to {ap_hi - 10}")
            ap_lo = ap_hi - 10

        # calculate derived features
        bmi            = round(weight / (height / 100) ** 2, 1)
        pulse_pressure = ap_hi - ap_lo

        # build patient dataframe
        patient = pd.DataFrame([{
            'age': age, 'gender': gender, 'height': height, 'weight': weight,
            'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
            'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 1,
            'bmi': bmi, 'pulse_pressure': pulse_pressure
        }])[FEATURE_COLS]

        patient_scaled = patient.copy()
        patient_scaled[SCALE_COLS] = scaler.transform(patient[SCALE_COLS])

        prob = rf.predict_proba(patient_scaled)[0][1]

        if prob < GREY_ZONE_LOW:
            risk   = 'LOW RISK'
            action = 'No cardiology referral needed. Continue routine monitoring.'
            color  = COLOR_LOW
            symbol = 'GREEN'
        elif prob > GREY_ZONE_HIGH:
            risk   = 'HIGH RISK'
            action = 'Refer to cardiologist. Immediate follow-up recommended.'
            color  = COLOR_HIGH
            symbol = 'RED'
        else:
            risk   = 'UNCERTAIN'
            action = 'Additional testing recommended before referral decision.'
            color  = COLOR_GREY
            symbol = 'YELLOW'

        chol_map   = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
        gender_map = {1: 'Female', 2: 'Male'}

        # print result
        print("\n" + "="*55)
        print("  RESULT")
        print("="*55)
        print(f"  Age          : {age:.0f} years")
        print(f"  Gender       : {gender_map.get(int(gender), '?')}")
        print(f"  BP           : {ap_hi:.0f}/{ap_lo:.0f} mmHg  (PP: {pulse_pressure:.0f})")
        print(f"  Cholesterol  : {chol_map.get(cholesterol, '?')}")
        print(f"  BMI          : {bmi}")
        print(f"{'─'*55}")
        print(f"  P(CVD)       :  {prob:.1%}")
        print(f"  RISK LEVEL   :  {risk}  [{symbol}]")
        print(f"  ACTION       :  {action}")
        print("="*55)

        # show visual gauge
        patient_label = f"Age {age:.0f}y | BP {ap_hi:.0f}/{ap_lo:.0f} | Chol {chol_map[cholesterol]}"
        show_gauge(prob, risk, color, patient_label)

        # ask to continue
        again = input("\n  Run another patient? (y/n): ").strip().lower()
        if again != 'y':
            print("\n  Thank you. Calculator closed.")
            plt.close('all')
            break


# run the calculator
if __name__ == '__main__':
    run_calculator()