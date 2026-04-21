# Phase 0: Setup & Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, learning_curve
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

RANDOM_STATE = 2871
np.random.seed(RANDOM_STATE)

COLOR_NO_CVD   = '#2E86AB'
COLOR_CVD      = '#C73E1D'
COLOR_ACCENT1  = '#A23B72'
COLOR_ACCENT2  = '#F18F01'
COLOR_ACCENT3  = '#2BA84A'

COLORS_TARGET  = [COLOR_NO_CVD, COLOR_CVD]
LABELS_TARGET  = ['No CVD', 'Has CVD']

FN_COST = 150_000
FP_COST = 300

plt.rcParams.update({
    'figure.dpi':        150,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'font.family':       'sans-serif',
})

CV_FOLDS = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

print("Setup complete.")
print(f"  False Negative cost : ${FN_COST:,}")
print(f"  False Positive cost : ${FP_COST:,}")
print(f"  Cost ratio          : {FN_COST // FP_COST}:1")
