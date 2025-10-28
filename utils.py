import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Tuple, Dict
def validate_data(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Check for nulls, correct datatypes, and minimum sample size.
    Returns tuple: (is_valid, warnings)
    """
    warnings = []
    if df.isnull().values.any():
        warnings.append("Null values found.")
    if len(df) < 50:
        warnings.append("Dataset recommended to have at least 50 samples.")
    for col in df.columns:
        if df[col].dtype == "object":
            if len(df[col].unique()) > 100:
                warnings.append(f"Column '{col}' has unusually high cardinality.")
    return (len(warnings) == 0, warnings)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example preprocessing: fill nulls, encode categoricals, scale numerics.
    """
    df = df.copy()
    # Fill missing numerics with median, categoricals with mode
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    # Encode categoricals
    for col in df.select_dtypes('object').columns:
        df[col] = pd.factorize(df[col])[0]
    # Optionally scale (min-max)
    for col in df.select_dtypes([np.float64, np.int64]).columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-6)
    return df

def membership_inference_test(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    """
    Calculates percentage overlap between real and synthetic records (rows).
    High overlap may indicate privacy risk.
    """
    overlap = pd.merge(real_df, synth_df)
    risk_score = len(overlap) / len(real_df)
    return risk_score  

def compare_distributions(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict:
    """
    Compare feature-wise distributions using KS-test (Kolmogorov-Smirnov).
    Returns dict with per-column p-values.
    """
    from scipy.stats import ks_2samp
    results = {}
    for col in real_df.columns:
        stat, p = ks_2samp(real_df[col], synth_df[col])
        results[col] = {"ks_stat": stat, "p_value": p}
    return results

def plot_feature_distributions(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    """
    Overlay histograms for visual comparison.
    """
    cols = real_df.columns
    for col in cols:
        plt.figure(figsize=(6, 3))
        sns.histplot(real_df[col], color="blue", label="Real", kde=True, stat="density")
        sns.histplot(synth_df[col], color="red", label="Synthetic", kde=True, stat="density")
        plt.title(f"Feature '{col}' Distribution")
        plt.legend()
        plt.show()

def utility_score(real_df: pd.DataFrame, synth_df: pd.DataFrame, target: str) -> float:
    """
    Train classifier on synthetic, test on real (for utility).
    Returns accuracy.
    """
    from sklearn.linear_model import LogisticRegression
    X_synth, y_synth = synth_df.drop(columns=[target]), synth_df[target]
    X_real, y_real = real_df.drop(columns=[target]), real_df[target]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_synth, y_synth)
    y_pred = clf.predict(X_real)
    return accuracy_score(y_real, y_pred)
