import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, f_oneway, chi2_contingency, shapiro
from sklearn.linear_model import LinearRegression, LogisticRegression



# ✅ 10. P-value interpretation
def interpret_p_value(p_val, alpha=0.05):
    return "Reject null hypothesis" if p_val < alpha else "Fail to reject null hypothesis"
==========================================================================================
# ✅ 11. Histogram + KDE Plot
def plot_distribution(data, title="Distribution"):
    plt.figure(figsize=(6, 4))
    sns.histplot(data, kde=True, color='skyblue', bins=30)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
===============================================================
# ✅ 12. Q-Q Plot
def plot_qq(data, title="Q-Q Plot"):
    plt.figure(figsize=(6, 4))
    probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.show()
--------------------------------------------
from scipy.stats import shapiro, probplot
import seaborn as sns
import matplotlib.pyplot as plt

def check_normality(series, alpha=0.05):
    # Histogram + KDE
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(series, kde=True, bins=20, color='skyblue')
    plt.title("Histogram + KDE")

    # Q-Q Plot
    plt.subplot(1, 2, 2)
    probplot(series, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk Test
    stat, p_val = shapiro(series)
    print(f"Shapiro-Wilk Test p-value: {p_val:.4f}")
    if p_val > alpha:
        print("✅ Normally distributed")
    else:
        print("❌ Not normally distributed")
)

