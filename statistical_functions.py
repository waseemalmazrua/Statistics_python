import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, f_oneway, chi2_contingency, shapiro
from sklearn.linear_model import LinearRegression, LogisticRegression

# ✅ 1. Independent T-Test
def t_test_independent(group1, group2):
    """
    يقارن بين متوسط مجموعتين مستقلتين.
    مثال الاستخدام:
    group1 = df[df['Gender'] == 'Male']['Score']
    group2 = df[df['Gender'] == 'Female']['Score']
    t_stat, p_val = t_test_independent(group1, group2)
    """
    return ttest_ind(group1, group2, nan_policy='omit')

# ✅ 2. One Sample T-Test
def t_test_one_sample(data, popmean):
    """
    يقارن بين متوسط العينة ومتوسط معروف.
    مثال:
    t_stat, p_val = t_test_one_sample(df['Satisfaction'], 3.0)
    """
    return ttest_1samp(data, popmean)

# ✅ 3. Paired T-Test
def t_test_paired(before, after):
    """
    يقارن بين متوسط قبل وبعد لنفس المجموعة.
    مثال:
    t_stat, p_val = t_test_paired(df['BP_before'], df['BP_after'])
    """
    return ttest_rel(before, after)

# ✅ 4. ANOVA Test
def run_anova(*groups):
    """
    يستخدم لمقارنة أكثر من مجموعتين.
    مثال:
    run_anova(group1, group2, group3)
    """
    return f_oneway(*groups)

# ✅ 5. Chi-Square Test
def chi_square_test(table):
    """
    اختبار العلاقة بين متغيرين تصنيفيين.
    مثال:
    table = pd.crosstab(df['Gender'], df['Readmitted'])
    chi2, p, _, _ = chi_square_test(table)
    """
    return chi2_contingency(table)

# ✅ 6. Correlation Matrix
def correlation_matrix(df):
    """
    يحسب الارتباط بين جميع المتغيرات الرقمية.
    مثال:
    corr = correlation_matrix(df)
    """
    return df.corr()

# ✅ 7. Shapiro-Wilk Normality Test
def normality_test(data):
    """
    يتحقق إذا كانت البيانات تتبع التوزيع الطبيعي.
    مثال:
    stat, p_val = normality_test(df['Score'])
    """
    return shapiro(data)

# ✅ 8. Linear Regression
def run_linear_regression(X, y):
    """
    انحدار خطي لتوقع قيمة رقمية.
    مثال:
    model = run_linear_regression(df[['Age', 'Experience']], df['Salary'])
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

# ✅ 9. Logistic Regression
def run_logistic_regression(X, y):
    """
    انحدار لوجستي لتصنيف ثنائي.
    مثال:
    model = run_logistic_regression(df[['Age']], df['Readmitted'])
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

# ✅ 10. P-value interpretation
def interpret_p_value(p_val, alpha=0.05):
    """
    يفسر القيمة الاحتمالية.
    """
    return "Reject null hypothesis" if p_val < alpha else "Fail to reject null hypothesis"






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, f_oneway, chi2_contingency, shapiro, norm, probplot
from sklearn.linear_model import LinearRegression, LogisticRegression

# ✅ 1. Independent T-Test
def t_test_independent(group1, group2):
    return ttest_ind(group1, group2, nan_policy='omit')

# ✅ 2. One Sample T-Test
def t_test_one_sample(data, popmean):
    return ttest_1samp(data, popmean)

# ✅ 3. Paired T-Test
def t_test_paired(before, after):
    return ttest_rel(before, after)

# ✅ 4. ANOVA Test
def run_anova(*groups):
    return f_oneway(*groups)

# ✅ 5. Chi-Square Test
def chi_square_test(table):
    return chi2_contingency(table)

# ✅ 6. Correlation Matrix
def correlation_matrix(df):
    return df.corr()

# ✅ 7. Shapiro-Wilk Normality Test
def normality_test(data):
    return shapiro(data)

# ✅ 8. Linear Regression
def run_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# ✅ 9. Logistic Regression
def run_logistic_regression(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

# ✅ 10. P-value interpretation
def interpret_p_value(p_val, alpha=0.05):
    return "Reject null hypothesis" if p_val < alpha else "Fail to reject null hypothesis"

# ✅ 11. Histogram + KDE Plot
def plot_distribution(data, title="Distribution"):
    plt.figure(figsize=(6, 4))
    sns.histplot(data, kde=True, color='skyblue', bins=30)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# ✅ 12. Q-Q Plot
def plot_qq(data, title="Q-Q Plot"):
    plt.figure(figsize=(6, 4))
    probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ✅ 13. Combined Normality Check (Histogram + QQ + Shapiro)
def check_normality(data, alpha=0.05):
    plot_distribution(data, "Histogram + KDE")
    plot_qq(data, "Q-Q Plot")
    stat, p_val = normality_test(data)
    print(f"Shapiro-Wilk Test p-value: {p_val:.4f}")
    print("✅ Normally distributed" if p_val > alpha else "❌ Not normally distributed")

