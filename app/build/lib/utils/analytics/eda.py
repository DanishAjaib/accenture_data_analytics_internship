import io
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def get_info_df(df):
    """Get DataFrame information in a structured format"""
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    lines = [line.split() for line in s.splitlines()[3:-2]]
    return pd.DataFrame(lines)

def detect_outliers(df, column):
    """Identify outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def handle_outliers(df, column):
    """Cap outliers at 5th and 95th percentiles"""
    cap = df[column].quantile(0.95)
    floor = df[column].quantile(0.05)
    df[column] = np.where(df[column] > cap, cap, df[column])
    df[column] = np.where(df[column] < floor, floor, df[column])
    return df

def categorical_corr_matrix(df, cat_vars, alpha=0.05):
    """Compute Chi-squared correlation for categorical variables"""
    corr_matrix = pd.DataFrame(index=cat_vars, columns=cat_vars)
    
    for i in range(len(cat_vars)):
        for j in range(len(cat_vars)):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
            else:
                contingency_table = pd.crosstab(df[cat_vars[i]], df[cat_vars[j]])
                chi2, p, _, _ = chi2_contingency(contingency_table)
                corr_matrix.iloc[i, j] = p
                
    significance_matrix = corr_matrix.applymap(
        lambda x: 'associated' if x < alpha else 'not associated'
    )
    return corr_matrix, significance_matrix