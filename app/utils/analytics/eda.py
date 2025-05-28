import io
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data(persist='disk')
def get_info_df(df):
    """Get DataFrame information in a structured format"""
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    lines = [line.split() for line in s.splitlines()[3:-2]]
    return pd.DataFrame(lines)

@st.cache_data(persist='disk')
def get_proportions_top_only(categorical_columns, df):
    """Get proportions of categorical variables"""
    proportions = []
    for column in categorical_columns:
        column_value_counts = df[column].value_counts().reset_index()
        column_value_counts.columns = [column, 'count']
        column_value_counts['percentage'] = ((column_value_counts['count'] / column_value_counts['count'].sum()) * 100).round(2)
        
        if column_value_counts.shape[0] > 10:
            column_value_counts = column_value_counts.head(10)
        
        column_value_counts = column_value_counts.sort_values(by='percentage', ascending=False).reset_index(drop=True)
        proportions.append(column_value_counts)
    
    return proportions

@st.cache_data(persist='disk')
def get_proportions(categorical_columns, df):
    """Get proportions of categorical variables"""
    proportions = []
    for column in categorical_columns:
        column_value_counts = df[column].value_counts().reset_index()
        column_value_counts.columns = ['category', 'count']
        column_value_counts['percentage'] = ((column_value_counts['count'] / column_value_counts['count'].sum()) * 100).round(2)
        if(column_value_counts.shape[0] < 1000):
            column_value_counts = column_value_counts.sort_values(by='percentage', ascending=False).reset_index(drop=True)
            proportions.append(column_value_counts)
    return proportions


def remove_outliers(df):
    data = df.drop(['booking_complete'], axis=1).copy()
    cols = data.select_dtypes(include=['int64', 'float64']).columns
    df_sub = data.loc[:, cols]
    iqr = df_sub.quantile(0.75, numeric_only=False) - df_sub.quantile(0.25, numeric_only=False)
    lim = np.abs((df_sub - df_sub.median()) / iqr) < 2.22
    data.loc[:, cols] = df_sub.where(lim, np.nan)
    return data
