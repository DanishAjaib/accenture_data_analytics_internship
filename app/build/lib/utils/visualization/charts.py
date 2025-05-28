import altair as alt
import pandas as pd
import streamlit as st
from typing import List

def create_histogram(df, column: str, title: str = None, 
                    width: int = 300, height: int = 300) -> alt.Chart:
    """Create a basic histogram chart"""
    title = title if title else f'Histogram of {column}'
    return alt.Chart(df).mark_bar().encode(
        alt.X(f"{column}:Q", bin=True),
        y='count()'
    ).properties(
        title=title,
        width=width,
        height=height
    )

def create_histogram_grid(features: List[str], df: pd.DataFrame, 
                         cols_per_row: int = 3) -> None:
    """Display a grid of histograms"""
    rows = len(features) // cols_per_row + (len(features) % cols_per_row > 0)
    
    for i in range(rows):
        columns = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < len(features):
                with columns[j]:
                    st.altair_chart(
                        create_histogram(df, features[idx]), 
                        use_container_width=True
                    )

def plot_correlation_matrix(data: pd.DataFrame) -> None:
    """Display correlation matrix"""
    try:
        st.dataframe(data.corr())
    except Exception as e:
        st.error(f'Error calculating correlations: {str(e)}')

def plot_confusion_matrix(confusion_matrix: pd.DataFrame) -> None:
    """Display styled confusion matrix"""
    st.dataframe(
        confusion_matrix.style.background_gradient(cmap='Blues'),
        use_container_width=True
    )

def show_donut_chart(probabilities: List[float], 
                    labels: List[str] = None,
                    colors: List[str] = None) -> alt.Chart:
    """Create a donut chart for classification probabilities"""
    labels = labels or ['Not Complete', 'Complete']
    colors = colors or ['#FF204E', '#BED754']
    
    data = pd.DataFrame({
        'Category': labels,
        'Probability': probabilities,
        'Color': colors
    })
    
    return alt.Chart(data).mark_arc(innerRadius=50).encode(
        theta='Probability',
        color=alt.Color('Color', scale=None),
        tooltip=['Category', 'Probability']
    ).properties(
        width=400,
        height=400
    )