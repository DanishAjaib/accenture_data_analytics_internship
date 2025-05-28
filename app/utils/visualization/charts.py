import altair as alt
import pandas as pd
import streamlit as st
from typing import List

def create_histogram(df, column: str, title: str = None,  # type: ignore
                    width: int = 300, height: int = 300) -> alt.Chart:
    """Create a basic histogram chart"""
    title = title if title else f'Histogram of {column}'
    return alt.Chart(df).mark_bar().encode(
        alt.X(f"{column}:Q", bin=alt.Bin(maxbins=30), ),
        y='count()'
    ).properties(
        title=title,
        width=width,
        height=height
    )

def create_boxplot_grid(features: List[str], df: pd.DataFrame) -> None:
    """Display a grid of boxplots"""
    rows = len(features) // 3 + (len(features) % 3 > 0)
    
    for i in range(rows):
        columns = st.columns(3)
        for j in range(3):
            idx = i * 3 + j
            if idx < len(features):
                with columns[j]:
                    st.altair_chart(
                        create_boxplot(df, features[idx]), # type: ignore
                        use_container_width=True
                    )

def create_scatterplot(df: pd.DataFrame, x: str, y: str) -> alt.Chart:
    """Create a scatterplot for two given columns"""
    return alt.Chart(df).mark_circle(size=60).encode(
            x=x,
            y=y,
            tooltip=[x, y],
            color=alt.condition(alt.datum.booking_complete == 1), # type: ignore
        ).interactive()

def create_scatterplot_grid(features: List[str], df: pd.DataFrame) -> None:
    """Display a grid of scatterplots"""
    rows = len(features) // 2 + (len(features) % 2 > 0)
    
    for i in range(rows):
        columns = st.columns(2)
        for j in range(2):
            idx = i * 2 + j
            if idx < len(features):
                with columns[j]:
                    st.altair_chart(
                        create_scatterplot(df, features[idx], features[idx+1]), 
                        use_container_width=True
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
                        create_histogram(df, features[idx]), use_container_width=True
                    )
@st.cache_data(persist='disk')
def create_donut_chart(df):
    import altair as alt

    if not all(col in df.columns for col in ['count', 'percentage']):
        raise ValueError("DataFrame must contain 'category', 'count', and 'percentage' columns")
    
    # Create the donut chart
    chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field='count', type='quantitative'),
        color=alt.Color(field=df.columns.tolist()[0], type='nominal'),
        tooltip=[alt.Tooltip(field=df.columns.tolist()[0], type='nominal',), 
                 alt.Tooltip(field='count', type='quantitative'), 
                 alt.Tooltip(field='percentage', type='quantitative', format='.2f')]
    )
    
    return chart

@st.cache_data(persist='disk')
def create_stacked_bar_chart(combined_df: pd.DataFrame) -> alt.Chart:
    chart = alt.Chart(combined_df).mark_bar().encode(
        x=alt.X('percentage:Q', stack='normalize', axis=alt.Axis(title='Proportion')),
        y=alt.Y('category:N', axis=alt.Axis(title='Category')),
        color=alt.Color('subcategory:N', legend=None, scale=alt.Scale(scheme='tableau20')),
        tooltip=['category', 'subcategory', 'count', 'percentage']
    ).properties(
        title='Stacked Bar Chart of Categorical Features',
        height=400,
        width=600
    )
    return chart


@st.cache_data(persist='disk')
def create_boxplot(df: pd.DataFrame) -> alt.Chart:
    df_long = df.melt(var_name='variable', value_name='value')

    """Create a boxplot for a given column"""
    return alt.Chart(df_long).mark_boxplot().encode(
        color=alt.Color('variable:N', legend=None),
        x=alt.X('value:Q', title='Value'),
        y=alt.Y('variable:N', title='Variable')
        )

def plot_correlation_matrix(data: pd.DataFrame) -> None:
    """Display correlation matrix"""
    numerical_columns = data.select_dtypes(include=['int64', 'float64'])
    try:
        corr_matrix = numerical_columns.loc[:, numerical_columns.nunique() > 2].corr().reset_index().melt('index')
        corr_matrix.columns = ['Feature1', 'Feature2', 'Correlation']
        heatmap = alt.Chart(corr_matrix).mark_rect().encode(
            x='Feature1:O',
            y='Feature2:O',
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['Feature1', 'Feature2', 'Correlation']
        ).properties(
            width=600,
            height=600,
        )

        text = heatmap.mark_text(baseline='middle').encode(
            text=alt.Text('Correlation:Q', format='.2f'),
            color=alt.condition(
                alt.datum.Correlation > 0.5,
                alt.value('black'),  # Text color for high correlation
                alt.value('white')   # Text color for low correlation
            )
        )

        heatmap = heatmap + text
        st.altair_chart(heatmap, use_container_width=True)

    except Exception as e:
        st.error(f'Error calculating correlations: {str(e)}')

def plot_confusion_matrix(confusion_matrix: pd.DataFrame) -> None:
    """Display styled confusion matrix"""
    st.dataframe(
        confusion_matrix.style.background_gradient(cmap='Blues'),
        use_container_width=True
    )

def show_donut_chart(probabilities: List[float], 
                    labels: List[str] = None, # type: ignore
                    colors: List[str] = None) -> alt.Chart: # type: ignore
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