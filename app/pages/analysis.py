import streamlit as st
import pandas as pd
from app.utils.analytics.wrangeling import merge_proportion_dfs
from app.utils.analytics.eda import get_info_df, remove_outliers , get_proportions_top_only
from app.utils.visualization.charts import create_boxplot, create_histogram_grid, create_stacked_bar_chart, plot_correlation_matrix
from scipy import stats


@st.cache_data(persist='disk')
def load_data():
    return pd.read_csv('../data/raw/british_airways_bookings.csv', encoding="ISO-8859-1", header=0).reset_index(drop=True)

df = load_data()

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
binary_columns = df[df.columns[df.nunique() == 2]].select_dtypes(include=['int64']).columns.tolist()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

st.title("Data Exploration and Analysis")
st.markdown('In this project we will analyse customer bookings data for Brith Airways , prepare it for training and finally train a model predict whether a customer completes a booking or not.')

st.dataframe(df.head())
st.markdown('First we try to understand what the dataset looks like')
st.dataframe(get_info_df(df), width=920)

st.markdown(
    '''
From the histograms above we can see that the data is highly skewed. To fix the scaling we need to 
apply a transformation strategy. But before we apply a transformation, we need to check for outliers and 
remove them.
'''
)

st.markdown(
    '''
We can see that the dataset has a total of `13` columns, 50K rows , `5 integer`, `1 float` and 
`5 categorical` features. Next, lets see the data summary for further exploration.
'''
)

st.markdown(
    '''
The table above gives us an overview of the data. We can see that all columns have 50K Non Null values which means
there is no missing data. The other metrics give us useful insights about each feature, 
e.g total number of `non-null` rows, `average value`, `standard deviation`, `minimum value`, `maximum value`, 
and the `percentiles` of the distribution. Next, we need to check how the data in each variable is distributed to see if any variables are skewed or not.
'''
)


st.markdown('#### Categorical features:')
st.markdown(
    '''
    The categorical features are `sales_channel`, `trip_type`, `flight_day`, `route`, and `booking_origin`.
    We can see that the `route` and `booking_origin` features have a large number of unique values.
''')

st.dataframe(df[categorical_columns].describe(), width=920)
st.markdown(
    '''
     Lets see the distribution of each categorical feature. We will use a donut chart to visualize the distribution of each feature.
''')

proportions = get_proportions_top_only(categorical_columns, df)

# num_columns_donut = 2
# for i in range(0, len(proportions), num_columns_donut):
#    cols = st.columns(num_columns_donut)
#    for j, col in enumerate(cols):
#        if i + j < len(proportions):
#            with col:
#                st.altair_chart(create_donut_chart(proportions[i + j]))

combined_proportions = merge_proportion_dfs(proportions)

st.altair_chart(create_stacked_bar_chart(combined_proportions))
st.markdown('#### Numerical Features:')


st.dataframe(df[numerical_columns].describe(), width=920)
st.markdown(
    '''
    Looking at the descriptive statistics of the numerical features, we can see that we have a mix of features with different scales.
    We have discrete features , continuous features and binary features. Lets take a look at the distribution of each numerical feature.
''')

st.markdown('#### Handling Outliers:')

create_histogram_grid(features=['length_of_stay', 'purchase_lead', 'flight_hour', 'num_passengers', 'flight_duration'], cols_per_row=3, df=df)


## create altair boxplot
with st.spinner('Creating boxplot...'):
    st.altair_chart(create_boxplot(df=df[['length_of_stay', 'purchase_lead', 'flight_hour', 'num_passengers', 'flight_duration']]), use_container_width=True)


st.markdown(
    '''
        We can see that the `light_of_stay` and `purchase_lead` features have a lot of outliers. We will handle the outliers by capping them to the 95th percentile.
        We will also apply the Box-Cox transformation to the numerical features to make them more normally distributed.
        The Box-Cox transformation is a family of power transformations that can be used to stabilize variance and make the data more normally distributed.
    ''')

# X = df.drop(columns=['booking_complete'])
# y = df['booking_complete']
# from sklearn.model_selection import train_test_split
# df, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df['length_of_stay'], _ = stats.boxcox(df['length_of_stay'] + 1)  # type: ignore # Adding 1 to ensure positivity
df['purchase_lead'], _ = stats.boxcox(df['purchase_lead'] + 1) # type: ignore
df['flight_hour'], _ = stats.boxcox(df['flight_hour'] + 1) # type: ignore
df['num_passengers'], _ = stats.boxcox(df['num_passengers'] + 1) # type: ignore
df['flight_duration'], _ = stats.boxcox(df['flight_duration'] + 1) # type: ignore

df_no_outliers = remove_outliers(df=df)
null_counts = df_no_outliers.isnull().sum()
non_zero_null_counts = null_counts[null_counts > 0]
non_zero_null_counts_df = pd.DataFrame(non_zero_null_counts).reset_index().rename(columns={'index': 'feature', 0: 'null_count'})
st.dataframe(non_zero_null_counts_df, width=920)
# st.dataframe(outliers, width=920)

with st.spinner('Creating boxplot...'):
    st.altair_chart(create_boxplot(df=df_no_outliers), use_container_width=True)

create_histogram_grid(features=['length_of_stay', 'purchase_lead', 'flight_hour', 'num_passengers', 'flight_duration'], cols_per_row=3, df=df_no_outliers)

st.markdown(
    '''
    The numerical features are `booking_complete`, `lead_time`, `num_passengers`, `num_segments`, `num_stops`, `total_price`, and `total_distance`.
    We can see that the `lead_time`, `num_passengers`, and `total_price` features have a large number of unique values.
    We can also see that the `booking_complete` feature is highly imbalanced. This means that we have a lot of `0`s and very few `1`s.
''')

st.markdown(
    '''
    The histograms above show us the distribution of each numerical feature. We can see that most of the features are right skewed.
    We can also see that the `booking_complete` feature is highly imbalanced. This means that we have a lot of `0`s and very few `1`s.
    Next, lets check for outliers in the numerical features. 
''')

## loader
with st.spinner('Computing correlation matrix...'):
    plot_correlation_matrix(df)

st.markdown(
    '''
A large number of outliers were detected so lets apply the Box-Cox transfrmation and feature scaling since 
it can work for right or left skewed dataset and requires positive values.
'''
)

st.title("Feature Scaling")