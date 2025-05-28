import streamlit as st
import pandas as pd
import pandas as pd
import altair as alt
import streamlit as st
import pandas as pd
import altair as alt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

import modelbit as mb

import nltk
from nltk.corpus import stopwords

nltk.download()

#@st.cache_data(persist='disk')
def get_info_df(df):
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    lines = [line.split() for line in s.splitlines()[3:-2]]
    return pd.DataFrame(lines)

#@st.cache_data(persist='disk')
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

#@st.cache_data(persist='disk')
def create_histogram(df, column):
    chart = alt.Chart(df).mark_bar().encode(
        alt.X(f"{column}:Q", bin=True),
        y='count()'
    ).properties(
        title=f'Histogram of {column}',
        width=300,
        height=300
    )
    return chart

#@st.cache_resource
def create_histogram_grid(features_num, num_cols, df):
    rows = len(features_num) // num_cols + (len(features_num) % num_cols > 0)
    for i in range(rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < len(features_num):
                col_name = features_num[idx]
                with cols[j]:
                    st.altair_chart(create_histogram(df, col_name), use_container_width=True)

#@st.cache_resource
def print_outliers(num_cols, df):
    # Check for outliers and handle them
    for col in num_cols:
        outliers = detect_outliers(df, col)
        st.text(f"Outliers detected in {col}: {len(outliers) if not outliers.empty else 0}")
        # Capping outliers
        cap = df[col].quantile(0.95)
        floor = df[col].quantile(0.05)
        df[col] = np.where(df[col] > cap, cap, df[col])
        df[col] = np.where(df[col] < floor, floor, df[col])

#@st.cache_resource
def plot_correlation_matrix(data):
    import seaborn as sns
    try:
        correlation_matrix = data.corr()
        st.text(correlation_matrix)
    except:
        st.text('Error')

st.title('British Airways bookings analysis')
st.markdown('In this project we will analyse customer bookings data for Brith Airways , prepare it for training and finally train a model predict whether a customer completes a booking or not.')
df = pd.read_csv('../data/british_airways_bookings.csv', encoding="ISO-8859-1", header=0).reset_index(drop=True)

st.dataframe(df.head())

st.markdown('First we try to understand what the dataset looks like')
st.dataframe(get_info_df(df), width=920)


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


# Streamlit app
st.title("Data Exploration and Preparation")

# Select numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).nunique().loc[lambda x: x > 2].index.tolist()

# Check if 'num_passengers' is in the list of numerical columns
if 'num_passengers' not in numerical_columns:
    st.text('Error: num_passengers column not found in numerical columns')
else:
    st.text('num_passengers column found in numerical columns')

create_histogram_grid(features_num=numerical_columns, num_cols=3, df=df)

st.markdown(
    '''
From the histograms above we can see that the data is highly skewed. To fix the scaling we need to 
apply a transformation strategy. But before we apply a transformation, we need to check for outliers and 
remove them.
'''
)

continuous_features = df.select_dtypes(include=['int64', 'float64']).nunique().loc[lambda x: x > 2].index.tolist()
print_outliers(num_cols=continuous_features, df=df)


st.markdown(
    '''
A large number of outliers were detected so lets apply the Box-Cox transfrmation and feature scaling since 
it can work for right or left skewed dataset and requires positive values.
'''
)
df_transformed = df.copy()
continuous_features_transformed = df_transformed.select_dtypes(include=['int64', 'float64']).nunique().loc[lambda x: x > 2].index.tolist()
categorical_features = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']
for col in continuous_features_transformed:
    df_transformed[col] = np.log1p(df[col])

st.title("Feature Scaling")

from sklearn.preprocessing import MinMaxScaler
import joblib
scaler = joblib.load('models/british_airways/scaler.pkl')

df_transformed[continuous_features_transformed] = scaler.fit_transform(df_transformed[continuous_features_transformed])

create_histogram_grid(features_num=continuous_features_transformed, num_cols=3, df=df_transformed)

st.markdown(
    '''
    After applying a transformation and scaling the features our data looks more normally distributed which
    means most of the outliers are gone. Now we need to apply one-hot-encoding to convert the categorical features.
'''
)
st.title("Encoding categorical features")
df_encoded = pd.get_dummies(df_transformed, columns = categorical_features,)
st.dataframe(df_encoded.head())
all_features = df_encoded.columns.tolist()
st.markdown(
    f'''
   Our data is now clean but before we proceed with training a model we first need to 
   check if any of the features are correlated with each other.
'''
)

plot_correlation_matrix(data=df_transformed[continuous_features_transformed])

st.markdown(
    f'''
   There doesn't seem to be any strong correlations among features. Now we check correlation between categorical variables using the chi2 contigency from scipy.
'''
)

#@st.cache_data(persist='disk')
def predict_booking_complete(
    num_passengers,
    sales_channel,
    trip_type,
    purchase_lead,
    length_of_stay,
    flight_hour,
    flight_day,
    route,
    booking_origin,
    wants_extra_baggage,
    wants_preferred_seat,
    wants_in_flight_meals,
    flight_duration):
    
    # Load the model
    # model = joblib.load('models/british_airways_bookings_random_forest.pkl')
    model = mb.get_model("british_airways_bookings_random_forest")

    # Create a dataframe with the new data
    new_data = pd.DataFrame({
        'num_passengers': [num_passengers],
        'sales_channel': [sales_channel],
        'trip_type': [trip_type],
        'purchase_lead': [purchase_lead],
        'length_of_stay': [length_of_stay],
        'flight_hour': [flight_hour],
        'flight_day': [flight_day],
        'route': [route],
        'booking_origin': [booking_origin],
        'wants_extra_baggage': [wants_extra_baggage],
        'wants_preferred_seat': [wants_preferred_seat],
        'wants_in_flight_meals': [wants_in_flight_meals],
        'flight_duration': [flight_duration]
    })

    #all all_features to the new data


    st.write("New data before preprocessing:")
    st.dataframe(new_data)
    
    # Preprocess the data
    num_columns = ['num_passengers', 'purchase_lead', 'length_of_stay', 'flight_hour', 'flight_duration']
    cat_columns = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']


    
    # Log transform the numerical columns
    new_data[num_columns] = new_data[num_columns].apply(lambda x: np.log1p(x.clip(lower=0) + 1))
    
    st.text("After log transformation:")
    st.dataframe(new_data.head())
    
    new_data[num_columns] = scaler.transform(new_data[num_columns])
    
    st.text("After scaling:")
    st.dataframe(new_data.head())
    

    new_data = pd.get_dummies(new_data, columns=cat_columns)
    
    st.text("After one-hot encoding:")
    st.dataframe(new_data)
    
    #remove booking_complete from all_features
    all_features.remove('booking_complete')
    for col in all_features:
        if col not in new_data.columns and col != 'booking_complete':
            new_data[col] = False
    
    new_data = new_data[all_features]

    st.text("Prepared Data:")
    st.dataframe(new_data.head(10))
    # Make a prediction

    
    return prediction, probablity

def categorical_corr_matrix(df, cat_vars, alpha=0.05):
    """
    Compute the Chi-squared correlation matrix for categorical variables.

    Parameters:
    - df: pd.DataFrame containing the data
    - cat_vars: list of categorical variable names
    - alpha: significance level for the Chi-squared test

    Returns:
    - pd.DataFrame with p-values and significance indication
    """
    corr_matrix = pd.DataFrame(index=cat_vars, columns=cat_vars)

    for i in range(len(cat_vars)):
        for j in range(len(cat_vars)):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0  # Self-correlation
            else:
                # Create a contingency table
                contingency_table = pd.crosstab(df[cat_vars[i]], df[cat_vars[j]])
                # Perform the Chi-squared test
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                corr_matrix.iloc[i, j] = p # type: ignore

    # Indicate significance
    significance_matrix = corr_matrix.applymap(lambda x: 'associated' if x < alpha else 'not associated') # type: ignore

    return corr_matrix, significance_matrix


corr_matrix, significance_matrix = categorical_corr_matrix(df, categorical_features)

st.dataframe(corr_matrix, width=920)

st.header('Model Training')
st.markdown('''
Now its time to train a model. Based on the dataset we have, we can use Logistic Regression, Random Forest and Decision Tree with GridSearchCV
then assess the evaluation metrics to select the best model for our use case.
''')

X = df_encoded.drop(columns=['booking_complete'], axis=1)
y = df_encoded['booking_complete']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.subheader('Random Forest')
import joblib

#@st.cache_resource
def train_model():
    from sklearn.ensemble import RandomForestClassifier
    _model=RandomForestClassifier(n_estimators=100)
    _model.fit(X_train,y_train)
    joblib.dump(_model, '../models/british_airways/model.pkl')

#@st.cache_data(persist='disk')
def get_feature_importance(_model, X):
    #plot feature importance
    # feature_importance = _model.feature_importances_
    # feat_importances = pd.Series(feature_importance, index=X_train.columns)
    # feat_importances = feat_importances.nlargest(10).reset_index()
    # feat_importances.columns = ['Feature', 'Importance']
    # joblib.dump(feat_importances, 'models/british_airways/feat_importances.pkl')
    feat_importances = joblib.load('../models/british_airways/feat_importances.pkl')
    return feat_importances

#@st.cache_data(persist='disk')

def plot_feature_importance(feat_importances):
    chart = alt.Chart(feat_importances).mark_bar().encode(
    x='Importance',
    y=alt.Y('Feature', sort='-x')
    ).properties(
        title='Feature Importance'
    )




    # Display the chart using Streamlit
    st.altair_chart(chart, use_container_width=True)


#@st.cache_data(persist='disk')
def get_confusion_matrix(_model, X_test, y_test):
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, _model.predict(X_test))
    # joblib.dump(cm, 'models/british_airways/cm.pkl')
    cm = joblib.load('../models/british_airways/cm.pkl')
    return cm

#@st.cache_data(persist='disk')
def plot_confusion_matrix(confusion_matrix):
    cm_df = pd.DataFrame(confusion_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    st.dataframe(cm_df)

#@st.cache_data(persist='disk')
def get_classification_report(_model, X_test, y_test):
    # from sklearn.metrics import classification_report
    # report = classification_report(y_test, _model.predict(X_test), output_dict=True)
    # joblib.dump(report, 'models/british_airways/classification_report.pkl')
    report = joblib.load('models/british_airways/classification_report.pkl')
    return report

#@st.cache_data(persist='disk')
def plot_classification_report(report):
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)



st.header('Model Evaluation')
# train_model()
st.subheader('Feature Importance')
feature_importance = get_feature_importance()
# Create the Altair chart
plot_feature_importance(feature_importance)

confusion_matrix = get_confusion_matrix()
plot_confusion_matrix(confusion_matrix)

classification_report = get_classification_report()
plot_classification_report(classification_report)


st.header('Conclusion')

st.markdown(
    '''
    The model demonstrates a strong performance with an overall accuracy of 85.18%. The precision and recall for the negative class (0) are notably high at 86.68% and 97.61%, respectively, indicating that the model is highly effective at correctly identifying negative instances. However, the precision and recall for the positive class (1) are lower at 49.42% and 13.49%, respectively, suggesting that the model struggles to accurately predict positive instances.
The confusion matrix further illustrates this, with 12,478 true negatives and 299 true positives, but also 306 false negatives and 1,917 false positives. This imbalance highlights the challenge in predicting the positive class accurately.
The feature importance analysis reveals that 'purchase_lead', 'flight_hour', and 'length of stay' are the top three most influential features in the model. These features significantly contribute to the model's predictions and should be considered key factors in understanding booking behaviors.
Overall, while the model performs well in predicting the negative class, there is room for improvement in predicting the positive class. Future efforts could focus on addressing this imbalance to enhance the model's predictive capabilities for both classes
    '''
)

st.title('Test the model')

# Define the options for dropdowns
sales_channel_options = df['sales_channel'].unique().tolist()
trip_type_options = df['trip_type'].unique().tolist()
flight_day_options = df['flight_day'].unique().tolist()
route_options = df['route'].unique().tolist()
booking_origin_options = df['booking_origin'].unique().tolist()


num_passengers = st.number_input(
    'Number of Passengers', 
    min_value=int(df['num_passengers'].min()), 
    max_value=int(df['num_passengers'].max()), 
    value=1
)
sales_channel = st.selectbox('Sales Channel', sales_channel_options)
trip_type = st.selectbox('Trip Type', trip_type_options)
purchase_lead = st.number_input(
    'Purchase Lead (days)', 
    min_value=int(df['purchase_lead'].min()), 
    max_value=int(df['purchase_lead'].max()), 
    value=5, 
    step=1
)
length_of_stay = st.number_input(
    'Length of Stay (days)', 
    min_value=int(df['length_of_stay'].min()), 
    max_value=int(df['length_of_stay'].max()), 
    value=3,
)
flight_hour = st.number_input('Flight Hour', min_value=0, max_value=23, value=0)
flight_day = st.selectbox('Flight Day', flight_day_options)
route = st.selectbox('Route', route_options)
booking_origin = st.selectbox('Booking Origin', booking_origin_options)
wants_extra_baggage = st.checkbox('Wants Extra Baggage')
wants_preferred_seat = st.checkbox('Wants Preferred Seat')
wants_in_flight_meals = st.checkbox('Wants In-Flight Meals')
flight_duration = st.number_input(
    'Flight Duration (hours)', 
    min_value=float(df['flight_duration'].min()),
    max_value=float(df['flight_duration'].max()),
    value=5.0
)
st.subheader('Booking Completion Prediction')
# display the input values
st.subheader('Input Values')
st.write(
    {
        'num_passengers': num_passengers,
        'sales_channel': sales_channel,
        'trip_type': trip_type,
        'purchase_lead': purchase_lead,
        'length_of_stay': length_of_stay,
        'flight_hour': flight_hour,
        'flight_day': flight_day,
        'route': route,
        'booking_origin': booking_origin,
        'wants_extra_baggage': wants_extra_baggage,
        'wants_preferred_seat': wants_preferred_seat,
        'wants_in_flight_meals': wants_in_flight_meals,
        'flight_duration': flight_duration,
    }
)

prediction, probablity = predict_booking_complete(num_passengers, sales_channel, trip_type, purchase_lead, length_of_stay, flight_hour, flight_day, route, booking_origin, wants_extra_baggage, wants_preferred_seat, wants_in_flight_meals, flight_duration)
# probablity = probablity.flatten()
st.header('Prediction')

#@st.cache_resource
def show_donut_chart(probablity):
    color_red, color_green = '#FF204E', '#BED754'

    # Create a DataFrame with the prediction probabilities
    df = pd.DataFrame({
        'Class': ['Not Complete', 'Complete'],
        'Probability': [probablity[0], probablity[1]],
        'Color': [color_red, color_green]
    })

    # Create a donut chart using Altair
    chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field='Probability', type='quantitative'),
        color=alt.Color(field='Color', type='nominal', scale=None),
        tooltip=['Class', 'Probability']
    ).properties(
        width=400,
        height=400
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

# show_donut_chart(probablity)
# st.text(evaluate_model(model=log_reg))