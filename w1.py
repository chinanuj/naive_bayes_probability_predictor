import pandas as pd
import streamlit as st

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
   background-image: url(https://imgs.search.brave.com/LrI0X5JvzZPS3WTIqghqnySKvSfVtIO3nF3T-QTO0jQ/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9pbWcu/ZnJlZXBpay5jb20v/cHJlbWl1bS1waG90/by9wYWlyLXJlZC1k/aWNlLWJsYWNrLWJh/Y2tncm91bmRjbG9z/ZS11cC12aWV3Xzc4/NjIxLTU4MzcuanBn/P3NpemU9NjI2JmV4/dD1qcGc);
   background-size: cover;
}

[data-testid="stHeader"]{
background-color=rgba(0,0,0,0);
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
# Function to calculate prior probabilities
def calculate_prior_probabilities(df):
    count_yes = (df['Play'] == 'yes').sum()
    count_no = (df['Play'] == 'no').sum()
    total_samples = len(df)
    p_yes = count_yes / total_samples
    p_no = count_no / total_samples
    return p_yes, p_no

# Function to calculate likelihood probabilities without Laplace Smoothing
def calculate_likelihood_probabilities(df):
    likelihood_probabilities = {}
    for column in df.columns[:-1]:
        for feature_value in df[column].unique():
            for class_value in df['Play'].unique():
                count_class_and_feature = ((df[column] == feature_value) & (df['Play'] == class_value)).sum()
                count_class = (df['Play'] == class_value).sum()
                key = f"probability of {column} = {feature_value} and Play = {class_value}"
                likelihood_probabilities[key] = count_class_and_feature / count_class
    return likelihood_probabilities

# Function to calculate posterior probabilities
def calculate_posterior_probabilities(df, input_combination, prior_probabilities, likelihood_probabilities):
    posterior_probabilities = {}
    for class_value in df['Play'].unique():
        posterior_probabilities[class_value] = prior_probabilities[class_value]
        for column, feature_value in input_combination.items():
            key = f"probability of {column} = {feature_value} and Play = {class_value}"
            posterior_probabilities[class_value] *= likelihood_probabilities[key]
    return posterior_probabilities

# Function to make prediction based on posterior probabilities
def predict(posterior_probabilities):
    if posterior_probabilities['yes'] > posterior_probabilities['no']:
        return "Yes, you can play outside!"
    else:
        return "No, it's not suitable to play outside."

# Load data
df = pd.read_csv('naive_bayes.csv')

# Calculate prior probabilities
p_yes, p_no = calculate_prior_probabilities(df)

# Calculate likelihood probabilities
likelihood_probabilities = calculate_likelihood_probabilities(df)

# Streamlit app
st.markdown('<style>@keyframes color-change {0% {color: green;} 50% {color: red;} 100% {color: green;}}</style>', unsafe_allow_html=True)
st.markdown('<h1 style="text-align: center; text-decoration: underline;">Naive Bayes Probability Predictor on Golf dataset to play outside or not(without Laplace Smoothing)</h1>', unsafe_allow_html=True)
st.header('Select Features')

# Checkbox for Outlook
outlook_option = st.selectbox('Outlook', ['', *df['Outlook'].unique()], index=0)

# Checkbox for Temp
temp_option = st.selectbox('Temp', ['', *df['Temp'].unique()], index=0)

# Checkbox for Humidity
humidity_option = st.selectbox('Humidity', ['', *df['Humidity'].unique()], index=0)

# Checkbox for Windy
windy_option = st.selectbox('Windy', ['', *df['Windy'].unique()], index=0)

# Submit Button
if st.button('Predict', key='predict_button'):
    if all([outlook_option, temp_option, humidity_option, windy_option]):
        # Create input combination dictionary
        input_combination = {
            'Outlook': outlook_option,
            'Temp': temp_option,
            'Humidity': humidity_option,
            'Windy': windy_option
        }
        # Calculate posterior probabilities
        posterior_probabilities = calculate_posterior_probabilities(df, input_combination, {'yes': p_yes, 'no': p_no}, likelihood_probabilities)
        # Make prediction
        prediction = predict(posterior_probabilities)
        # Display prediction
        st.write(f"Prediction: {prediction}")
        # Display posterior probabilities
        st.write("Posterior Probabilities:")
        st.write(f"- Yes: {posterior_probabilities['yes']}")
        st.write(f"- No: {posterior_probabilities['no']}")
        # Display likelihood probabilities
        st.write("Likelihood Probabilities:")
        for key, value in likelihood_probabilities.items():
            st.write(f"- {key}: {value}")
        # Display prior probabilities
        st.write("Prior Probabilities:")
        st.write(f"- Yes: {p_yes}")
        st.write(f"- No: {p_no}")
    else:
        st.error("Please select one option from each feature group.")

# Clear Button
if st.button('Clear Output', key='clear_button'):
    st.text('')

