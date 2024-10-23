# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the saved Random Forest model
import os
print(os.getcwd())
model = joblib.load(os.getcwd()+'/random_forest_model.pkl')

# Define a function to take user input
def user_input_features():
    # Add input fields based on the features of your dataset
    price = st.number_input('Price', min_value=0.0, max_value=100.0, step=0.1)
    total_reviews = st.number_input('Total Reviews', min_value=0, max_value=100000, step=100)
    developer_count = st.number_input('Developer Count', min_value=1, max_value=100, step=1)
    achievements = st.number_input('Achievements', min_value=0, max_value=100, step=1)
    avg_playtime = st.number_input('Average Playtime', min_value=0, max_value=10000, step=10)

    # Create a dataframe with the input
    data = {
        'Price': price,
        'TotalReviews': total_reviews,
        'DeveloperCount': developer_count,
        'Achievements': achievements,
        'avg_playtime': avg_playtime
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Streamlit app title
st.title('Game Budget Category Prediction')

# Get user input
input_df = user_input_features()

# Display the input features for review
st.write("User Input:")
st.write(input_df)

# Predict using the loaded model
prediction = model.predict(input_df)

# Show the prediction result
st.subheader('Prediction')
st.write(f'Predicted Total Points: {prediction[0]:.2f}')
