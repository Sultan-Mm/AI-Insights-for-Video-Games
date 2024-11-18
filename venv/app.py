
import streamlit as st
import pandas as pd
import numpy as np
import os

# Load dataset
@st.cache_data
def load_data():
    file_path = os.path.abspath('merged_df_games_with_ranking_df_games_enriched.csv')
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Filter columns for binary features, including the new ones
binary_features = [
    'Adventure', 'Casual', 'Indie', 'RPG', 'Free To Play', 'Action',
    'Strategy', 'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
    'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op', 'Cross-Platform Multiplayer',
    'Family Sharing', 'HDR available', 'In-App Purchases', 'Multi-player', 'VR Support', 'age_ranking'
]

game_columns = ['QueryName', 'Popularity_score', 'TotalReviews', 'Sales', 'Pred_Owners']

# Extract OHE genre columns
genre_columns = [col for col in df.columns if 'Genre' in col]

# Create sidebar filters for binary features
st.sidebar.header('Filter by Genre')
selected_binary = st.sidebar.multiselect('Select a genre', binary_features)

# Filter by budget
st.sidebar.header('Filter by Budget')
budget_options = df[['Budget_AAA', 'Budget_AA', 'Budget_Indie']].columns.tolist()
selected_budget = st.sidebar.selectbox('Select a budget type', budget_options)

# Define thresholds for popularity score to categorize as High, Medium, Low
max_popularity = df['Popularity_score'].max()
min_popularity = df['Popularity_score'].min()
threshold_high = min_popularity + (max_popularity - min_popularity) * 0.67
threshold_medium = min_popularity + (max_popularity - min_popularity) * 0.33

# Apply binary filters
filtered_df = df
if selected_binary:
    for feature in selected_binary:
        filtered_df = filtered_df[filtered_df[feature] == 1]

# Apply budget filter
if selected_budget:
    filtered_df = filtered_df[filtered_df[selected_budget] == 1]

# Filter game names by filtered DataFrame
filtered_df_sorted = filtered_df.sort_values(by='Popularity_score', ascending=False)
selected_game_name = st.sidebar.selectbox('Select a game', filtered_df_sorted['QueryName'])

# Display the filtered game list without the index and AppID
st.header('Filtered Games')
st.dataframe(filtered_df[['QueryName', 'Popularity_score', 'TotalReviews', 'Sales', 'Pred_Owners']])

# Display information for the selected game
st.header(f"Details of '{selected_game_name}'")
selected_game = df[df['QueryName'] == selected_game_name]

# Display game information
# Determine popularity level
popularity_score = selected_game['Popularity_score'].values[0]
if popularity_score >= threshold_high:
    popularity_label = 'High'
elif popularity_score >= threshold_medium:
    popularity_label = 'Medium'
else:
    popularity_label = 'Low'

if popularity_score == max_popularity:
    st.write(f"**This game is extremely popular and deserves a remake!**")
elif popularity_score == max_popularity:
    st.write(f"**This game has medium popularity!**")
else:
    st.write(f"**This game has low popularity and no need for reamake!**")

st.write(f"**Popularity Level:** {popularity_label} ({np.round(popularity_score)})")

st.write(f"**Release Date:** {selected_game['Release date'].values[0]}")
st.write(f"**Predicted Owners:** {np.round(selected_game['Pred_Owners'].values[0])}")
st.write(f"**Total Reviews:** {np.round(selected_game['TotalReviews'].values[0])}")
st.write(f"**Average Playtime (in Hours):** {np.round(selected_game['avg_playtime'].values[0])}")
st.write(f"**About the game:** {selected_game['about_the_game'].values[0]}")