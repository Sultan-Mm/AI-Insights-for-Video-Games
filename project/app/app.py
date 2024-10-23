# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import joblib
# # # import os

# # # # Load saved models and data files
# # # model_path = os.path.abspath('project/app/gradient_boosting_model.pkl')
# # # file_path_tree = os.path.abspath('data/GamesFinish_woScaling.csv')
# # # file_path_games = os.path.abspath('data/games_c5.csv')

# # # # Title for the Streamlit app
# # # st.title('Game Popularity Prediction App')

# # # # Load the GradientBoosting model
# # # st.write("Loading the GradientBoostingRegressor model...")
# # # gb_reg = joblib.load(model_path)

# # # # Load the datasets
# # # df_tree = pd.read_csv(file_path_tree)
# # # df_games = pd.read_csv(file_path_games)

# # # # Filter rows where 'YearDifference' >= 10
# # # df_tree = df_tree[df_tree['YearDifference'] >= 10]

# # # # Prepare Features and Target for the prediction
# # # X_tree = df_tree.drop(columns='SteamSpyOwners')
# # # y_tree = df_tree['SteamSpyOwners']

# # # # GradientBoosting Regressor Prediction
# # # st.write("Predicting SteamSpy Owners using GradientBoosting Regressor...")
# # # X_tree['Pred_Owners'] = gb_reg.predict(X_tree)

# # # # Merge predictions back with the main dataset
# # # df_games_enriched = pd.concat([df_games, X_tree[['Pred_Owners']]], axis=1)

# # # # Calculate the Popularity Score
# # # df_games_enriched['Popularity_score'] = (
# # #     0.3 * df_games_enriched['TotalReviews'] +
# # #     0.3 * df_games_enriched['RecommendationCount'] +
# # #     0.4 * df_games_enriched['Pred_Owners']
# # # )

# # # # standardize the popularity score

# # # # Show the popularity scores in descending order
# # # st.subheader('Most Popular Games Based on Popularity Score')
# # # st.dataframe(df_games_enriched[['QueryName', 'Popularity_score']].sort_values('Popularity_score', ascending=False))

# # # # Show games with negative popularity score (if any)
# # # st.subheader('Games with Negative Popularity Score')
# # # negative_popularity_games = df_games_enriched[df_games_enriched['Popularity_score'] < 0]['QueryName']
# # # if not negative_popularity_games.empty:
# # #     st.write(negative_popularity_games)
# # # else:
# # #     st.write("No games have a negative popularity score.")

# # # # Option to download the enriched data with predictions
# # # st.subheader('Download Predictions and Popularity Scores')
# # # st.write('You can download the data with the predicted popularity scores here.')
# # # csv = df_games_enriched.to_csv(index=False).encode('utf-8')
# # # st.download_button(label="Download as CSV", data=csv, file_name='games_with_popularity_scores.csv', mime='text/csv')

# # # # Filtering Options
# # # st.subheader('Data Filtering Options')

# # # # Feature selection for filtering
# # # features = ['DeveloperCount', 'RecommendationCount', 'PublisherCount',
# # #             'SteamSpyPlayersEstimate', 'Achievements', 'Adventure', 'Casual',
# # #             'Indie', 'RPG', 'Free To Play', 'Action', 'Strategy', 'Simulation',
# # #             'Racing', 'Sports', 'Massively Multiplayer', 'Education', 'Violent',
# # #             'Design & Illustration', 'Animation & Modeling', 'Co-op',
# # #             'Cross-Platform Multiplayer', 'Family Sharing', 'HDR available',
# # #             'In-App Purchases', 'Multi-player', 'VR Support', 'age_ranking',
# # #             'Price', 'dlc_count', 'rating', 'TotalReviews', 'ReviewScore', 'Sales',
# # #             'avg_playtime', 'YearDifference', 'Month_sin', 'Month_cos',
# # #             'balance_pos_neg', 'PurchaseAvail', 'CategorySinglePlayer',
# # #             'TotalPoints', 'Budget_AA', 'Budget_AAA', 'Budget_Indie']

# # # selected_features = st.multiselect("Select features to filter by", features)

# # # # Initialize an empty DataFrame to hold filtered results
# # # filtered_df = df_games_enriched

# # # # Filtering logic based on selected features
# # # if selected_features:
# # #     for feature in selected_features:
# # #         if feature in df_games_enriched.columns:
# # #             min_value = st.slider(f"Select minimum value for {feature}", float(df_games_enriched[feature].min()), float(df_games_enriched[feature].max()), float(df_games_enriched[feature].min()))
# # #             filtered_df = filtered_df[filtered_df[feature] >= min_value]

# # # st.subheader('Filtered Data')
# # # st.write(filtered_df[['QueryName', 'Popularity_score']])

# # # # Add a dropdown for selecting a specific game
# # # st.subheader('Select a Game to View Details')
# # # selected_game = st.selectbox("Select a game:", filtered_df['QueryName'])

# # # # Display details for the selected game
# # # if selected_game:
# # #     game_details = filtered_df[filtered_df['QueryName'] == selected_game]
# # #     st.write(game_details)

# # #     # Show additional metrics if desired
# # #     st.write(f"Predicted Owners: {game_details['Pred_Owners'].values[0]}")
# # #     st.write(f"Popularity Score: {game_details['Popularity_score'].values[0]}")

# # import streamlit as st
# # import pandas as pd
# # import os

# # # Set the title of the app
# # st.title("Game Popularity and Budget Analysis")

# # # Load the merged dataset
# # file_path = os.path.abspath('data/merged_df_games_with_ranking_df_games_enriched.csv')
# # df = pd.read_csv(file_path)

# # # Sidebar for filtering options
# # st.sidebar.header("Filter Options")

# # # QueryName Selection
# # queryname_filter = st.sidebar.selectbox("Select Game by QueryName:", df['QueryName'].unique())

# # # Filter by Budget Types
# # budget_aa = st.sidebar.checkbox('Budget_AA')
# # budget_aaa = st.sidebar.checkbox('Budget_AAA')
# # budget_indie = st.sidebar.checkbox('Budget_Indie')

# # # Filter the dataset based on the selected QueryName
# # df_filtered = df[df['QueryName'] == queryname_filter]

# # # Filter by budget type
# # if budget_aa:
# # #     df_filtered = df_filtered[df_filtered['Budget_AA'] == 1]
# # # if budget_aaa:
# # #     df_filtered = df_filtered[df_filtered['Budget_AAA'] == 1]
# # # if budget_indie:
# # #     df_filtered = df_filtered[df_filtered['Budget_Indie'] == 1]

# # # # Display filtered data
# # # st.write("Filtered Games Data:")
# # # st.dataframe(df_filtered)

# # # # Show important game information in the table
# # # st.write("Game Information")
# # # columns_to_show = [
# # #     'QueryName', 'DeveloperCount', 'RecommendationCount', 'PublisherCount',
# # #     'SteamSpyOwners', 'SteamSpyPlayersEstimate', 'PurchaseAvail',
# # #     'CategorySinglePlayer', 'Price', 'TotalReviews', 'ReviewScore',
# # #     'Popularity_score', 'Pred_Owners', 'Sales', 'avg_playtime'
# # # ]

# # # st.write(df_filtered[columns_to_show])

# # # # Visualization (Optional)
# # # st.write("Popularity Score vs. Sales")
# # # st.bar_chart(df_filtered.set_index('QueryName')[['Popularity_score', 'Sales']])

# # # # Footer
# # # st.sidebar.write("Select filter options to explore the game dataset by budget, QueryName, and popularity score.")

# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import os
# # # Load dataset
# # @st.cache_data
# # def load_data():
# #     file_path = os.path.abspath('data/merged_df_games_with_ranking_df_games_enriched.csv')
# #     df = pd.read_csv(file_path)

# #     return df
# # df = load_data()
# # # Filter columns for binary features (assuming budget and binary columns like Budget_AA are already binary)
# # binary_features = ['Budget_AA', 'Budget_AAA', 'Budget_Indie']  # Modify this based on your binary features
# # game_columns = ['QueryName', 'Popularity_score', 'TotalReviews', 'Sales', 'Pred_Owners', 'avg_playtime', 'DeveloperCount']
# # # Extract OHE genre columns
# # genre_columns = [col for col in df.columns if 'Genre' in col]
# # # Create sidebar filters for binary features
# # st.sidebar.header('Filter by Binary Features')
# # selected_binary = st.sidebar.multiselect('Select binary features to filter', binary_features, default=binary_features)
# # # Filter by budget
# # st.sidebar.header('Filter by Budget')
# # budget_options = df[['Budget_AA', 'Budget_AAA', 'Budget_Indie']].columns.tolist()
# # selected_budget = st.sidebar.selectbox('Select a budget type', budget_options)
# # # Filter by genre
# # st.sidebar.header('Filter by Genre')
# # selected_genre = st.sidebar.selectbox('Select a genre', genre_columns)
# # # Filter by game names (sorted by popularity score in descending order by default)
# # df_sorted = df.sort_values(by='Popularity_score', ascending=False)
# # selected_game_name = st.sidebar.selectbox('Select a game', df_sorted['QueryName'])
# # # Define thresholds for popularity score to categorize as High, Medium, Low
# # max_popularity = df['Popularity_score'].max()
# # min_popularity = df['Popularity_score'].min()
# # threshold_high = min_popularity + (max_popularity - min_popularity) * 0.67
# # threshold_medium = min_popularity + (max_popularity - min_popularity) * 0.33
# # # Apply binary filters
# # filtered_df = df
# # if selected_binary:
# #     for feature in selected_binary:
# #         filtered_df = filtered_df[filtered_df[feature] == 1]
# # # Apply budget filter
# # if selected_budget:
# #     filtered_df = filtered_df[filtered_df[selected_budget] == 1]
# # # Apply genre filter
# # if selected_genre:
# #     filtered_df = filtered_df[filtered_df[selected_genre] == 1]
# # # Display the filtered game list without the index and AppID
# # st.header('Filtered Games')
# # st.dataframe(filtered_df[['QueryName', 'Popularity_score', 'TotalReviews', 'Sales', 'Pred_Owners', 'avg_playtime', 'DeveloperCount']])
# # # Display information for the selected game
# # st.header(f"Details of '{selected_game_name}'")
# # selected_game = df[df['QueryName'] == selected_game_name]
# # # Display game information
# # st.write(f"**About the game:** {selected_game['QueryName'].values[0]}")
# # st.write(f"**Release Date:** {selected_game['Unnamed: 11'].values[0]}")
# # st.write(f"**Predicted Owners:** {selected_game['Pred_Owners'].values[0]}")
# # st.write(f"**Total Reviews:** {selected_game['TotalReviews'].values[0]}")
# # st.write(f"**Average Playtime (in minutes):** {selected_game['avg_playtime'].values[0]}")
# # st.write(f"**Developer Count:** {selected_game['DeveloperCount'].values[0]}")
# # # Determine popularity level
# # popularity_score = selected_game['Popularity_score'].values[0]
# # if popularity_score >= threshold_high:
# #     popularity_label = 'High'
# # elif popularity_score >= threshold_medium:
# #     popularity_label = 'Medium'
# # else:
# #     popularity_label = 'Low'
# # st.write(f"**Popularity Level:** {popularity_label} ({popularity_score})")
# # # Create simple charts for the selected game
# # st.header('Charts for the Selected Game')
# # # Bar chart for Total Reviews
# # st.subheader('Total Reviews')
# # fig, ax = plt.subplots()
# # ax.bar(selected_game['QueryName'], selected_game['TotalReviews'], color='blue')
# # ax.set_ylabel('Total Reviews')
# # ax.set_title(f"Total Reviews of {selected_game_name}")
# # st.pyplot(fig)
# # # Bar chart for Predicted Owners
# # st.subheader('Predicted Owners')
# # fig, ax = plt.subplots()
# # ax.bar(selected_game['QueryName'], selected_game['Pred_Owners'], color='green')
# # ax.set_ylabel('Predicted Owners')
# # ax.set_title(f"Predicted Owners of {selected_game_name}")
# # st.pyplot(fig)
# # # Bar chart for Average Playtime
# # st.subheader('Average Playtime')
# # fig, ax = plt.subplots()
# # ax.bar(selected_game['QueryName'], selected_game['avg_playtime'], color='orange')
# # ax.set_ylabel('Average Playtime (minutes)')
# # ax.set_title(f"Average Playtime of {selected_game_name}")
# # st.pyplot(fig)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import os

# # Load dataset
# @st.cache_data
# def load_data():
#     file_path = os.path.abspath('data/merged_df_games_with_ranking_df_games_enriched.csv')
#     df = pd.read_csv(file_path)
#     return df

# df = load_data()

# # Filter columns for binary features, including the new ones
# binary_features = [
#     'Achievements', 'Adventure',
#     'Casual', 'Indie', 'RPG', 'Free To Play', 'Action', 'Strategy', 'Simulation',
#     'Racing', 'Sports', 'Massively Multiplayer', 'Education', 'Violent',
#     'Design & Illustration', 'Animation & Modeling', 'Co-op', 'Cross-Platform Multiplayer',
#     'Family Sharing', 'HDR available', 'In-App Purchases', 'Multi-player',
#     'VR Support', 'age_ranking'
# ]

# game_columns = ['QueryName', 'Popularity_score', 'TotalReviews', 'Sales', 'Pred_Owners', 'avg_playtime', 'DeveloperCount']

# # Extract OHE genre columns
# genre_columns = [col for col in df.columns if 'Genre' in col]

# # Create sidebar filters for binary features
# st.sidebar.header('Filter by Binary Features')
# selected_binary = st.sidebar.multiselect('Select binary features to filter', binary_features)

# # Filter by budget
# st.sidebar.header('Filter by Budget')
# budget_options = df[['Budget_AAA', 'Budget_AA', 'Budget_Indie']].columns.tolist()
# selected_budget = st.sidebar.selectbox('Select a budget type', budget_options)

# # Filter by genre
# st.sidebar.header('Filter by Genre')
# selected_genre = st.sidebar.selectbox('Select a genre', genre_columns)

# # Filter by game names (sorted by popularity score in descending order by default)
# df_sorted = df.sort_values(by='Popularity_score', ascending=False)
# selected_game_name = st.sidebar.selectbox('Select a game', df_sorted['QueryName'])

# # Define thresholds for popularity score to categorize as High, Medium, Low
# max_popularity = df['Popularity_score'].max()
# min_popularity = df['Popularity_score'].min()
# threshold_high = min_popularity + (max_popularity - min_popularity) * 0.67
# threshold_medium = min_popularity + (max_popularity - min_popularity) * 0.33

# # Apply binary filters
# filtered_df = df
# if selected_binary:
#     for feature in selected_binary:
#         filtered_df = filtered_df[filtered_df[feature] == 1]

# # Apply budget filter
# if selected_budget:
#     filtered_df = filtered_df[filtered_df[selected_budget] == 1]

# # Apply genre filter
# if selected_genre:
#     filtered_df = filtered_df[filtered_df[selected_genre] == 1]

# # Display the filtered game list without the index and AppID
# st.header('Filtered Games')
# st.dataframe(filtered_df[['QueryName', 'Popularity_score', 'TotalReviews', 'Sales', 'Pred_Owners']])

# # Display information for the selected game
# st.header(f"Details of '{selected_game_name}'")
# selected_game = df[df['QueryName'] == selected_game_name]

# # Display game information
# st.write(f"**About the game:** {selected_game['about_the_game'].values[0]}")
# st.write(f"**Release Date:** {selected_game['Release date'].values[0]}")
# st.write(f"**Predicted Owners:** {np.round(selected_game['Pred_Owners'].values[0])}")
# st.write(f"**Total Reviews:** {np.round(selected_game['TotalReviews'].values[0])}")
# st.write(f"**Average Playtime (in minutes):** {np.round(selected_game['avg_playtime'].values[0])}")
# st.write(f"**Developer Count:** {selected_game['DeveloperCount'].values[0]}")

# # Determine popularity level
# popularity_score = selected_game['Popularity_score'].values[0]
# if popularity_score >= threshold_high:
#     popularity_label = 'High'
# elif popularity_score >= threshold_medium:
#     popularity_label = 'Medium'
# else:
#     popularity_label = 'Low'

# st.write(f"**Popularity Level:** {popularity_label} ({popularity_score})")
import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page title
st.set_page_config(page_title="Game Remake Estimator", page_icon="🎮")

# Add the main title and description
st.markdown("<h1 style='text-align: center; color: #333333; font-family:Georgia;'>Game Remake Estimator</h1>", unsafe_allow_html=True)
st.markdown("""<p style='text-align: center; font-size: 18px; color: #4d4d4d; font-family:Georgia;'>
This page provides the predicted potential remake score for games, estimated using a Gradient Boosting Regressor. If a game has a high score, it indicates the game is well known and may deserve a remake. Explore the games below to see which ones are gaining traction!
</p>""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    file_path = os.path.abspath('data/merged_df_games_with_ranking_df_games_enriched.csv')
    df = pd.read_csv(file_path, index_col=False)
    return df

df = load_data()
df = df[df['QueryName'] != 'Counter-Strike: Condition Zero Deleted Scenes']

# Rescale the popularity score from 100 to 0
max_popularity = df['Popularity_score'].max()
min_popularity = df['Popularity_score'].min()
df['Rescaled_Popularity_Score'] = ((df['Popularity_score'] - min_popularity) / (max_popularity - min_popularity)) * 100

# Filter columns for binary features
binary_features = ['Adventure', 'Casual', 'RPG', 'Action', 'Strategy', 'Simulation', 'Sports', 'Massively Multiplayer', 'Education', 'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op', 'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases', 'Multi-player', 'VR Support']

# Sidebar for filters
st.sidebar.header('🎮 Filter by Genre')
selected_binary = st.sidebar.multiselect('Select a genre', binary_features)

# Budget filter
st.sidebar.header('💰 Filter by Budget')
budget_options = df[['Budget_AAA', 'Budget_AA', 'Budget_Indie']].columns.tolist()
selected_budget = st.sidebar.selectbox('Select a budget type', budget_options)

# Apply binary filters
filtered_df = df
if selected_binary:
    for feature in selected_binary:
        filtered_df = filtered_df[filtered_df[feature] == 1]

# Apply budget filter
if selected_budget:
    filtered_df = filtered_df[filtered_df[selected_budget] == 1]

# Sort and filter games
filtered_df_sorted = filtered_df.sort_values(by='Rescaled_Popularity_Score', ascending=False)
selected_game_name = st.sidebar.selectbox('Select a game', filtered_df_sorted['QueryName'])

# Display the filtered game list
st.markdown("<h2 style='text-align: center; color: #333333; font-family:Georgia;'>Games Released in 2014 and Earlier</h2>", unsafe_allow_html=True)

# Display the filtered DataFrame without the index column
st.markdown("""<style>
.dataframe table {
    margin: 20px auto;
    border-collapse: collapse;
}
.dataframe th, .dataframe td {
    padding: 10px 15px;
}
.css-1d391kg {
    display: none;
}
</style>""", unsafe_allow_html=True)

st.dataframe(filtered_df[['QueryName' ,'TotalReviews', 'Sales', 'Pred_Owners']].reset_index(drop=True))

# Calculate popularity thresholds
max_rescaled_popularity = df['Rescaled_Popularity_Score'].max()
min_rescaled_popularity = df['Rescaled_Popularity_Score'].min()
threshold_high = min_rescaled_popularity + (max_rescaled_popularity - min_rescaled_popularity) * 0.70
threshold_medium = min_rescaled_popularity + (max_rescaled_popularity - min_rescaled_popularity) * 0.30

# Display information for the selected game
st.markdown(f"<h2 style='color: #4d4d4d; font-family:Georgia;'>Game Details: <em>{selected_game_name}</em></h2>", unsafe_allow_html=True)
selected_game = df[df['QueryName'] == selected_game_name]

# Check if selected_game is empty
if not selected_game.empty:
    # Get rescaled popularity score
    rescaled_popularity_score = selected_game['Rescaled_Popularity_Score'].values[0]

    # Adjust the popularity score based on the budget type
    if selected_budget == 'Budget_AAA' and selected_game['Budget_AAA'].values[0] == 1:
        rescaled_popularity_score = min(rescaled_popularity_score * 1.3, 100)  # Increase by 20% for AAA games, capped at 100
    elif selected_budget == 'Budget_AA' and selected_game['Budget_AA'].values[0] == 1:
        rescaled_popularity_score = min(rescaled_popularity_score * 1.7, 100)  # Increase by 10% for AA games, capped at 100
    elif selected_budget == 'Budget_Indie' and selected_game['Budget_Indie'].values[0] == 1:
        rescaled_popularity_score = min(rescaled_popularity_score * 4.0, 100)  # Decrease by 20% for Indie games, capped at 100

    # Determine popularity level based on rescaled score
    if rescaled_popularity_score >= threshold_high:
        popularity_label = 'High'
        st.markdown("<h3 style='color: #00FF00;'>🚀 This game is highly regarded and is ideal for a remake! 🚀</h3>", unsafe_allow_html=True)
    elif rescaled_popularity_score >= threshold_medium:
        popularity_label = 'Medium'
        st.markdown("<h3 style='color: #FFBF00;'>⚖️ This game has moderate appeal and could benefit from a remake! ⚖️</h3>", unsafe_allow_html=True)
    else:
        popularity_label = 'Low'
        st.markdown("<h3 style='color: #FF6347;'>💤 This game has a modest following, making a remake less likely. 💤</h3>", unsafe_allow_html=True)

    # Display game information
    st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: #333333;'>Estimated Percentage: {popularity_label} ({int(np.round(rescaled_popularity_score))}%)</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 22px;'><strong>Predicted Owners:</strong> {int(np.round(selected_game['Pred_Owners'].values[0]))}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 18px;'><strong>Release Date:</strong> {selected_game['Release date'].values[0]}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 18px;'><strong>Total Reviews:</strong> {int(np.round(selected_game['TotalReviews'].values[0]))}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 18px;'><strong>Average Playtime (Hours):</strong> {int(np.round(selected_game['avg_playtime'].values[0]))}</p>", unsafe_allow_html=True)

    # Split 'about_the_game' text into paragraphs
    about_text = selected_game['about_the_game'].values[0].split('. ')

    # Display 'about_the_game' section
    st.markdown(f"<p style='font-size: 18px;'><strong>About the game:</strong></p>", unsafe_allow_html=True)
    for paragraph in about_text:
        st.markdown(f"<p style='font-size: 18px;'>{paragraph.strip()}.</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='color: red;'>No data available for the selected game.</p>", unsafe_allow_html=True)

# Footer with a sleek message
st.markdown("<hr style='border: 1px solid #D3D3D3;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #333333; font-family:Georgia;'>Crafted with precision by passionate Data Scientists</p>", unsafe_allow_html=True)
