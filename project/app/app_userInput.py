import streamlit as st
import pandas as pd
import os

# Define the file path for user input data
file_path = "data/user/user_input_data.csv"

# Load dataset
@st.cache_data
def load_game_data():
    file_path_games = os.path.abspath('data/merged_df_games_with_ranking_df_games_enriched.csv')
    df_games = pd.read_csv(file_path_games, index_col=False)
    return df_games

df_games = load_game_data()

# Set page title
st.title("Let us know what you think")

# Form elements
with st.form(key="user_input_form"):
    # Text input fields
    name = st.text_input("Enter your name:")
    email = st.text_input("Enter your email:")
    favorite_game = st.selectbox('Select your favorite game', df_games['QueryName'])
    #favorite_game = st.selectbox('Select your favorite game', options=df_games['QueryName'])
    custom_game = st.text_input("Or type a new game name if it's not listed:")

# Determine the final input
    final_game = custom_game if custom_game.strip() else favorite_game

    st.write(f"Your selected game: {final_game}")
    feedback = st.text_area("Enter your feedback:")

    # Submit button
    submit_button = st.form_submit_button("Submit")

# Check if the form is submitted and validate required fields
if submit_button:
    # Validate the required fields
    if not name or not feedback:
        st.error("Name and Feedback are required fields!")
    else:
        # Check if the CSV file exists
        if os.path.exists(file_path):
            # Read the existing CSV file
            df_user = pd.read_csv(file_path)

            # Check for duplicate entries
            is_duplicate = df_user[(df_user["Name"] == name) & (df_user["Feedback"] == feedback)].any().any()
            if is_duplicate:
                st.warning("This feedback has already been submitted.")
            else:
                # Create a new entry as a dictionary
                new_entry = {
                    "Name": name,
                    "Email": email,
                    "Favorite Game": final_game,
                    "Feedback": feedback
                }
                # Append the new entry to the DataFrame
                df_user = pd.concat([df_user, pd.DataFrame([new_entry])], ignore_index=True)
                # Save the updated DataFrame back to CSV
                df_user.to_csv(file_path, index=False)
                st.success("Your submission has been saved!")
        else:
            # If the file does not exist, create a new DataFrame and save it as a CSV
            new_entry = {
                "Name": name,
                "Email": email,
                "Favorite Game": final_game,
                "Feedback": feedback
            }
            df_user = pd.DataFrame([new_entry])
            df_user.to_csv(file_path, index=False)
            st.success("Your submission has been saved!")

        # Optionally, display the saved data
        st.write(df_user)
