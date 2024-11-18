# # import streamlit as st
# # import pandas as pd
# # import joblib
# # import os

# # # Load the trained model
# # model_path = 'project/app/random_forest_model.pkl'
# # rf_model = joblib.load(model_path)

# # # Define the exact feature names and order used during model training
# # feature_names = [
# #     'DeveloperCount', 'RecommendationCount', 'PublisherCount',
# #     'PurchaseAvail', 'CategorySinglePlayer', 'Achievements', 'Year',
# #     'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
# #     'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
# #     'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
# #     'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
# #     'Multi-player', 'VR Support', 'age_ranking', 'Price', 'dlc_count',
# #     'positive', 'negative', 'num_reviews_total', 'rating', 'TotalReviews',
# #     'ReviewScore'
# # ]

# # # Function to collect user input
# # def get_user_input():
# #     # Sidebar for user input
# #     st.sidebar.header("Game Features")

# #     # Multi-select for binary features
# #     Category_features = [
# #         'Purchase Availability', 'Category Single Player', 'Adventure', 'Casual',
# #         'Indie', 'RPG', 'Action', 'Strategy', 'Simulation',
# #         'Racing', 'Sports', 'Massively Multiplayer', 'Education',
# #         'Violent', 'Design & Illustration', 'Animation & Modeling',
# #         'Co-op', 'Cross-Platform Multiplayer', 'Family Sharing',
# #         'In-App Purchases', 'Multi-player', 'VR Support'
# #     ]

# #     selected_features = st.sidebar.multiselect(
# #         'Select Game Features:',
# #         Category_features
# # #     )

# # #     # Create a dictionary to store the values for binary features
# # #     feature_values = {feature: 1 if feature in selected_features else 0 for feature in Category_features}

# # #     # Numeric feature inputs
# # #     DeveloperCount = st.sidebar.number_input('Developer Count', min_value=1, value=10)
# # #     RecommendationCount = st.sidebar.number_input('Recommendation Count', min_value=0, value=10000)
# # #     PublisherCount = st.sidebar.number_input('Publisher Count', min_value=1, value=1)
# # #     Achievements = st.sidebar.number_input('Achievements', min_value=0, value=1)
# # #     Year = st.sidebar.number_input('Year', min_value=2024, max_value=2034, value=2024)
# # #     Month = st.sidebar.number_input('Month', min_value=1, max_value=12, value=1)
# # #     Price = st.sidebar.number_input('Price', min_value=0.0, value=29.99)
# # #     dlc_count = st.sidebar.number_input('DLC Count', min_value=0, value=5)
# # #     positive = st.sidebar.number_input('Positive Reviews', min_value=0, value=20000)
# # #     negative = st.sidebar.number_input('Negative Reviews', min_value=0, value=1000)
# # #     num_reviews_total = st.sidebar.number_input('Total Reviews', min_value=0, value=21000)
# # #     rating = st.sidebar.number_input('Rating', min_value=0.0, max_value=10.0, value=7.5)
# # #     TotalReviews = st.sidebar.number_input('Total Review Count', min_value=0, value=21000)
# # #     age_ranking = st.sidebar.number_input('Age Ranking', min_value=0, max_value=18, value=0)
# # #     ReviewScore = st.sidebar.number_input('Review Score', min_value=0.0, max_value=10.0, value=7.5)

# # #     # Create a DataFrame for model input, ensuring the order is correct
# # #     input_data = {
# # #         'DeveloperCount': DeveloperCount,
# # #         'RecommendationCount': RecommendationCount,
# # #         'PublisherCount': PublisherCount,
# # #         'PurchaseAvail': feature_values['Purchase Availability'],
# # #         'CategorySinglePlayer': feature_values['Category Single Player'],
# # #         'Achievements': Achievements,
# # #         'Year': Year,
# # #         'Month': Month,
# # #         'Adventure': feature_values['Adventure'],
# # #         'Casual': feature_values['Casual'],
# # #         'Indie': feature_values['Indie'],
# # #         'RPG': feature_values['RPG'],
# # #         'Action': feature_values['Action'],
# # #         'Strategy': feature_values['Strategy'],
# # #         'Simulation': feature_values['Simulation'],
# # #         'Racing': feature_values['Racing'],
# # #         'Sports': feature_values['Sports'],
# # #         'Massively Multiplayer': feature_values['Massively Multiplayer'],
# # #         'Education': feature_values['Education'],
# # #         'Violent': feature_values['Violent'],
# # #         'Design & Illustration': feature_values['Design & Illustration'],
# # #         'Animation & Modeling': feature_values['Animation & Modeling'],
# # #         'Co-op': feature_values['Co-op'],
# # #         'Cross-Platform Multiplayer': feature_values['Cross-Platform Multiplayer'],
# # #         'Family Sharing': feature_values['Family Sharing'],
# # #         'In-App Purchases': feature_values['In-App Purchases'],
# # #         'Multi-player': feature_values['Multi-player'],
# # #         'VR Support': feature_values['VR Support'],
# # #         'Price': Price,
# # #         'dlc_count': dlc_count,
# # #         'positive': positive,
# # #         'negative': negative,
# # #         'num_reviews_total': num_reviews_total,
# # #         'rating': rating,
# # #         'TotalReviews': TotalReviews,
# # #         'age_ranking': age_ranking,
# # #         'ReviewScore': ReviewScore
# # #     }

# # #     # Create DataFrame with the correct order of feature names
# # #     input_df = pd.DataFrame(input_data, index=[0])[feature_names]

# # #     return input_df

# # # # Streamlit App Title
# # # st.title("Steam Game Ownership Prediction")
# # # st.write("### Input the details about the game, and the model will predict the estimated number of Steam Spy Owners.")

# # # # Get user input
# # # input_df = get_user_input()

# # # # Display user input for confirmation
# # # st.subheader('User Input')
# # # st.write(input_df)

# # # # Make prediction with the model
# # # try:
# # #     prediction = rf_model.predict(input_df)
# # #     st.subheader('Predicted Steam Spy Owners')
# # #     st.write(f"Estimated Steam Spy Owners: {int(prediction[0]):,}")
# # # except Exception as e:
# # #     st.error(f"Error in prediction: {e}")







# # import streamlit as st
# # import pandas as pd
# # import joblib

# # # Load the trained model
# # model_path = 'project/app/random_forest_model.pkl'
# # model = joblib.load(model_path)

# # # Set up the Streamlit app
# # st.title("Game Popularity Predictor")

# # # Sidebar inputs for the selected features
# # DeveloperCount = st.sidebar.number_input('Developer Count', min_value=0, value=0)
# # PublisherCount = st.sidebar.number_input('Publisher Count', min_value=0, value=0)

# # # Multi-select for binary features
# # Category_features = st.sidebar.multiselect(
# #     'Select Binary Features',
# #     options=[
# #         'Category Single Player',
# #         'Adventure',
# #         'Casual',
# #         'Indie',
# #         'RPG',
# #         'Action',
# #         'Strategy',
# #         'Simulation',
# #         'Racing',
# #         'Sports',
# #         'Massively Multiplayer',
# #         'Education',
# #         'Violent',
# #         'Design & Illustration',
# #         'Animation & Modeling',
# #         'Co-op',
# #         'Cross-Platform Multiplayer',
# #         'Family Sharing',
# #         'In-App Purchases',
# #         'Multi-player',
# #         'VR Support'
# #     ]
# # )

# # Achievements = st.sidebar.number_input('Achievements', min_value=0, value=0)
# # Year = st.sidebar.number_input('Year', min_value=2000, value=2022)
# # Month = st.sidebar.number_input('Month', min_value=1, max_value=12, value=1)

# # # Select box for Price categories
# # price_category = st.sidebar.selectbox(
# #     'Select Price Category',
# #     options=['Free', '4.99', '29.99', '49.99','69.99']
# # )

# # # Map the selected category to a numeric value for prediction
# # price_map = {
# #     'Free': 0,
# #     '4.99': 4.99,   # Midpoint of the range
# #     '29.99': 29.99, # Midpoint of the range
# #     '49.99': 49.99, # Midpoint of the range
# #     '69.99': 69.99    # You can choose a reasonable value, e.g., 50
# # }
# # Price = price_map[price_category]

# # dlc_count = st.sidebar.number_input('DLC Count', min_value=0, value=0)

# # # Select box for age ranking (0-4 where 4 means "not specified age")
# # age_ranking = st.sidebar.selectbox(
# #     'Select Age Ranking',
# #     options=[0, 1, 2, 3, 4],
# #     format_func=lambda x: {
# #         0: "6 <",
# #         1: "6-12",
# #         2: "12-16",
# #         3: "+18",
# #         4: "Not Specified"
# #     }[x]
# # )

# # # Create a DataFrame from the input data
# # input_data = pd.DataFrame({
# #     'DeveloperCount': [DeveloperCount],
# #     'PublisherCount': [PublisherCount],
# #     'CategorySinglePlayer': [int('Category Single Player' in Category_features)],
# #     'Achievements': [Achievements],
# #     'Year': [Year],
# #     'Month': [Month],
# #     'Adventure': [int('Adventure' in Category_features)],
# #     'Casual': [int('Casual' in Category_features)],
# #     'Indie': [int('Indie' in Category_features)],
# #     'RPG': [int('RPG' in Category_features)],
# #     'Action': [int('Action' in Category_features)],
# #     'Strategy': [int('Strategy' in Category_features)],
# #     'Simulation': [int('Simulation' in Category_features)],
# #     'Racing': [int('Racing' in Category_features)],
# #     'Sports': [int('Sports' in Category_features)],
# #     'Massively Multiplayer': [int('Massively Multiplayer' in Category_features)],
# #     'Education': [int('Education' in Category_features)],
# #     'Violent': [int('Violent' in Category_features)],
# #     'Design & Illustration': [int('Design & Illustration' in Category_features)],
# #     'Animation & Modeling': [int('Animation & Modeling' in Category_features)],
# #     'Co-op': [int('Co-op' in Category_features)],
# #     'Cross-Platform Multiplayer': [int('Cross-Platform Multiplayer' in Category_features)],
# #     'Family Sharing': [int('Family Sharing' in Category_features)],
# #     'In-App Purchases': [int('In-App Purchases' in Category_features)],
# #     'Multi-player': [int('Multi-player' in Category_features)],
# #     'VR Support': [int('VR Support' in Category_features)],
# #     'Price': [Price],
# #     'dlc_count': [dlc_count],
# #     'age_ranking': [age_ranking]  # Include age_ranking in the input data
# # })

# # # Make a prediction
# # prediction = model.predict(input_data)

# # # Display the result
# # st.subheader("Predicted Number of Owners:")
# # predicted_owners = int(prediction[0])
# # # Present the result with more context
# # if predicted_owners < 1000:
# #     st.write(f"The estimated number of owners for this game is **{predicted_owners}**. This suggests a niche appeal.")
# # elif predicted_owners < 10000:
# #     st.write(f"The estimated number of owners for this game is **{predicted_owners}**. This indicates a moderate level of popularity.")
# # elif predicted_owners < 100000:
# #     st.write(f"The estimated number of owners for this game is **{predicted_owners}**. This shows that the game is fairly popular among players.")
# # else:
# #     st.write(f"The estimated number of owners for this game is **{predicted_owners}**. This indicates that the game is highly popular and well-received!")

# # st.write("This prediction is based on the game features you provided. Please remember that actual sales can vary based on various factors like marketing, reviews, and current trends.")




# import streamlit as st
# import pandas as pd
# import joblib

# # Load the trained models
# owners_model_path = 'project/app/SteamSpyOwners_model.pkl'
# owners_model = joblib.load(owners_model_path)

# review_score_model_path = 'project/app/ReviewScore_model.pkl'
# review_score_model = joblib.load(review_score_model_path)

# rating_model_path = 'project/app/rating_model.pkl'
# rating_model = joblib.load(rating_model_path)

# # Set up the Streamlit app
# st.title("Game Popularity Predictor")

# # Sidebar inputs for the selected features
# DeveloperCount = st.sidebar.number_input('Developer Count', min_value=0, value=0)
# PublisherCount = st.sidebar.number_input('Publisher Count', min_value=0, value=0)

# # Multi-select for binary features
# Category_features = st.sidebar.multiselect(
#     'Select Binary Features',
#     options=[
#         'Category Single Player',
#         'Adventure',
#         'Casual',
#         'Indie',
#         'RPG',
#         'Action',
#         'Strategy',
# #         'Simulation',
# #         'Racing',
# #         'Sports',
# #         'Massively Multiplayer',
# #         'Education',
# #         'Violent',
# #         'Design & Illustration',
# #         'Animation & Modeling',
# #         'Co-op',
# #         'Cross-Platform Multiplayer',
# #         'Family Sharing',
# #         'In-App Purchases',
# #         'Multi-player',
# #         'VR Support'
# #     ]
# # )

# # Achievements = st.sidebar.number_input('Achievements', min_value=0, value=0)
# # Year = st.sidebar.number_input('Year', min_value=2000, value=2022)
# # Month = st.sidebar.number_input('Month', min_value=1, max_value=12, value=1)

# # # Select box for Price categories
# # price_category = st.sidebar.selectbox(
# #     'Select Price Category',
# #     options=['Free', '4.99', '29.99', '49.99', '69.99']
# # )

# # # Map the selected category to a numeric value for prediction
# # price_map = {
# #     'Free': 0,
# #     '4.99': 4.99,
# #     '29.99': 29.99,
# #     '49.99': 49.99,
# #     '69.99': 69.99
# # }
# # Price = price_map[price_category]

# # dlc_count = st.sidebar.number_input('DLC Count', min_value=0, value=0)

# # # Select box for age ranking (0-4 where 4 means "not specified age")
# # age_ranking = st.sidebar.selectbox(
# #     'Select Age Ranking',
# #     options=[0, 1, 2, 3, 4],
# #     format_func=lambda x: {
# #         0: "6 <",
# #         1: "6-12",
# #         2: "12-16",
# #         3: "+18",
# #         4: "Not Specified"
# #     }[x]
# # )

# # # Create a DataFrame from the input data
# # input_data = pd.DataFrame({
# #     'DeveloperCount': [DeveloperCount],
# #     'PublisherCount': [PublisherCount],
# #     'CategorySinglePlayer': [int('Category Single Player' in Category_features)],
# #     'Achievements': [Achievements],
# #     'Year': [Year],
# #     'Month': [Month],
# #     'Adventure': [int('Adventure' in Category_features)],
# #     'Casual': [int('Casual' in Category_features)],
# #     'Indie': [int('Indie' in Category_features)],
# #     'RPG': [int('RPG' in Category_features)],
# #     'Action': [int('Action' in Category_features)],
# #     'Strategy': [int('Strategy' in Category_features)],
# #     'Simulation': [int('Simulation' in Category_features)],
# #     'Racing': [int('Racing' in Category_features)],
# #     'Sports': [int('Sports' in Category_features)],
# #     'Massively Multiplayer': [int('Massively Multiplayer' in Category_features)],
# #     'Education': [int('Education' in Category_features)],
# #     'Violent': [int('Violent' in Category_features)],
# #     'Design & Illustration': [int('Design & Illustration' in Category_features)],
# #     'Animation & Modeling': [int('Animation & Modeling' in Category_features)],
# #     'Co-op': [int('Co-op' in Category_features)],
# #     'Cross-Platform Multiplayer': [int('Cross-Platform Multiplayer' in Category_features)],
# #     'Family Sharing': [int('Family Sharing' in Category_features)],
# #     'In-App Purchases': [int('In-App Purchases' in Category_features)],
# #     'Multi-player': [int('Multi-player' in Category_features)],
# #     'VR Support': [int('VR Support' in Category_features)],
# #     'Price': [Price],
# #     'dlc_count': [dlc_count],
# #     'age_ranking': [age_ranking]  # Include age_ranking in the input data
# # })

# # # Make predictions with all models
# # owners_prediction = owners_model.predict(input_data)
# # review_score_prediction = review_score_model.predict(input_data)
# # rating_prediction = rating_model.predict(input_data)

# # # Display the results
# # st.subheader("Predicted Values:")
# # predicted_owners = int(owners_prediction[0])
# # predicted_review_score = int(review_score_prediction[0])
# # predicted_rating = int(rating_prediction[0])

# # # Present the results with more context
# # st.write(f"**Estimated Number of Owners:** {predicted_owners}")
# # if predicted_owners < 1000:
# #     st.write("This suggests a niche appeal.")
# # elif predicted_owners < 10000:
# #     st.write("This indicates a moderate level of popularity.")
# # elif predicted_owners < 100000:
# #     st.write("This shows that the game is fairly popular among players.")
# # else:
# #     st.write("This indicates that the game is highly popular and well-received!")

# # st.write(f"**Predicted Review Score:** {predicted_review_score}")
# # st.write(f"**Predicted Rating:** {predicted_rating}")

# # st.write("These predictions are based on the game features you provided. Please remember that actual values can vary based on various factors like marketing, reviews, and current trends.")

# import streamlit as st
# import pandas as pd
# import joblib

# # Load the trained models
# owners_model_path = 'project/app/SteamSpyOwners_xgb_model.pkl'
# owners_model = joblib.load(owners_model_path)

# review_score_model_path = 'project/app/ReviewScore_model.pkl'
# review_score_model = joblib.load(review_score_model_path)

# rating_model_path = 'project/app/rating_model.pkl'
# rating_model = joblib.load(rating_model_path)

# # Set up the Streamlit app
# st.title("Game Popularity Predictor")

# # Sidebar inputs for the selected features
# DeveloperCount = st.sidebar.number_input('Developer Count', min_value=0, value=0)
# PublisherCount = st.sidebar.number_input('Publisher Count', min_value=0, value=0)

# # Multi-select for binary features
# Category_features = st.sidebar.multiselect(
#     'Select Binary Features',
#     options=[
#         'Category Single Player',
#         'Adventure',
#         'Casual',
#         'Indie',
#         'RPG',
#         'Action',
#         'Strategy',
#         'Simulation',
#         'Racing',
# #         'Sports',
# #         'Massively Multiplayer',
# #         'Education',
# #         'Violent',
# #         'Design & Illustration',
# #         'Animation & Modeling',
# #         'Co-op',
# #         'Cross-Platform Multiplayer',
# #         'Family Sharing',
# #         'In-App Purchases',
# #         'Multi-player',
# #         'VR Support'
# #     ]
# # )

# # Achievements = st.sidebar.number_input('Achievements', min_value=0, value=0)
# # Year = st.sidebar.number_input('Year', min_value=2000, value=2022)
# # Month = st.sidebar.number_input('Month', min_value=1, max_value=12, value=1)

# # # Select box for Price categories
# # price_category = st.sidebar.selectbox(
# #     'Select Price Category',
# #     options=['Free', '4.99', '29.99', '49.99', '69.99']
# # )

# # # Map the selected category to a numeric value for prediction
# # price_map = {
# #     'Free': 0,
# #     '4.99': 4.99,
# #     '29.99': 29.99,
# #     '49.99': 49.99,
# #     '69.99': 69.99
# # }
# # Price = price_map[price_category]

# # dlc_count = st.sidebar.number_input('DLC Count', min_value=0, value=0)

# # # Select box for age ranking (0-4 where 4 means "not specified age")
# # age_ranking = st.sidebar.selectbox(
# #     'Select Age Ranking',
# #     options=[0, 1, 2, 3, 4],
# #     format_func=lambda x: {
# #         0: "6 <",
# #         1: "6-12",
# #         2: "12-16",
# #         3: "+18",
# #         4: "Not Specified"
# #     }[x]
# # )

# # # Create a DataFrame from the input data
# # input_data = pd.DataFrame({
# #     'DeveloperCount': [DeveloperCount],
# #     'PublisherCount': [PublisherCount],
# #     'CategorySinglePlayer': [int('Category Single Player' in Category_features)],
# #     'Achievements': [Achievements],
# #     'Year': [Year],
# #     'Month': [Month],
# #     'Adventure': [int('Adventure' in Category_features)],
# #     'Casual': [int('Casual' in Category_features)],
# #     'Indie': [int('Indie' in Category_features)],
# #     'RPG': [int('RPG' in Category_features)],
# #     'Action': [int('Action' in Category_features)],
# #     'Strategy': [int('Strategy' in Category_features)],
# #     'Simulation': [int('Simulation' in Category_features)],
# #     'Racing': [int('Racing' in Category_features)],
# #     'Sports': [int('Sports' in Category_features)],
# #     'Massively Multiplayer': [int('Massively Multiplayer' in Category_features)],
# #     'Education': [int('Education' in Category_features)],
# #     'Violent': [int('Violent' in Category_features)],
# #     'Design & Illustration': [int('Design & Illustration' in Category_features)],
# #     'Animation & Modeling': [int('Animation & Modeling' in Category_features)],
# #     'Co-op': [int('Co-op' in Category_features)],
# #     'Cross-Platform Multiplayer': [int('Cross-Platform Multiplayer' in Category_features)],
# #     'Family Sharing': [int('Family Sharing' in Category_features)],
# #     'In-App Purchases': [int('In-App Purchases' in Category_features)],
# #     'Multi-player': [int('Multi-player' in Category_features)],
# #     'VR Support': [int('VR Support' in Category_features)],
# #     'Price': [Price],
# #     'dlc_count': [dlc_count],
# #     'age_ranking': [age_ranking],
# #     'SteamSpyOwners': [0]  # Add a placeholder for SteamSpyOwners
# # })

# # # Make prediction for SteamSpyOwners
# # owners_prediction = owners_model.predict(input_data)

# # # Use the predicted owners to create a new input DataFrame for ReviewScore and rating
# # # Add the predicted owners to a new DataFrame
# # input_data_with_owners = input_data.copy()
# # input_data_with_owners['SteamSpyOwners'] = owners_prediction

# # # Make predictions for ReviewScore and rating
# # review_score_prediction = review_score_model.predict(input_data_with_owners)
# # rating_prediction = rating_model.predict(input_data_with_owners)

# # # Display the results
# # st.subheader("Predicted Values:")
# # predicted_owners = int(owners_prediction[0])
# # predicted_review_score = int(review_score_prediction[0])
# # predicted_rating = int(rating_prediction[0])

# # # Present the results with more context
# # st.write(f"**Estimated Number of Owners:** {predicted_owners}")
# # if predicted_owners < 1000:
# #     st.write("This suggests a niche appeal.")
# # elif predicted_owners < 10000:
# #     st.write("This indicates a moderate level of popularity.")
# # elif predicted_owners < 100000:
# #     st.write("This shows that the game is fairly popular among players.")
# # else:
# #     st.write("This indicates that the game is highly popular and well-received!")

# # st.write(f"**Predicted Review Score:** {predicted_review_score}")
# # st.write(f"**Predicted Rating:** {predicted_rating}")

# # st.write("These predictions are based on the game features you provided. Please remember that actual values can vary based on various factors like marketing, reviews, and current trends.")



# import streamlit as st
# import pandas as pd
# import joblib

# # Load the trained model for SteamSpyOwners
# owners_model_path = 'project/app/SteamSpyOwners_xgb_model.pkl'
# owners_model = joblib.load(owners_model_path)

# # Set up the Streamlit app
# st.title("Game Popularity Predictor")

# # Sidebar inputs for the selected features
# DeveloperCount = st.sidebar.number_input('Developer Count', min_value=0, value=0)
# PublisherCount = st.sidebar.number_input('Publisher Count', min_value=0, value=0)

# # Multi-select for binary features
# Category_features = st.sidebar.multiselect(
#     'Select Binary Features',
#     options=[
#         'Category Single Player',
#         'Adventure',
#         'Casual',
#         'Indie',
#         'RPG',
#         'Action',
#         'Strategy',
#         'Simulation',
#         'Racing',
#         'Sports',
#         'Massively Multiplayer',
#         'Education',
#         'Violent',
#         'Design & Illustration',
#         'Animation & Modeling',
#         'Co-op',
#         'Cross-Platform Multiplayer',
#         'Family Sharing',
#         'In-App Purchases',
#         'Multi-player',
#         'VR Support'
#     ]
# )

# Achievements = st.sidebar.number_input('Achievements', min_value=0, value=0)
# Year = st.sidebar.number_input('Year', min_value=2000, value=2022)
# Month = st.sidebar.number_input('Month', min_value=1, max_value=12, value=1)

# # Select box for Price categories
# price_category = st.sidebar.selectbox(
#     'Select Price Category',
#     options=['Free', '4.99', '29.99', '49.99', '69.99']
# )

# # Map the selected category to a numeric value for prediction
# price_map = {
#     'Free': 0,
#     '4.99': 4.99,
#     '29.99': 29.99,
#     '49.99': 49.99,
#     '69.99': 69.99
# }
# Price = price_map[price_category]

# dlc_count = st.sidebar.number_input('DLC Count', min_value=0, value=0)

# # Select box for age ranking (0-4 where 4 means "not specified age")
# age_ranking = st.sidebar.selectbox(
#     'Select Age Ranking',
#     options=[0, 1, 2, 3, 4],
#     format_func=lambda x: {
#         0: "6 <",
#         1: "6-12",
#         2: "12-16",
#         3: "+18",
#         4: "Not Specified"
#     }[x]
# )

# # Create a DataFrame from the input data
# input_data = pd.DataFrame({
#     'DeveloperCount': [DeveloperCount],
#     'PublisherCount': [PublisherCount],
#     'CategorySinglePlayer': [int('Category Single Player' in Category_features)],
#     'Achievements': [Achievements],
#     'Year': [Year],
#     'Month': [Month],
#     'Adventure': [int('Adventure' in Category_features)],
#     'Casual': [int('Casual' in Category_features)],
#     'Indie': [int('Indie' in Category_features)],
#     'RPG': [int('RPG' in Category_features)],
#     'Action': [int('Action' in Category_features)],
#     'Strategy': [int('Strategy' in Category_features)],
#     'Simulation': [int('Simulation' in Category_features)],
#     'Racing': [int('Racing' in Category_features)],
#     'Sports': [int('Sports' in Category_features)],
#     'Massively Multiplayer': [int('Massively Multiplayer' in Category_features)],
#     'Education': [int('Education' in Category_features)],
#     'Violent': [int('Violent' in Category_features)],
#     'Design & Illustration': [int('Design & Illustration' in Category_features)],
#     'Animation & Modeling': [int('Animation & Modeling' in Category_features)],
#     'Co-op': [int('Co-op' in Category_features)],
#     'Cross-Platform Multiplayer': [int('Cross-Platform Multiplayer' in Category_features)],
#     'Family Sharing': [int('Family Sharing' in Category_features)],
#     'In-App Purchases': [int('In-App Purchases' in Category_features)],
#     'Multi-player': [int('Multi-player' in Category_features)],
#     'VR Support': [int('VR Support' in Category_features)],
#     'Price': [Price],
#     'dlc_count': [dlc_count],
#     'age_ranking': [age_ranking],
# })

# # Make prediction for SteamSpyOwners
# owners_prediction = owners_model.predict(input_data)

# # Display the results
# st.subheader("Predicted Number of Owners:")
# predicted_owners = int(owners_prediction[0])

# # Present the results with more context
# st.write(f"**Estimated Number of Owners:** {predicted_owners}")
# if predicted_owners < 10000:
#     st.write("This suggests a niche appeal.")
# elif predicted_owners < 60000:
#     st.write("This indicates a moderate level of popularity.")
# elif predicted_owners < 500000:
#     st.write("This shows that the game is fairly popular among players.")
# else:
#     st.write("This indicates that the game is highly popular and well-received!")

# st.write("These predictions are based on the game features you provided. Please remember that actual values can vary based on various factors like marketing, reviews, and current trends.")

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# Load the trained XGBoost model for SteamSpyOwners
owners_model_path = 'project/app/SteamSpyOwners_xgb_model.pkl'
owners_model = joblib.load(owners_model_path)

# Set up the Streamlit app
st.title("Game Popularity Predictor (SteamSpyOwners)")

# Sidebar inputs for the selected features
DeveloperCount = st.sidebar.number_input('Developer Count', min_value=0, value=0)
PublisherCount = st.sidebar.number_input('Publisher Count', min_value=0, value=0)


Budget_features = st.sidebar.multiselect(
    'Select Budget type',
    options=[
        'Budget_AA',
        'Budget_AAA',
        'Budget_Indie'  # Added new binary features

    ]
)



# Multi-select for binary features
Category_features = st.sidebar.multiselect(
    'Select Games Category',
    options=[
        'Category Single Player',
        'Adventure',
        'Casual',
        'Indie',
        'RPG',
        'Action',
        'Strategy',
        'Simulation',
        'Racing',
        'Sports',
        'Massively Multiplayer',
        'Education',
        'Violent',
        'Design & Illustration',
        'Animation & Modeling',
        'Co-op',
        'Cross-Platform Multiplayer',
        'Family Sharing',
        'In-App Purchases',
        'Multi-player',
        'VR Support'

    ]
)




Achievements = st.sidebar.number_input('Achievements', min_value=0, value=0)
Year = st.sidebar.number_input('Year', min_value=2000, value=2022)
Month = st.sidebar.number_input('Month', min_value=1, max_value=12, value=1)

# Select box for Price categories
price_category = st.sidebar.selectbox(
    'Select Price Category',
    options=['Free', '4.99', '29.99', '49.99', '69.99']
)

# Map the selected category to a numeric value for prediction
price_map = {
    'Free': 0,
    '4.99': 4.99,
    '29.99': 29.99,
    '49.99': 49.99,
    '69.99': 69.99
}
Price = price_map[price_category]

dlc_count = st.sidebar.number_input('DLC Count', min_value=0, value=0)

# Select box for age ranking (0-4 where 4 means "not specified age")
age_ranking = st.sidebar.selectbox(
    'Select Age Ranking',
    options=[0, 1, 2, 3, 4],
    format_func=lambda x: {
        0: "6 <",
        1: "6-12",
        2: "12-16",
        3: "+18",
        4: "Not Specified"
    }[x]
)

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'DeveloperCount': [DeveloperCount],
    'PublisherCount': [PublisherCount],
    'CategorySinglePlayer': [int('Category Single Player' in Category_features)],
    'Achievements': [Achievements],
    'Year': [Year],
    'Month': [Month],
    'Adventure': [int('Adventure' in Category_features)],
    'Casual': [int('Casual' in Category_features)],
    'Indie': [int('Indie' in Category_features)],
    'RPG': [int('RPG' in Category_features)],
    'Action': [int('Action' in Category_features)],
    'Strategy': [int('Strategy' in Category_features)],
    'Simulation': [int('Simulation' in Category_features)],
    'Racing': [int('Racing' in Category_features)],
    'Sports': [int('Sports' in Category_features)],
    'Massively Multiplayer': [int('Massively Multiplayer' in Category_features)],
    'Education': [int('Education' in Category_features)],
    'Violent': [int('Violent' in Category_features)],
    'Design & Illustration': [int('Design & Illustration' in Category_features)],
    'Animation & Modeling': [int('Animation & Modeling' in Category_features)],
    'Co-op': [int('Co-op' in Category_features)],
    'Cross-Platform Multiplayer': [int('Cross-Platform Multiplayer' in Category_features)],
    'Family Sharing': [int('Family Sharing' in Category_features)],
    'In-App Purchases': [int('In-App Purchases' in Category_features)],
    'Multi-player': [int('Multi-player' in Category_features)],
    'VR Support': [int('VR Support' in Category_features)],
    'Price': [Price],
    'dlc_count': [dlc_count],
    'age_ranking': [age_ranking],
    'Budget_AA': [int('Budget_AA' in Budget_features)],   # Added Budget_AA feature
    'Budget_AAA': [int('Budget_AAA' in Budget_features)],  # Added Budget_AAA feature
    'Budget_Indie': [int('Budget_Indie' in Budget_features)]  # Added Budget_Indie feature
})

# Make prediction for SteamSpyOwners
owners_prediction = owners_model.predict(input_data)
predicted_owners = int(owners_prediction[0])

# Display the results
st.subheader("Predicted Values:")
st.write(f"**Estimated Number of Owners:** {predicted_owners}")

if predicted_owners < 1000:
    st.write("This suggests a niche appeal.")
elif predicted_owners < 10000:
    st.write("This indicates a moderate level of popularity.")
elif predicted_owners < 100000:
    st.write("This shows that the game is fairly popular among players.")
else:
    st.write("This indicates that the game is highly popular and well-received!")

# Only perform cross-validation if there are enough samples
if len(input_data) > 1:
    # Cross-validation and performance metrics (MAE, R²)
    cv_mae = cross_val_score(owners_model, input_data, owners_prediction, cv=5, scoring='neg_mean_absolute_error')
    cv_r2 = cross_val_score(owners_model, input_data, owners_prediction, cv=5, scoring='r2')

    st.write(f"**Mean Absolute Error (MAE):** {-cv_mae.mean():,.2f}")
    st.write(f"**R² Score:** {cv_r2.mean():.4f}")
else:
    st.write("Not enough data for cross-validation. Add more samples for evaluation.")

st.write("These predictions are based on the game features you provided. Please remember that actual values can vary based on various factors like marketing, reviews, and current trends.")
