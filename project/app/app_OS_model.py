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
import altair as alt
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Load the trained XGBoost model for SteamSpyOwners
owners_model_path = 'project/app/SteamSpyOwners_xgb_model.pkl'
owners_model = joblib.load(owners_model_path)

st.set_page_config(page_title="New Game Estimator", page_icon="🎮")
st.link_button("Home Page","http://127.0.0.1:8000")
# Add the main title and description
st.markdown("<h1 style='text-align: center; color: #333333;'>New Game Estimator</h1>", unsafe_allow_html=True)
st.markdown("""<p style='text-align: center; font-size: 18px; color: #4d4d4d;'>
This page helps you predict the potential success of creating a new game! Using XGBoost, a powerful and fast prediction tool, the model analyzes your inputs to show which game ideas could attract more players. Try it out below and see which ideas could be a hit!!
</p>""", unsafe_allow_html=True)

# Set up the Streamlit app


# Sidebar inputs for the selected features
DeveloperCount = st.sidebar.number_input('Number of Developer teams ', min_value=1, value=1, max_value=50)
PublisherCount = 1

# Selectbox for Budget type (only one can be selected now)
Budget_features = st.sidebar.selectbox(
    'Select Budget type',
    options=['Budget_Indie', 'Budget_AA', 'Budget_AAA']
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

Achievements = st.sidebar.number_input('Number of Achievements', min_value=0, value=0)
Year = st.sidebar.number_input('Year', min_value=2025, value=2025)
Month = 1
# Select box for Price categories
price_category = st.sidebar.selectbox(
    'Select Price Category',
    options=['Free', '4.99', '29.99', '49.99', '69.99']
)

# Map the selected category to a numeric value for prediction
price_map = {
    'Free': 2.87,
    '4.99': 4.99,
    '29.99': 29.99,
    '49.99': 49.99,
    '69.99': 69.99
}
Price = price_map[price_category]

dlc_count = st.sidebar.number_input('Number of DLC', min_value=0, value=0, help='Additional content gamers download for video games after their initial release')

# Select box for age ranking (0-4 where 4 means "not specified age")
age_ranking = st.sidebar.selectbox(
    'Select Age Ranking',
    options=[0, 1, 2, 3, 4],
    format_func=lambda x: {
        0: "6 or less",
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
    'Budget_AA': [int('Budget_AA' == Budget_features)],   # Added Budget_AA feature
    'Budget_AAA': [int('Budget_AAA' == Budget_features)],  # Added Budget_AAA feature
    'Budget_Indie': [int('Budget_Indie' == Budget_features)]  # Added Budget_Indie feature
})

# Prediction button
if st.sidebar.button("Predict"):
    # Make prediction for SteamSpyOwners
    owners_prediction = owners_model.predict(input_data)
    predicted_owners = int(owners_prediction[0])


# Predicted output
    st.subheader("Predicted Values:")
    st.write(f"**Estimated Number of Owners:** {predicted_owners:,}")
     # Horizontal bar chart for predicted owners
    st.subheader("Predicted Owners")
    owners_data = pd.DataFrame({
        'Metric': ['Predicted Owners'],
        'Value': [predicted_owners]
    })
    owners_chart = alt.Chart(owners_data).mark_bar().encode(
        x='Value:Q',
        y=alt.Y('Metric:N'),
        tooltip=['Metric', 'Value']
    ).properties(
        width=500,
        height=100,
        title="Predicted Owners"
    )
    st.altair_chart(owners_chart)
# Popularity analysis
    if predicted_owners < 100000:
        st.write("This suggests a niche appeal. The game is more likely to attract a specific audience rather than a broad mainstream following.")
    elif predicted_owners < 500000:
        st.write("This indicates a moderate level of popularity. The game has the potential to gain a strong following, but it may not yet be a global phenomenon.")
    elif predicted_owners < 1000000:
        st.write("This shows that the game is fairly popular among players. It has achieved widespread attention, but it hasn't reached blockbuster status.")
    else:
        st.write("This indicates that the game is highly popular and well-received! It has garnered a large, enthusiastic player base and is performing at a top-tier level.")

    estimated_revenue = np.int64(np.round(Price * predicted_owners))
    st.write(f"**Estimated Revenue:** ${estimated_revenue:,}")
    sar = int(estimated_revenue * 3.75)
    st.write(f"**Estimated Revenue in (SAR):** {sar:,}")


    budget_data = {
    'Metric': ['Min Budget ($)', 'Max Budget ($)', 'Predicted Revenue ($)'],
    'Value': [
        # Min Budget
        100_000 if Budget_features == 'Budget_Indie' else
        1_000_000 if Budget_features == 'Budget_AA' else
        5_000_000,
        # Max Budget
        1_000_000 if Budget_features == 'Budget_Indie' else
        5_000_000 if Budget_features == 'Budget_AA' else
        10_000_000,
        # Predicted Revenue
        estimated_revenue
        ]
    }
    budget_df = pd.DataFrame(budget_data)

# Calculate profit and determine if it's profitable or at a loss
    average_budget = {
        'Budget_Indie': (100_000 + 1_000_000) / 3,
        'Budget_AA': (1_000_000 + 5_000_000) / 3,
        'Budget_AAA': (5_000_000 + 10_000_000) / 3
    }[Budget_features]

    profit = estimated_revenue - average_budget
    is_profitable = profit > 0

    # Display profitability status
    st.write("### Estimated general Profitability Analysis")
    if is_profitable:
        st.success(f"The game is **profitable** with an estimated profit of ${profit:,.2f}!")
    else:
        st.error(f"Your business is **at a loss** with an estimated deficit of ${-profit:,.2f}.")
        st.write("***Try to adjust the parameters to make it profitable***")
        st.write("")
    # Add colors to charts
    # Horizontal bar chart for budget and predicted revenue
    budget_chart = alt.Chart(budget_df).mark_bar().encode(
        x=alt.X('Value:Q', title="Value ($)"),
        y=alt.Y('Metric:N', sort='-x'),
        color=alt.condition(
            alt.datum.Metric == 'Predicted Revenue ($)',
            alt.value('green'),  # Highlight revenue in green
            alt.value('blue')    # Use blue for budget values
        ),
        tooltip=['Metric', 'Value']
    ).properties(
        width=600,
        height=300,
        title="Budget Range and Predicted Revenue"
    )
    st.altair_chart(budget_chart)


    # Display budget range based on the selected budget type
    if Budget_features == 'Budget_AAA':
        st.write("**Budget Range for AAA games:** 5,000,000 - 10,000,000")
        st.write("AAA games often have high production values, including extensive content, advanced graphics, and large teams behind their development.")
        st.write("**Minimum System Requirements (AAA):**")
        st.write("   - OS: Windows 10 64-bit")
        st.write("   - Processor: Intel i5-9600K / AMD Ryzen 5 2600")
        st.write("   - Memory: 16 GB RAM")
        st.write("   - Graphics: NVIDIA GeForce GTX 1660 / AMD Radeon RX 580")
        st.write("   - Storage: 50 GB available space")
    elif Budget_features == 'Budget_AA':
        st.write("**Budget Range for AA games:** 1,000,000 - 5,000,000")
        st.write("AA games generally focus on a mid-range production budget and are often created by smaller teams than AAA games.")
        st.write("**Minimum System Requirements (AA):**")
        st.write("   - OS: Windows 10 64-bit")
        st.write("   - Processor: Intel i5-8400 / AMD Ryzen 3 3300X")
        st.write("   - Memory: 8 GB RAM")
        st.write("   - Graphics: NVIDIA GeForce GTX 1050 Ti / AMD Radeon RX 560")
        st.write("   - Storage: 30 GB available space")
    elif Budget_features == 'Budget_Indie':
        st.write("**Budget Range for Indie games:** 100,000 - 1,000,000")
        st.write("Indie games are usually developed by small teams or even individual developers, with a focus on creativity and unique gameplay.")
        st.write("**Minimum System Requirements (Indie):**")
        st.write("   - OS: Windows 7/8/10")
        st.write("   - Processor: Intel i3-7100 / AMD Ryzen 3 1200")
        st.write("   - Memory: 4 GB RAM")
        st.write("   - Graphics: NVIDIA GeForce GTX 750 Ti / AMD Radeon R7 360")
        st.write("   - Storage: 5 GB available space")

    # Calculate and display the estimated revenue

    # Define a weight multiplier for DeveloperCount
    developer_cost_multiplier = 200_000  # Cost per developer
    developer_weighted_cost = DeveloperCount * developer_cost_multiplier

    # Update the average budget calculation to include the developer cost
    average_budget = {
        'Budget_Indie': (100_000 + 1_000_000) / 2 + developer_weighted_cost,
        'Budget_AA': (1_000_000 + 5_000_000) / 2 + developer_weighted_cost,
        'Budget_AAA': (5_000_000 + 10_000_000) / 2 + developer_weighted_cost
    }[Budget_features]

    # Update profit calculation to reflect weighted developer costs
    profit = estimated_revenue - average_budget
    is_profitable = profit > 0

    # Display profitability status
    st.write("### Estimated Profitability Analysis after deducting developers cost")
    if is_profitable:
        st.success(f"The game is **profitable** with an estimated profit of ${profit:,.2f}!")
    else:
        st.error(f"Your business is **at a loss** with an estimated deficit of ${-profit:,.2f}.")
        st.write("***Try to adjust the parameters to make it profitable***")
        st.write("")
    # Visualize the developer cost impact on the budget and revenue
    budget_data = {
        'Metric': ['Min Budget ($)', 'Max Budget ($)', 'Developer Cost ($)', 'Predicted Revenue ($)'],
        'Value': [
            # Min Budget
            100_000 if Budget_features == 'Budget_Indie' else
            1_000_000 if Budget_features == 'Budget_AA' else
            5_000_000,
            # Max Budget
            1_000_000 if Budget_features == 'Budget_Indie' else
            5_000_000 if Budget_features == 'Budget_AA' else
            10_000_000,
            # Developer Cost
            developer_weighted_cost,
            # Predicted Revenue
            estimated_revenue
        ]
    }
    budget_df = pd.DataFrame(budget_data)

    # Horizontal bar chart for budget and predicted revenue with developer cost
    budget_chart = alt.Chart(budget_df).mark_bar().encode(
        x=alt.X('Value:Q', title="Value ($)"),
        y=alt.Y('Metric:N', sort='-x'),
        color=alt.condition(
            alt.datum.Metric == 'Predicted Revenue ($)',
            alt.value('green'),  # Highlight revenue in green
            alt.value('blue')    # Use blue for budget values and developer cost
        ),
        tooltip=['Metric', 'Value']
    ).properties(
        width=600,
        height=300,
        title="Budget Range, Developers Cost, and Predicted Revenue"
    )
    st.altair_chart(budget_chart)

    # Display the impact of DeveloperCount on profitability

    st.write(f"**Developers Cost Impact:** ${developer_weighted_cost:,}")


    st.write("Increasing the number of developers increases the development cost, which may reduce profitability.")



    components.html("""<iframe src="http://localhost:8502"
                width="600" height="1000"
                frameborder="0"></iframe>""", height=1000)



st.write("**Please Note!! These predictions are based on the game features you provided. Please remember that actual values can vary based on various factors like marketing, reviews, and current trends.**")
