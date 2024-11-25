# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import GradientBoostingRegressor
# import pickle
# import os

# # Construct the absolute file path
# file_path = os.path.abspath('project/app/trained_model.pkl')

# try:
#     # Open and load the model using pickle
#     with open(file_path, "rb") as file:
#         model = pickle.load(file)
#     print("Model loaded successfully using pickle.")
# except pickle.UnpicklingError as e:
#     print(f"Pickle UnpicklingError: {e}")
# except FileNotFoundError:
#     print("The specified file was not found.")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")

# # Thresholds for profit expectations
# AAA_profit_range = [5_000_000, 10_000_000]  # Example range for AAA games
# AA_profit_range = [1_000_000, 5_000_000]
# Indie_profit_range = [100_000, 1_000_000]

# # Streamlit App
# st.title("Game Success Predictor")
# st.header("Estimate the success of your game based on input features.")

# # Input Form
# st.sidebar.header("Input Features")
# game_price = st.sidebar.number_input("Game Price ($)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
# developer_count = st.sidebar.number_input("Number of Developers", min_value=1, max_value=100, value=5, step=1)
# reviews_count = st.sidebar.number_input("Number of Reviews", min_value=0, max_value=100000, value=500, step=100)
# cpu_ghz = st.sidebar.number_input("CPU (GHz)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
# ram_gb = st.sidebar.number_input("RAM (GB)", min_value=1, max_value=64, value=8, step=1)
# storage_gb = st.sidebar.number_input("Storage (GB)", min_value=10, max_value=500, value=50, step=10)

# # Input Data Preparation
# input_data = pd.DataFrame({
#     "Price": [game_price],
#     "DeveloperCount": [developer_count],
#     "Reviews": [reviews_count],
#     "cpu_ghz": [cpu_ghz],
#     "ram_gb": [ram_gb],
#     "storage_gb": [storage_gb]
# })

# # Make Predictions
# predicted_owners = model.predict(input_data)[0]
# predicted_revenue = predicted_owners * game_price

# # Assess Budget Rank
# if developer_count > 20 and game_price > 40:
#     budget_rank = "AAA"
# elif developer_count > 5 and game_price > 20:
#     budget_rank = "AA"
# else:
#     budget_rank = "Indie"

# # Profit Assessment
# if budget_rank == "AAA":
#     if AAA_profit_range[0] <= predicted_revenue <= AAA_profit_range[1]:
#         profit_assessment = "Above Expectations"
#     else:
#         profit_assessment = "Below Expectations"
# elif budget_rank == "AA":
#     if AA_profit_range[0] <= predicted_revenue <= AA_profit_range[1]:
#         profit_assessment = "Above Expectations"
#     else:
#         profit_assessment = "Below Expectations"
# else:
#     if Indie_profit_range[0] <= predicted_revenue <= Indie_profit_range[1]:
#         profit_assessment = "Above Expectations"
#     else:
#         profit_assessment = "Below Expectations"

# # Display Results
# st.header("Predicted Outputs")
# st.write(f"**Budget Rank:** {budget_rank}")
# st.write(f"**Predicted Owners:** {int(predicted_owners):,}")
# st.write(f"**Estimated Revenue:** ${predicted_revenue:,.2f}")
# st.write(f"**Profit Assessment:** {profit_assessment}")

# # Additional Insights
# st.subheader("Additional Insights")
# if profit_assessment == "Above Expectations":
#     st.success("Your game is predicted to perform well!")
# else:
#     st.warning("Consider optimizing your game's budget or pricing strategy.")
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Load Trained Model
file_path = os.path.abspath('project/app/trained_model.pkl')

try:
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    st.error("Model file not found. Please ensure the file path is correct.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Streamlit App Title
st.title("Game Success Predictor")
st.header("Estimate the success of your game based on key features.")

# Sidebar Inputs
st.sidebar.header("Input Features")

# Numerical Features
price = st.sidebar.number_input("Price ($)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
developer_count = st.sidebar.number_input("Developer Count", min_value=1, max_value=100, value=5, step=1)
recommendation_count = st.sidebar.number_input("Recommendation Count", min_value=0, max_value=100000, value=1000, step=100)
total_reviews = st.sidebar.number_input("Total Reviews", min_value=0, max_value=100000, value=500, step=100)
review_score = st.sidebar.slider("Review Score (0-100)", min_value=0, max_value=100, value=80, step=1)
sales = st.sidebar.number_input("Sales ($)", min_value=0.0, max_value=1_000_000.0, value=10000.0, step=1000.0)
avg_playtime = st.sidebar.number_input("Average Playtime (hours)", min_value=0.0, max_value=1000.0, value=10.0, step=1.0)
dlc_count = st.sidebar.number_input("DLC Count", min_value=0, max_value=50, value=2, step=1)
rating = st.sidebar.slider("Game Rating (1-5)", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
age_ranking = st.sidebar.slider("Age Ranking (1 to 18+)", min_value=1, max_value=18, value=12, step=1)
total_points = st.sidebar.number_input("Total Points", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
balance_pos_neg = st.sidebar.number_input("Balance of Positive/Negative Reviews", min_value=-1.0, max_value=1.0, value=0.5, step=0.1)
year_difference = st.sidebar.number_input("Years Since Release", min_value=0, max_value=20, value=4, step=1)
cpu_ghz = st.sidebar.number_input("CPU Speed (GHz)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
ram_gb = st.sidebar.number_input("RAM (GB)", min_value=1, max_value=64, value=8, step=1)
storage_gb = st.sidebar.number_input("Storage (GB)", min_value=10, max_value=500, value=50, step=10)

# Binary Features
achievements = st.sidebar.checkbox("Has Achievements", value=False)
adventure = st.sidebar.checkbox("Adventure Genre", value=False)
casual = st.sidebar.checkbox("Casual Genre", value=False)
indie = st.sidebar.checkbox("Indie Genre", value=False)
free_to_play = st.sidebar.checkbox("Free to Play", value=False)
action = st.sidebar.checkbox("Action Genre", value=False)
strategy = st.sidebar.checkbox("Strategy Genre", value=False)
simulation = st.sidebar.checkbox("Simulation Genre", value=False)
rpg = st.sidebar.checkbox("RPG Genre", value=False)
racing = st.sidebar.checkbox("Racing Genre", value=False)
sports = st.sidebar.checkbox("Sports Genre", value=False)
massively_multiplayer = st.sidebar.checkbox("Massively Multiplayer", value=False)
education = st.sidebar.checkbox("Education Genre", value=False)
violent = st.sidebar.checkbox("Violent Content", value=False)
design_illustration = st.sidebar.checkbox("Design & Illustration", value=False)
animation_modeling = st.sidebar.checkbox("Animation & Modeling", value=False)
co_op = st.sidebar.checkbox("Co-op", value=False)
cross_platform_multiplayer = st.sidebar.checkbox("Cross-Platform Multiplayer", value=False)
family_sharing = st.sidebar.checkbox("Family Sharing", value=False)
hdr_available = st.sidebar.checkbox("HDR Available", value=False)
in_app_purchases = st.sidebar.checkbox("In-App Purchases", value=False)
multi_player = st.sidebar.checkbox("Multi-player", value=False)
vr_support = st.sidebar.checkbox("VR Support", value=False)
purchase_avail = st.sidebar.checkbox("Purchase Available", value=False)
category_singleplayer = st.sidebar.checkbox("Category Single Player", value=False)

# Budget Features
budget_aa = st.sidebar.checkbox("Budget AA", value=False)
budget_aaa = st.sidebar.checkbox("Budget AAA", value=False)
budget_indie = st.sidebar.checkbox("Budget Indie", value=False)

# Price Categories
price_category_high = st.sidebar.checkbox("Price Category: High", value=False)
price_category_medium = st.sidebar.checkbox("Price Category: Medium", value=False)
price_category_low = st.sidebar.checkbox("Price Category: Low", value=False)

# Derived Features
month_sin = np.sin(np.pi / 6)  # Example placeholder value
month_cos = np.cos(np.pi / 6)  # Example placeholder value

# Prepare Input Data
input_data = pd.DataFrame({
    "RecommendationCount": [recommendation_count],
    "Achievements": [int(achievements)],
    "Adventure": [int(adventure)],
    "Casual": [int(casual)],
    "Indie": [int(indie)],
    "RPG": [int(rpg)],
    "Free To Play": [int(free_to_play)],
    "Action": [int(action)],
    "Strategy": [int(strategy)],
    "Simulation": [int(simulation)],
    "Racing": [int(racing)],
    "Sports": [int(sports)],
    "Massively Multiplayer": [int(massively_multiplayer)],
    "Education": [int(education)],
    "Violent": [int(violent)],
    "Design & Illustration": [int(design_illustration)],
    "Animation & Modeling": [int(animation_modeling)],
    "Co-op": [int(co_op)],
    "Cross-Platform Multiplayer": [int(cross_platform_multiplayer)],
    "Family Sharing": [int(family_sharing)],
    "HDR available": [int(hdr_available)],
    "In-App Purchases": [int(in_app_purchases)],
    "Multi-player": [int(multi_player)],
    "VR Support": [int(vr_support)],
    "age_ranking": [age_ranking],
    "Price": [price],
    "dlc_count": [dlc_count],
    "rating": [rating],
    "TotalReviews": [total_reviews],
    "ReviewScore": [review_score],
    "Sales": [sales],
    "avg_playtime": [avg_playtime],
    "YearDifference": [year_difference],
    "Month_sin": [month_sin],
    "Month_cos": [month_cos],
    "balance_pos_neg": [balance_pos_neg],
    "PurchaseAvail": [int(purchase_avail)],
    "CategorySinglePlayer": [int(category_singleplayer)],
    "TotalPoints": [total_points],
    "Budget_AA": [int(budget_aa)],
    "Budget_AAA": [int(budget_aaa)],
    "Budget_Indie": [int(budget_indie)],
    "cpu_ghz": [cpu_ghz],
    "ram_gb": [ram_gb],
    "storage_gb": [storage_gb],
    "PriceCategory_High": [int(price_category_high)],
    "PriceCategory_Low": [int(price_category_low)],
    "PriceCategory_Medium": [int(price_category_medium)],
})

# Reorder columns to match the model's expected order (ensure this order matches the model's feature order)
input_data = input_data[model.feature_names_in_]

# Prediction and Display
try:
    predicted_success = model.predict(input_data)[0]
    st.header("Predicted Outputs")
    st.write(f"**Predicted Success Score:** {predicted_success:.2f}")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")

# Additional Insights
st.subheader("Insights")
st.write("""
Adjust your game's features and see how they impact its predicted success.
This tool helps you optimize key decisions for development and marketing.
""")
