# # import pandas as pd
# # from google.cloud import bigquery
# # from project.params import *
# # import os
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # from sklearn.preprocessing import RobustScaler
# # from scipy.stats import randint
# # import joblib

# # # Function to clean data
# # def clean_data(df: pd.DataFrame) -> pd.DataFrame:
# #     df = df.drop_duplicates()
# #     df = df.dropna(how='any', axis=0)
# #     return df

# # def get_data(source=True) -> pd.DataFrame:
# #     if source:
# #         file_path = os.path.abspath('data/GamesFinishDeletedDeveloperCountZero.csv')
# #         return pd.read_csv(file_path)
# #     else:
# #         client = bigquery.Client(project=GCP_PROJECT)
# #         query = "SELECT * FROM your_dataset.your_table"  # Specify your query here
# #         query_job = client.query(query)
# #         result = query_job.result()
# #         df = result.to_dataframe()
# #     return df

# # # Load and clean data
# # df = get_data()
# # df = df.drop(columns=['Unnamed: 0', 'SteamSpyPlayersEstimate', 'Month_sin', 'Month_cos', 'YearDifference', 'Free To Play', 'HDR available', 'Sales'])
# # df = clean_data(df)

# # # Define features
# # # feature_cols_for_ReviewScore_rating = ['SteamSpyOwners','DeveloperCount', 'PublisherCount', 'CategorySinglePlayer', 'Achievements', 'Year',
# # #                 'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
# # #                 'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
# # #                 'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
# # #                 'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
# # #                 'Multi-player', 'VR Support', 'Price', 'dlc_count', 'age_ranking']
# # feature_cols_for_SteamSpyOwners = ['DeveloperCount', 'PublisherCount', 'CategorySinglePlayer', 'Achievements', 'Year',
# #                 'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
# #                 'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
# #                 'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
# #                 'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
# #                 'Multi-player', 'VR Support', 'Price', 'dlc_count', 'age_ranking']


# # #Prepare models for ReviewScore, rating, and SteamSpyOwners (expected sales)
# # #targets = ['ReviewScore', 'rating']  # Assuming SteamSpyOwners represents expected sales

# # # Dictionary to hold models and scalers
# # models = {}
# # scalers = {}

# # # for target in targets:
# # #     print(f"Training model for {target}...")

# # #     X = df[feature_cols_for_ReviewScore_rating]
# # #     y = df[target]

# # #     # Train-Test Split
# # #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # #     # Scale the features
# # #     scaler = RobustScaler()
# # #     X_train_scaled = scaler.fit_transform(X_train)
# # #     X_test_scaled = scaler.transform(X_test)

# # #     # Initialize Random Forest with default hyperparameters
# # #     rf_model = RandomForestRegressor(random_state=42)

# # #     # Hyperparameter Tuning with RandomizedSearchCV
# # #     param_dist = {
# # #         'n_estimators': randint(100, 500),
# # #         'max_depth': randint(3, 20),
# # #         'min_samples_split': randint(2, 10),
# # #         'min_samples_leaf': randint(1, 10),
# # #         'max_features': ['sqrt', 'log2']
# # #     }

# # #     # RandomizedSearchCV for hyperparameter tuning
# # #     random_search = RandomizedSearchCV(
# # #         rf_model, param_distributions=param_dist,
# # #         n_iter=50, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
# # #     )
# # #     random_search.fit(X_train_scaled, y_train)

# # #     # Best model after tuning
# # #     best_rf_model = random_search.best_estimator_

# # #     # Save the best model
# # #     model_path = os.path.abspath(f'project/app/{target}_model.pkl')
# # #     joblib.dump(best_rf_model, model_path)
# # #     print(f"Model saved as {model_path}")

# # #     # Evaluate the model on test data
# # #     y_pred_train = best_rf_model.predict(X_train_scaled)
# # #     y_pred_test = best_rf_model.predict(X_test_scaled)

# # #     # Performance metrics
# # #     def print_evaluation_metrics(y_true, y_pred, dataset_type="Test"):
# # #         print(f"--- {dataset_type} Set Performance for {target} ---")
# # #         print(f"R²: {r2_score(y_true, y_pred):.4f}")
# # #         print(f"MAE: {mean_absolute_error(y_true, y_pred):,.2f}")
# # #         print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):,.2f}")

# # #     # Training set evaluation
# # #     print_evaluation_metrics(y_train, y_pred_train, dataset_type="Train")

# # #     # Test set evaluation
# # #     print_evaluation_metrics(y_test, y_pred_test, dataset_type="Test")

# # #     # Cross-Validation Score
# # #     cv_scores = cross_val_score(best_rf_model, scaler.transform(X), y, cv=5, scoring='r2')
# # #     print(f"Cross-Validation R² scores: {cv_scores}")
# # #     print(f"Mean Cross-Validation R² for {target}: {cv_scores.mean():.4f}")

# # #     # Store the model and scaler
# # #     models[target] = best_rf_model
# # #     scalers[target] = scaler

# # targets_S = ['SteamSpyOwners']

# # for target in targets_S:
# #     print(f"Training model for {target}...")

# #     X = df[feature_cols_for_SteamSpyOwners]
# #     y = df[target]

# #     # Train-Test Split
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Scale the features
# #     scaler = RobustScaler()
# #     X_train_scaled = scaler.fit_transform(X_train)
# #     X_test_scaled = scaler.transform(X_test)

# #     # Initialize Random Forest with default hyperparameters
# #     rf_model = RandomForestRegressor(random_state=42)

# #     # Hyperparameter Tuning with RandomizedSearchCV
# #     param_dist = {
# #         'n_estimators': randint(100, 500),
# #         'max_depth': randint(3, 20),
# #         'min_samples_split': randint(2, 10),
# #         'min_samples_leaf': randint(1, 10),
# #         'max_features': ['sqrt', 'log2']
# #     }

# #     # RandomizedSearchCV for hyperparameter tuning
# #     random_search = RandomizedSearchCV(
# #         rf_model, param_distributions=param_dist,
# #         n_iter=50, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
# #     )
# #     random_search.fit(X_train_scaled, y_train)

# #     # Best model after tuning
# #     best_rf_model = random_search.best_estimator_

# #     # Save the best model
# #     model_path = os.path.abspath(f'project/app/{target}_model.pkl')
# #     joblib.dump(best_rf_model, model_path)
# #     print(f"Model saved as {model_path}")

# #     # Evaluate the model on test data
# #     y_pred_train = best_rf_model.predict(X_train_scaled)
# #     y_pred_test = best_rf_model.predict(X_test_scaled)

# #     # Performance metrics
# #     def print_evaluation_metrics(y_true, y_pred, dataset_type="Test"):
# #         print(f"--- {dataset_type} Set Performance for {target} ---")
# #         print(f"R²: {r2_score(y_true, y_pred):.4f}")
# #         print(f"MAE: {mean_absolute_error(y_true, y_pred):,.2f}")
# #         print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):,.2f}")

# #     # Training set evaluation
# #     print_evaluation_metrics(y_train, y_pred_train, dataset_type="Train")

# #     # Test set evaluation
# #     print_evaluation_metrics(y_test, y_pred_test, dataset_type="Test")

# #     # Cross-Validation Score
# #     cv_scores = cross_val_score(best_rf_model, scaler.transform(X), y, cv=5, scoring='r2')
# #     print(f"Cross-Validation R² scores: {cv_scores}")
# #     print(f"Mean Cross-Validation R² for {target}: {cv_scores.mean():.4f}")

# #     # Store the model and scaler
# #     models[target] = best_rf_model
# #     scalers[target] = scaler

# # import pandas as pd
# # from google.cloud import bigquery
# # from project.params import *
# # import os
# # import joblib
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import RobustScaler
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense
# # from tensorflow.keras.callbacks import EarlyStopping
# # from tensorflow.keras.optimizers import Adam

# # # Function to clean data
# # def clean_data(df: pd.DataFrame) -> pd.DataFrame:
# #     df = df.drop_duplicates()
# #     df = df.dropna(how='any', axis=0)
# #     return df

# # def get_data(source=True) -> pd.DataFrame:
# #     if source:
# #         file_path = os.path.abspath('data/GamesFinishDeletedDeveloperCountZero.csv')
# #         return pd.read_csv(file_path)
# #     else:
# #         client = bigquery.Client(project=GCP_PROJECT)
# #         query = "SELECT * FROM your_dataset.your_table"  # Specify your query here
# #         query_job = client.query(query)
# #         result = query_job.result()
# #         df = result.to_dataframe()
# #     return df

# # # Load and clean data
# # df = get_data()
# # df = df.drop(columns=['Unnamed: 0', 'SteamSpyPlayersEstimate', 'Month_sin', 'Month_cos', 'YearDifference', 'Free To Play', 'HDR available', 'Sales'])
# # df = clean_data(df)

# # # Define features
# # feature_cols_for_ReviewScore_rating = ['SteamSpyOwners','DeveloperCount', 'PublisherCount', 'CategorySinglePlayer', 'Achievements', 'Year',
# #                 'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
# #                 'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
# #                 'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
# #                 'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
# #                 'Multi-player', 'VR Support', 'Price', 'dlc_count', 'age_ranking']
# # feature_cols_for_SteamSpyOwners = ['DeveloperCount', 'PublisherCount', 'CategorySinglePlayer', 'Achievements', 'Year',
# #                 'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
# #                 'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
# #                 'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
# #                 'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
# #                 'Multi-player', 'VR Support', 'Price', 'dlc_count', 'age_ranking']

# # # Prepare deep learning models for ReviewScore, rating, and SteamSpyOwners
# # targets = ['ReviewScore', 'rating']

# # # Dictionary to hold models and scalers
# # models = {}
# # scalers = {}

# # # Function to create and compile a neural network model
# # def build_model(input_dim):
# #     model = Sequential()
# #     model.add(Dense(10, activation='relu', input_dim=input_dim))
# #     model.add(Dense(5, activation='relu'))
# #     model.add(Dense(2, activation='relu'))
# #     model.add(Dense(1,activation='linear'))  # Output layer
# #     model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
# #     return model

# # # Training and evaluation for each target
# # for target in targets:
# #     print(f"Training model for {target}...")

# #     X = df[feature_cols_for_ReviewScore_rating]
# #     y = df[target]

# #     # Train-Test Split
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Scale the features
# #     scaler = RobustScaler()
# #     X_train_scaled = scaler.fit_transform(X_train)
# #     X_test_scaled = scaler.transform(X_test)

# #     # Build and train the neural network
# #     model = build_model(X_train_scaled.shape[1])

# #     # Early stopping to prevent overfitting
# #     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# #     # Train the model
# #     model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# #     # Evaluate the model
# #     y_pred_train = model.predict(X_train_scaled)
# #     y_pred_test = model.predict(X_test_scaled)

# #     # Performance metrics
# #     def print_evaluation_metrics(y_true, y_pred, dataset_type="Test"):
# #         print(f"--- {dataset_type} Set Performance for {target} ---")
# #         print(f"R²: {r2_score(y_true, y_pred):.4f}")
# #         print(f"MAE: {mean_absolute_error(y_true, y_pred):,.2f}")
# #         print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):,.2f}")

# #     # Training set evaluation
# #     print_evaluation_metrics(y_train, y_pred_train, dataset_type="Train")

# #     # Test set evaluation
# #     print_evaluation_metrics(y_test, y_pred_test, dataset_type="Test")

# #     # Save the model and scaler
# #     model_path = os.path.abspath(f'project/app/{target}_model.h5')
# #     model.save(model_path)
# #     print(f"Model saved as {model_path}")

# #     # Store the model and scaler
# #     models[target] = model
# #     scalers[target] = scaler

# # # Repeat for SteamSpyOwners target
# # targets_S = ['SteamSpyOwners']

# # for target in targets_S:
# #     print(f"Training model for {target}...")

# #     X = df[feature_cols_for_SteamSpyOwners]
# #     y = df[target]

# #     # Train-Test Split
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Scale the features
# #     scaler = RobustScaler()
# #     X_train_scaled = scaler.fit_transform(X_train)
# #     X_test_scaled = scaler.transform(X_test)

# #     # Build and train the neural network
# #     model = build_model(X_train_scaled.shape[1])

# #     # Early stopping to prevent overfitting
# #     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# #     # Train the model
# #     model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# #     # Evaluate the model
# #     y_pred_train = model.predict(X_train_scaled)
# #     y_pred_test = model.predict(X_test_scaled)

# #     # Performance metrics
# #     def print_evaluation_metrics(y_true, y_pred, dataset_type="Test"):
# #         print(f"--- {dataset_type} Set Performance for {target} ---")
# #         print(f"R²: {r2_score(y_true, y_pred):.4f}")
# #         print(f"MAE: {mean_absolute_error(y_true, y_pred):,.2f}")
# #         print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):,.2f}")

# #     # Training set evaluation
# #     print_evaluation_metrics(y_train, y_pred_train, dataset_type="Train")

# #     # Test set evaluation
# #     print_evaluation_metrics(y_test, y_pred_test, dataset_type="Test")

# #     # Save the model and scaler
# #     model_path = os.path.abspath(f'project/app/{target}_model.h5')
# #     model.save(model_path)
# #     print(f"Model saved as {model_path}")

# #     # Store the model and scaler
# #     models[target] = model
# #     scalers[target] = scaler

# # import pandas as pd
# # from google.cloud import bigquery
# # from project.params import *
# # import os
# # import joblib
# # from sklearn.model_selection import train_test_split, RandomizedSearchCV
# # from sklearn.preprocessing import RobustScaler,MinMaxScaler
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # from xgboost import XGBRegressor
# # from scipy.stats import randint

# # # Function to clean data
# # def clean_data(df: pd.DataFrame) -> pd.DataFrame:
# #     df = df.drop_duplicates()
# #     df = df.dropna(how='any', axis=0)
# #     return df

# # def get_data(source=True) -> pd.DataFrame:
# #     if source:
# #         file_path = os.path.abspath('data/GamesFinishDeletedDeveloperCountZero.csv')
# #         return pd.read_csv(file_path)
# #     else:
# #         client = bigquery.Client(project=GCP_PROJECT)
# #         query = "SELECT * FROM your_dataset.your_table"  # Specify your query here
# #         query_job = client.query(query)
# #         result = query_job.result()
# #         df = result.to_dataframe()
# #     return df

# # # Load and clean data
# # df = get_data()
# # df = df.drop(columns=['Unnamed: 0', 'SteamSpyPlayersEstimate', 'Month_sin', 'Month_cos', 'YearDifference', 'Free To Play', 'HDR available', 'Sales'])
# # df = clean_data(df)

# # # Define features
# # feature_cols_for_ReviewScore_rating = ['SteamSpyOwners','DeveloperCount', 'PublisherCount', 'CategorySinglePlayer', 'Achievements', 'Year',
# #                 'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
# #                 'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
# #                 'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
# #                 'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
# #                 'Multi-player', 'VR Support', 'Price', 'dlc_count', 'age_ranking']
# # feature_cols_for_SteamSpyOwners = ['DeveloperCount', 'PublisherCount', 'CategorySinglePlayer', 'Achievements', 'Year',
# #                 'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
# #                 'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
# #                 'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
# #                 'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
# #                 'Multi-player', 'VR Support', 'Price', 'dlc_count', 'age_ranking']

# # # Prepare XGBoost models for ReviewScore, rating, and SteamSpyOwners
# # targets = ['ReviewScore', 'rating']

# # # Dictionary to hold models and scalers
# # models = {}
# # scalers = {}

# # # XGBoost hyperparameter tuning with RandomizedSearchCV
# # param_dist = {
# #     'n_estimators': randint(100, 1000),
# #     'max_depth': randint(3, 20),
# #     'learning_rate': [0.01, 0.05, 0.1, 0.3],
# #     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
# #     'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
# #     'gamma': randint(0, 5)
# # }

# # # Training and evaluation for each target
# # for target in targets:
# #     print(f"Training XGBoost model for {target}...")

# #     X = df[feature_cols_for_ReviewScore_rating]
# #     y = df[target]

# #     # Train-Test Split
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Scale the features
# #     scaler = MinMaxScaler()
# #     X_train_scaled = scaler.fit_transform(X_train)
# #     X_test_scaled = scaler.transform(X_test)

# #     # Initialize XGBRegressor
# #     xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)

# #     # RandomizedSearchCV for hyperparameter tuning
# #     # random_search = RandomizedSearchCV(
# #     #     xgb_model, param_distributions=param_dist,
# #     #     n_iter=1, cv=1, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
# #     # )
# #     #xgb_model.fit(X_train_scaled, y_train)

# #     # Best model after tuning
# #     best_xgb_model = xgb_model.fit(X_train_scaled, y_train)

# #     # Save the best model
# #     model_path = os.path.abspath(f'project/app/{target}_xgb_model.pkl')
# #     joblib.dump(best_xgb_model, model_path)
# #     print(f"Model saved as {model_path}")

# #     # Evaluate the model on test data
# #     y_pred_train = best_xgb_model.predict(X_train_scaled)
# #     y_pred_test = best_xgb_model.predict(X_test_scaled)

# #     # Performance metrics
# #     def print_evaluation_metrics(y_true, y_pred, dataset_type="Test"):
# #         print(f"--- {dataset_type} Set Performance for {target} ---")
# #         print(f"R²: {r2_score(y_true, y_pred):.4f}")
# #         print(f"MAE: {mean_absolute_error(y_true, y_pred):,.2f}")
# #         print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):,.2f}")

# #     # Training set evaluation
# #     print_evaluation_metrics(y_train, y_pred_train, dataset_type="Train")

# #     # Test set evaluation
# #     print_evaluation_metrics(y_test, y_pred_test, dataset_type="Test")

# #     # Store the model and scaler
# #     models[target] = best_xgb_model
# #     scalers[target] = scaler

# # # Repeat for SteamSpyOwners target
# # targets_S = ['SteamSpyOwners']

# # for target in targets_S:
# #     print(f"Training XGBoost model for {target}...")

# #     X = df[feature_cols_for_SteamSpyOwners]
# #     y = df[target]

# #     # Train-Test Split
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Scale the features
# #     scaler = MinMaxScaler()
# #     X_train_scaled = scaler.fit_transform(X_train)
# # #     X_test_scaled = scaler.transform(X_test)

# # #     # Initialize XGBRegressor
# # #     xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)

# # #     # RandomizedSearchCV for hyperparameter tuning
# # #     random_search = RandomizedSearchCV(
# # #         xgb_model, param_distributions=param_dist,
# # #         n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
# # #     )
# # #     random_search.fit(X_train_scaled, y_train)

# # #     # Best model after tuning
# # #     best_xgb_model = random_search.best_estimator_

# # #     # Save the best model
# # #     model_path = os.path.abspath(f'project/app/{target}_xgb_model.pkl')
# # # #     joblib.dump(best_xgb_model, model_path)
# # # #     print(f"Model saved as {model_path}")

# # # #     # Evaluate the model on test data
# # # #     y_pred_train = best_xgb_model.predict(X_train_scaled)
# # # #     y_pred_test = best_xgb_model.predict(X_test_scaled)

# # # #     # Performance metrics
# # # #     def print_evaluation_metrics(y_true, y_pred, dataset_type="Test"):
# # # #         print(f"--- {dataset_type} Set Performance for {target} ---")
# # # #         print(f"R²: {r2_score(y_true, y_pred):.4f}")
# # # #         print(f"MAE: {mean_absolute_error(y_true, y_pred):,.2f}")
# # # #         print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):,.2f}")

# # # #     # Training set evaluation
# # # #     print_evaluation_metrics(y_train, y_pred_train, dataset_type="Train")

# # # #     # Test set evaluation
# # # #     print_evaluation_metrics(y_test, y_pred_test, dataset_type="Test")

# # # #     # Store the model and scaler
# # # #     models[target] = best_xgb_model
# # # #     scalers[target] = scaler


# import pandas as pd
# from google.cloud import bigquery
# from project.params import *
# import os
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import RobustScaler
# import joblib
# import xgboost as xgb

# # Function to clean data
# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.drop_duplicates()
#     df = df.dropna(how='any', axis=0)
#     return df

# def get_data(source=True) -> pd.DataFrame:
#     if source:
#         file_path = os.path.abspath('data/Games_with_ranking.csv')
#         return pd.read_csv(file_path)
#     else:
#         client = bigquery.Client(project=GCP_PROJECT)
#         query = "SELECT * FROM your_dataset.your_table"  # Specify your query here
#         query_job = client.query(query)
#         result = query_job.result()
#         df = result.to_dataframe()
#     return df

# # Load and clean data
# df = get_data()
# df = df.drop(columns=['QueryName','SteamSpyPlayersEstimate', 'YearDifference', 'Free To Play', 'HDR available', 'Sales'])
# df = clean_data(df)

# # Define features

# feature_cols_for_SteamSpyOwners = ['DeveloperCount', 'PublisherCount', 'CategorySinglePlayer', 'Achievements', 'Year',
#                 'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
#                 'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
#                 'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
#                 'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
#                 'Multi-player', 'VR Support', 'Price', 'dlc_count', 'age_ranking'
#                 ,'Budget_AA','Budget_AAA','Budget_Indie']

# # Prepare models for ReviewScore, rating, and SteamSpyOwners
# target = 'SteamSpyOwners'

# # Dictionary to hold models and scalers
# models = {}
# scalers = {}

# #for target in targets:
# print(f"Training model for {target}...")

# X = df[feature_cols_for_SteamSpyOwners]
# y = df[target]

# # Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# scaler = RobustScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Initialize XGBoost model
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=4)

# # Hyperparameter grid for GridSearchCV
# param_grid = {
#     'n_estimators': [50,100, 200],
#     'max_depth': [3, 6, 9, 12],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'gamma': [0, 0.1, 0.2, 0.3],
#     'reg_lambda': [1, 1.5, 2],
#     'reg_alpha': [0, 0.1, 0.5, 1]
# }

# # GridSearchCV with 5-fold cross-validation
# grid_search = GridSearchCV(
#     xgb_model, param_grid=param_grid,
#     cv=5, scoring='neg_mean_squared_error', n_jobs=-1
# )
# grid_search.fit(X_train_scaled, y_train)

# # Best model after tuning
# best_xgb_model = grid_search.best_estimator_

# # Save the best model
# model_path = os.path.abspath(f'project/app/{target}_xgb_model_l.pkl')
# joblib.dump(best_xgb_model, model_path)
# print(f"Model saved as {model_path}")

# # Evaluate the model on test data
# y_pred_train = best_xgb_model.predict(X_train_scaled)
# y_pred_test = best_xgb_model.predict(X_test_scaled)

# # Performance metrics
# def print_evaluation_metrics(y_true, y_pred, dataset_type="Test"):
#     print(f"--- {dataset_type} Set Performance for {target} ---")
#     print(f"R²: {r2_score(y_true, y_pred):.4f}")
#     print(f"MAE: {mean_absolute_error(y_true, y_pred):,.2f}")
#     print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):,.2f}")

# # Training set evaluation
# print_evaluation_metrics(y_train, y_pred_train, dataset_type="Train")

# # Test set evaluation
# print_evaluation_metrics(y_test, y_pred_test, dataset_type="Test")

# # Cross-Validation Score using `cross_val_score`
# cv_scores = cross_val_score(best_xgb_model, scaler.transform(X), y, cv=5, scoring='r2', n_jobs=-1)
# print(f"Cross-Validation R² scores: {cv_scores}")
# print(f"Mean Cross-Validation R² for {target}: {cv_scores.mean():.4f}")

# # Store the model and scaler
# models[target] = best_xgb_model
# scalers[target] = scaler


# # import pandas as pd
# # import xgboost as xgb
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.metrics import mean_squared_error
# # from sklearn.preprocessing import RobustScaler
# # import joblib

# # # Load and clean data (adjust the file path as needed)
# # df = pd.read_csv('data/GamesFinishDeletedDeveloperCountZero.csv')

# # # Drop unnecessary columns
# # df = df.drop(columns=['Unnamed: 0', 'SteamSpyPlayersEstimate', 'Month_sin', 'Month_cos', 'YearDifference', 'Free To Play', 'HDR available', 'Sales'])

# # # Features for SteamSpyOwners prediction
# # feature_cols_for_SteamSpyOwners = ['DeveloperCount', 'PublisherCount', 'CategorySinglePlayer', 'Achievements', 'Year',
# #                                    'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
# #                                    'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
# #                                    'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
# #                                    'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
# #                                    'Multi-player', 'VR Support', 'Price', 'dlc_count', 'age_ranking']

# # # Target for prediction
# # target = 'SteamSpyOwners'

# # # Splitting the data
# # X = df[feature_cols_for_SteamSpyOwners]
# # y = df[target]

# # # Train-test split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Scale the features
# # scaler = RobustScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)

# # # Initialize XGBoost model
# # xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# # # Reduced parameter grid for GridSearch
# # param_grid = {
# #     'n_estimators': [100, 200],
# #     'max_depth': [3, 5, 7],
# #     'learning_rate': [0.01, 0.1],
# #     'subsample': [0.7, 1.0],
# # }

# # # GridSearchCV for hyperparameter tuning
# # grid_search = GridSearchCV(
# #     estimator=xgb_model,
# #     param_grid=param_grid,
# #     scoring='neg_mean_squared_error',
# #     cv=3,
# #     verbose=1,
# #     n_jobs=-1
# # )

# # # Fit the model
# # grid_search.fit(X_train_scaled, y_train)

# # # Best model after tuning
# # best_xgb_model = grid_search.best_estimator_

# # # Save the model
# # model_path = 'project/app/SteamSpyOwners_xgb_model.pkl'
# # joblib.dump(best_xgb_model, model_path)
# # print(f"Model saved as {model_path}")

# # # Evaluate the model
# # y_pred_train = best_xgb_model.predict(X_train_scaled)
# # y_pred_test = best_xgb_model.predict(X_test_scaled)

# # # Calculate RMSE for train and test sets
# # rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
# # rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

# # # Print the results
# # print(f"Train RMSE: {rmse_train:.2f}")
# # print(f"Test RMSE: {rmse_test:.2f}")

# # # Save the scaler for use in the Streamlit app
# # # scaler_path = 'project/app/scaler.pkl'
# # # joblib.dump(scaler, scaler_path)
# # # print(f"Scaler saved as {scaler_path}")


# import pandas as pd
# from google.cloud import bigquery
# from project.params import *
# import os
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import RobustScaler
# import joblib
# import xgboost as xgb

# # Function to clean data
# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.drop_duplicates()
#     df = df.dropna(how='any', axis=0)
#     return df

# def get_data(source=True) -> pd.DataFrame:
#     if source:
#         file_path = os.path.abspath('data/Games_with_ranking.csv')
#         return pd.read_csv(file_path)
#     else:
#         client = bigquery.Client(project=GCP_PROJECT)
#         query = "SELECT * FROM your_dataset.your_table"  # Specify your query here
#         query_job = client.query(query)
#         result = query_job.result()
#         df = result.to_dataframe()
#     return df

# # Load and clean data
# df = get_data()
# df = df.drop(columns=['QueryName','SteamSpyPlayersEstimate', 'YearDifference', 'Free To Play', 'HDR available', 'Sales'])
# df = clean_data(df)

# # Define features

# feature_cols_for_SteamSpyOwners = ['DeveloperCount', 'PublisherCount', 'CategorySinglePlayer', 'Achievements', 'Year',
#                 'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
#                 'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
#                 'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
#                 'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
#                 'Multi-player', 'VR Support', 'Price', 'dlc_count', 'age_ranking'
#                 ,'Budget_AA','Budget_AAA','Budget_Indie']


# # Target for prediction
# target = 'SteamSpyOwners'

# # Splitting the data
# X = df[feature_cols_for_SteamSpyOwners]
# y = df[target]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# scaler = RobustScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Initialize XGBoost model
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

# # Reduced parameter grid for GridSearch
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1],
#     'subsample': [0.7, 1.0],
# }

# # GridSearchCV for hyperparameter tuning
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     scoring='neg_mean_squared_error',
#     cv=3,
#     verbose=1,
#     n_jobs=-1
# )

# # Fit the model
# grid_search.fit(X_train_scaled, y_train)

# # Best model after tuning
# best_xgb_model = grid_search.best_estimator_

# # Save the model
# model_path = 'project/app/SteamSpyOwners_xgb_model.pkl'
# joblib.dump(best_xgb_model, model_path)
# print(f"Model saved as {model_path}")

# # Evaluate the model
# y_pred_train = best_xgb_model.predict(X_train_scaled)
# y_pred_test = best_xgb_model.predict(X_test_scaled)

# # Calculate performance metrics
# rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
# rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
# mae_test = mean_absolute_error(y_test, y_pred_test)
# r2_test = r2_score(y_test, y_pred_test)

# # Print the results
# print(f"Train RMSE: {rmse_train:.2f}")
# print(f"Test RMSE: {rmse_test:.2f}")
# print(f"Test MAE: {mae_test:.2f}")
# print(f"Test R²: {r2_test:.2f}")

# # Cross-Validation Scores
# cv_scores = cross_val_score(best_xgb_model, scaler.transform(X), y, cv=5, scoring='neg_mean_squared_error')
# cv_rmse_scores = (-cv_scores)**0.5  # Convert to RMSE

# print(f"Cross-Validation RMSE scores: {cv_rmse_scores}")
# print(f"Mean Cross-Validation RMSE: {cv_rmse_scores.mean():.2f}")

# # Save the scaler for use in the Streamlit app
# scaler_path = 'project/app/scaler.pkl'
# joblib.dump(scaler, scaler_path)
# print(f"Scaler saved as {scaler_path}")


import pandas as pd
from google.cloud import bigquery
from project.params import *
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import joblib
from sklearn.neighbors import KNeighborsRegressor

# Function to clean data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)
    return df

def get_data(source=True) -> pd.DataFrame:
    if source:
        file_path = os.path.abspath('data/Games_with_ranking.csv')

        return pd.read_csv(file_path)
    else:
        client = bigquery.Client(project=GCP_PROJECT)
        query = "SELECT * FROM your_dataset.your_table"  # Specify your query here
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()
    return df

# Load and clean data
df = get_data()
df = df.drop(columns=['QueryName', 'SteamSpyPlayersEstimate', 'YearDifference', 'Free To Play', 'HDR available', 'Sales'])
df = clean_data(df)

# Define features
feature_cols_for_SteamSpyOwners = ['DeveloperCount', 'PublisherCount', 'CategorySinglePlayer', 'Achievements', 'Year',
                'Month', 'Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy',
                'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Education',
                'Violent', 'Design & Illustration', 'Animation & Modeling', 'Co-op',
                'Cross-Platform Multiplayer', 'Family Sharing', 'In-App Purchases',
                'Multi-player', 'VR Support', 'Price', 'dlc_count', 'age_ranking',
                'Budget_AA','Budget_AAA','Budget_Indie']

# Target for prediction
target = 'SteamSpyOwners'

# Splitting the data
X = df[feature_cols_for_SteamSpyOwners]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize KNeighborsRegressor model
knn_model = KNeighborsRegressor()

# Parameter grid for GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Best model after tuning
best_knn_model = grid_search.best_estimator_

# Save the model
model_path = 'project/app/SteamSpyOwners_knn_model.pkl'
joblib.dump(best_knn_model, model_path)
print(f"Model saved as {model_path}")

# Evaluate the model
y_pred_train = best_knn_model.predict(X_train_scaled)
y_pred_test = best_knn_model.predict(X_test_scaled)

# Calculate performance metrics
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print the results
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Test MAE: {mae_test:.2f}")
print(f"Test R²: {r2_test:.2f}")

# Cross-Validation Scores
cv_scores = cross_val_score(best_knn_model, scaler.transform(X), y, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = (-cv_scores)**0.5  # Convert to RMSE

print(f"Cross-Validation RMSE scores: {cv_rmse_scores}")
print(f"Mean Cross-Validation RMSE: {cv_rmse_scores.mean():.2f}")

# Save the scaler for use in the Streamlit app
scaler_path = 'project/app/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved as {scaler_path}")
