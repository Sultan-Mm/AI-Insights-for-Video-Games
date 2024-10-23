# Importing Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, ElasticNetCV
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import os
file_path = os.path.abspath('data/GamesFinish.csv')
print(file_path)
# Load Dataset
df = pd.read_csv(file_path)

# Prepare Features and Target
X = df.drop(columns='SteamSpyOwners')
y = df['SteamSpyOwners']

# Baseline Model - Using the average of Sales
avg_owners = np.mean(y)
y_pred = np.array([avg_owners] * len(df))

print("Baseline Model R2 Score:", r2_score(y, y_pred))
print("Baseline Model MAE:", mean_absolute_error(y, y_pred))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Linear Regression
lin_reg = LinearRegression()
cv_results = cross_validate(lin_reg, X_train, y_train, cv=5, scoring=['r2', 'neg_mean_absolute_error'])
print("Linear Regression R2:", np.mean(cv_results['test_r2']))
print("Linear Regression MAE:", np.mean(cv_results['test_neg_mean_absolute_error']))

# KNeighbors Regressor
neigh = KNeighborsRegressor()
cv_results = cross_validate(neigh, X_train, y_train, cv=5, scoring=['r2', 'neg_mean_absolute_error'])
print("KNeighbors Regressor R2:", np.mean(cv_results['test_r2']))
print("KNeighbors Regressor MAE:", np.mean(cv_results['test_neg_mean_absolute_error']))

# Ridge Regression
ridge = Ridge()
cv_results = cross_validate(ridge, X_train, y_train, cv=5, scoring=['r2', 'neg_mean_absolute_error'])
print("Ridge Regression R2:", np.mean(cv_results['test_r2']))
print("Ridge Regression MAE:", np.mean(cv_results['test_neg_mean_absolute_error']))

# ElasticNetCV
elastic = ElasticNetCV(random_state=0)
cv_results = cross_validate(elastic, X_train, y_train, cv=5, scoring=['r2', 'neg_mean_absolute_error'])
print("ElasticNet R2:", np.mean(cv_results['test_r2']))
print("ElasticNet MAE:", np.mean(cv_results['test_neg_mean_absolute_error']))


file_path3 = os.path.abspath('data/GamesFinish_woScaling.csv')
print(file_path3)

# # Decision Trees - RandomForest and GradientBoosting
df_tree = pd.read_csv(file_path3)
print('8888888888888888888',df_tree.columns)
df_tree = df_tree[df_tree['YearDifference'] >= 10]
print(df_tree.shape)
X_tree = df_tree.drop(columns='SteamSpyOwners')
y_tree = df_tree['SteamSpyOwners']

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.3, random_state=42)
print(X_train_tree.columns)
# RandomForest Regressor
regr = RandomForestRegressor(max_depth=2, random_state=0)
cv_results = cross_validate(regr, X_train_tree, y_train_tree, cv=5, scoring=['r2', 'neg_mean_absolute_error'])
print("RandomForest Regressor R2:", np.mean(cv_results['test_r2']))
print("RandomForest Regressor MAE:", np.mean(cv_results['test_neg_mean_absolute_error']))

# GradientBoosting Regressor
gb_reg = GradientBoostingRegressor(random_state=0)
cv_results = cross_validate(gb_reg, X_train_tree, y_train_tree, cv=5, scoring=['r2', 'neg_mean_absolute_error'])
print("GradientBoosting Regressor R2:", np.mean(cv_results['test_r2']))
print("GradientBoosting Regressor MAE:", np.mean(cv_results['test_neg_mean_absolute_error']))

# GradientBoosting Regressor - Prediction
gb_reg.fit(X_train_tree, y_train_tree)
y_pred = gb_reg.predict(X_test_tree)

X_test_tree['Pred_Owners'] = y_pred
X_tree['Pred_Owners'] = gb_reg.predict(X_tree)

# Save the trained GradientBoostingRegressor model
model_path = 'project/app/gradient_boosting_model.pkl'
joblib.dump(gb_reg, model_path)
print(f"Model saved as {model_path}")

file_path2 = os.path.abspath('data/games_c5.csv')
print(file_path2)

# Merge predictions back with the main dataset
df_games = pd.read_csv(file_path2)
df_games_enriched = pd.concat([df_games, X_tree.iloc[:,-1]], axis=1)

# Popularity Score Calculation
df_games_enriched['Popularity_score'] = (
    0.3 * df_games_enriched['TotalReviews'] +
    0.3 * df_games_enriched['RecommendationCount'] +
    0.4 * df_games_enriched['Pred_Owners']
)

df_games_enriched = df_games_enriched.reset_index(drop=True)

# Show the most popular games
print(df_games_enriched[['QueryName', 'Popularity_score']].sort_values('Popularity_score', ascending=False))

# Games with negative popularity score
print(df_games_enriched[df_games_enriched['Popularity_score'] < 0]['QueryName'])

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Drop rows where 'Popularity_score' is NaN
df_games_enriched = df_games_enriched.dropna(subset=['Popularity_score'])

# Apply scaling to the 'Popularity_score' column
df_games_enriched['Popularity_score'] = scaler.fit_transform(df_games_enriched[['Popularity_score']])

# Print the updated DataFrame
df_games_enriched.to_csv('data/df_games_enriched_scaled.csv')

#print(df_games_enriched)


file_path2 = os.path.abspath('data/Games_with_ranking.csv')
print(file_path2)

# Merge predictions back with the main dataset
df_games_with_ranking = pd.read_csv(file_path2)

df_games_with_ranking = df_games_with_ranking[['AppID','Budget_AA','Budget_AAA','Budget_Indie']]
merged = pd.merge(df_games_with_ranking,df_games_enriched, how='inner', on='AppID')
print(merged.columns,merged.shape)
merged.to_csv('data/merged_df_games_with_ranking_df_games_enriched.csv')
