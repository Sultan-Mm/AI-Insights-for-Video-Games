# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

# %%
df_games = pd.read_csv('games_c4.csv')
df_games = df_games.iloc[:, 1:]
df_games.head()

# %%
df_Player = pd.read_csv('../data/avg_play.csv')
df_games = pd.read_csv('games_c4.csv')
df_games = pd.merge(df_games ,df_Player ,how='left',on='AppID')
df_games

# %% [markdown]
# Sid Meier's Civilization IV                      2
# Sid Meier's Civilization IV: Warlords            2
# Ultimate Arena                                   2
# Spellbind                                        2
# Taxi                                             2
# New York Bus Simulator                           2
# Streamline                                       2
# Dark Matter                                      2
# Snapshot                                         2
# Monday Night Combat                              2
# Darksiders                                       2
# Call of Duty: Modern Warfare 3                   2
# Total War: SHOGUN 2                              2
# Alpha Protocol                                   2
# Arma 2                                           2
# Fallout: New Vegas                               2
# Sid Meier's Civilization IV: Colonization        2
# Sid Meier's Civilization V                       2
# Sid Meier's Civilization IV: Beyond the Sword    2
# Rise                                             2
# Name: QueryName, dtype: int64

# %%
df_games = df_games.sort_values(
    ['SteamSpyOwners', 'RecommendationCount'], ascending=False).drop_duplicates(subset='QueryName', keep='first'
)

df_games = df_games[df_games['Sales'] > 0]

df_games.to_csv("games_c5.csv", index=False)

df_games= df_games.drop(columns=[
    'Unnamed: 11', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 23', 'Unnamed: 25',
    'AppID', 'QueryName',
    'Developers', 'developers',  # Developer related columns
    'publishers',  # Publisher related columns
    'PriceCurrency',  # Price related
    'Recommendations',  # Reviews and ratings
    'DLC count',  # Redundant DLC count
    'Categories',  # Redundant category
    'Single-player',  # Redundant category
    'GenreIsAction', 'GenreIsAdventure', 'GenreIsCasual',  # Specific genres
    'PlatformWindows', 'ShortDescrip', 'DetailedDescrip',
    'PCMinReqsText', 'Release date', 'Windows', 'Genres', 'Day',
    # 'Cross-Platform', 'Multiplayer',
    'Online PvP', 'PvP', 'VR Supported', 'about_the_game', 'packages', 'categories',
    'genres', 'tags', 'pct_pos_total', 'pct_pos_recent', 'num_reviews_recent', 'CustomRating',
    'Metacritic score',
    'num_reviews_total'
])


# %% [markdown]
# # Data Cleaning

# %% [markdown]
# ## Zero replacement

# %%
zero_columns = ['Achievements', 'Price', 'dlc_count', 'VR Support'] + list(df_games.iloc[:, 10:32].columns)

df_games[zero_columns] = df_games[zero_columns].fillna(0)

# %% [markdown]
# ## Not found

# %%
df_games['Publishers'] = df_games['Publishers'].fillna('Not Found')

# %% [markdown]
# ## Most Frequent

# %%
import numpy as np
from datetime import datetime

# Assuming df_games is the dataset, current year for calculating year difference
current_year = datetime.now().year

# Fill 'Year' with the most frequent year and calculate (current_year - Year)
most_frequent_year = df_games['Year'].mode()[0]
df_games['Year'] = df_games['Year'].fillna(most_frequent_year)
df_games['YearDifference'] = current_year - df_games['Year']

# Fill 'Month' with the most frequent month and create sin and cos transformations
most_frequent_month = df_games['Month'].mode()[0]
df_games['Month'] = df_games['Month'].fillna(most_frequent_month)

# Creating sin and cos transformations for cyclical month representation
df_games['Month_sin'] = np.sin(2 * np.pi * df_games['Month'] / 12)
df_games['Month_cos'] = np.cos(2 * np.pi * df_games['Month'] / 12)

df_games.drop(columns=['Year', 'Month'], inplace=True)


# %% [markdown]
# ## Average

# %%
imp_mean = SimpleImputer(strategy='mean')
mean_feats = ['age_ranking', 'rating', 'ReviewScore', 'avg_playtime']

df_games[mean_feats] = imp_mean.fit_transform(df_games[mean_feats])

# %% [markdown]
# ## Median

# %%
imp_median = SimpleImputer(strategy='median')

median_feats = ['TotalReviews', 'positive', 'negative']

df_games[median_feats] = imp_median.fit_transform(df_games[median_feats])

# %% [markdown]
# ## Balance (positive x negative)

# %%
df_games['balance_pos_neg'] = df_games['positive'] - df_games['negative']

df_games.drop(columns=['positive', 'negative'], inplace=True)

# %% [markdown]
# # Preprocessing

# %% [markdown]
# Replace the null values
#
# Achievements -> equal to 0 - ok
#
# Year -> most frequent and (This year - year) - ok
#
# Month -> most frequent and sin and cos - ok
#
# Columns 10:31 -> replace with zero - ok
#
# age_ranking: most frequent and OHE - ok
#
# Price -> equal to 0 and scaler - ok
#
# dlc_count -> equal to 0 and scaler - ok
#
# positive and negative -> balance between then and scaler
#
# rating -> average and scaler
#
# TotalReviews -> median and scaler
#
# ReviewScore -> average and scaler

# %% [markdown]
# ## One Hot Encoder

# %%
enc_bool = OneHotEncoder(handle_unknown='ignore', drop='if_binary')

enc_bool.fit(df_games[['PurchaseAvail', 'CategorySinglePlayer']])

encoded_df = pd.DataFrame(enc_bool.transform(
    df_games[['PurchaseAvail', 'CategorySinglePlayer']]).toarray(),
    columns=enc_bool.get_feature_names_out(['PurchaseAvail', 'CategorySinglePlayer'])
)

encoded_df.index = df_games.index

df_games = df_games.drop(['PurchaseAvail', 'CategorySinglePlayer'], axis=1)
df_games = pd.concat([df_games, encoded_df], axis=1)
df_games.rename(columns={'PurchaseAvail_True':'PurchaseAvail', 'CategorySinglePlayer_True':'CategorySinglePlayer'},
                inplace=True)

# %%
dd = pd.read_csv('../data/dd.csv')


# %% [markdown]
# add budget category

# %%
# Function to calculate points based on game features
def calculate_points(row):
    points = 0
    AAA_List_P = dd['Publishers'][0:47].to_list()
    AA_List_P = dd['Publishers'][47:300].to_list()
    indie_List_P = dd['Publishers'][300:].to_list()

    # ckeck if the game's publisher is in the lists
    if row['Publishers']  in AAA_List_P:
        points += 15
    elif row['Publishers'] in AA_List_P:
        points += 9
    else:
        points += 1
    # Price points
    if row['Price'] >= 30.00:
        points += 5
    elif 20 <= row['Price'] < 30.00:
        points += 3
    else:
        points += 1
    # TotalReviews points
    if row['TotalReviews'] > 30000:
        points += 5
    elif 5000 <= row['TotalReviews'] <= 30000:
        points += 3
    else:
        points += 1
    # DeveloperCount points
    if row['DeveloperCount'] >= 2:
        points += 3
    elif row['DeveloperCount'] == 1:
        points += 2
    else:
        points += 1
    # Achievements points
    if row['Achievements'] > 20:
        points += 3
    else:
        points += 1
    # avg_playtime points
    if row['avg_playtime'] > 1000:
        points += 3
    elif 100 <= row['avg_playtime'] <= 1000:
        points += 2
    else:
        points += 1
    return points
# Apply the function to calculate points for each game
df_games['TotalPoints'] = df_games.apply(calculate_points, axis=1)
# Define thresholds for BudgetCategory based on points
def categorize_by_points(row):
    if row['TotalPoints'] >= 19:
        return 'AAA'
    elif 12 <= row['TotalPoints'] < 19:
        return 'AA'
    else:
        return 'Indie'
# Apply the categorization
df_games['BudgetCategory'] = df_games.apply(categorize_by_points, axis=1)

# delete Publishers
df_games.drop(columns='Publishers', inplace=True)

# %%
# One-hot encoding the 'BudgetCategory' column
df_games = pd.get_dummies(df_games, columns=['BudgetCategory'], prefix='Budget')

# This will create three new columns: 'Budget_Indie', 'Budget_AA', and 'Budget_AAA',
# with binary values indicating the presence of each category


# %%
df_games.to_csv('GamesFinish_woScaling.csv', index=False)

# %% [markdown]
# ## Scaler

# %%
#    'SteamSpyPlayersEstimate',


# %%


# %%
scaling_feat = [
    'DeveloperCount',
    'RecommendationCount',
    'PublisherCount',
    'Achievements',
    'SteamSpyPlayersEstimate',
    'Price',
    'dlc_count',
    'balance_pos_neg',
    'rating',
    'TotalReviews',
    'ReviewScore',
    'avg_playtime',
    'Sales']

fig, axs = plt.subplots(5, 3, figsize=(15, 10))

axes = axs.flatten()

for ax, f in zip(axes, scaling_feat):
    sns.histplot(data=df_games, x=f, ax=ax, bins=50)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
# df_games[scaling_feat]

# %%
df_games[scaling_feat].describe()

# %%
rob_feat = ['DeveloperCount',
    'RecommendationCount',
    'PublisherCount',
    'SteamSpyPlayersEstimate',
    'Achievements',
    'Price',
    'dlc_count',
    'balance_pos_neg',
    'TotalReviews',
    'Sales',
    'avg_playtime']

std_feat = ['rating', 'ReviewScore']

rob = RobustScaler()

std = StandardScaler()

df_games[rob_feat] = rob.fit_transform(df_games[rob_feat])
df_games[std_feat] = std.fit_transform(df_games[std_feat])

# %% [markdown]
#  14  Sales                       10849 non-null  float64
#  15  avg_playtime

# %%
df_games.to_csv("GamesFinish.csv", index=False)
