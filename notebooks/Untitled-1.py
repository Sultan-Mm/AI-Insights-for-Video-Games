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
df_games.columns

# %%
df_games= df_games.drop(columns=[
    'AppID', 'QueryName',
    'Unnamed: 11', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 23', 'Unnamed: 25',
    'Developers', 'developers',  # Developer related columns
    'Publishers', 'publishers',  # Publisher related columns
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

# %%
df_games.columns

# %%
len(df_games)

# %%
df_games.iloc[:,20:].info()


# %% [markdown]
# # Data Cleaning

# %% [markdown]
# ## Zero replacement

# %%
zero_columns = ['Achievements', 'Price', 'dlc_count'] + list(df_games.iloc[:, 10:32].columns)

df_games[zero_columns] = df_games[zero_columns].fillna(0)

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
mean_feats = ['age_ranking', 'rating', 'ReviewScore']

df_games[mean_feats] = imp_mean.fit_transform(df_games[mean_feats])

df_games.head()

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

# %%
df_games.info()

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

# %% [markdown]
# ## Scaler

# %%
# Price
# dlc_count
# balance_pos_neg
# rating
# TotalReviews
# ReviewScore



scaling_feat = [
    'DeveloperCount',
    'RecommendationCount',
    'PublisherCount',
    'SteamSpyOwners',
    'SteamSpyPlayersEstimate',
    'Achievements',
    'Price',
    'dlc_count',
    'balance_pos_neg',
    'rating',
    'TotalReviews',
    'ReviewScore']

fig, axs = plt.subplots(4, 3, figsize=(15, 10))

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
    'SteamSpyOwners',
    'SteamSpyPlayersEstimate',
    'Achievements',
    'Price',
    'dlc_count',
    'balance_pos_neg',
    'TotalReviews']

std_feat = ['rating', 'ReviewScore']

rob = RobustScaler()

std = StandardScaler()

df_games[rob_feat] = rob.fit_transform(df_games[rob_feat])
df_games[std_feat] = std.fit_transform(df_games[std_feat])

# %%
df_games.columns

# %%
#df_games[['num_reviews_total', 'TotalReviews']]

# %%
df_games

# %%
from sklearn.preprocessing import MinMaxScaler


# Step 3: Scaling for 'positive' and 'negative'
scaler = MinMaxScaler()
df_games[['positive', 'negative']] = scaler.fit_transform(df_games[['positive', 'negative']])

# Step 4: Fill missing values for 'rating', 'TotalReviews', and 'ReviewScore', then apply scaling
df_games['rating'] = df_games['rating'].fillna(df_games['rating'].mean())
df_games['TotalReviews'] = df_games['TotalReviews'].fillna(df_games['TotalReviews'].median())
df_games['ReviewScore'] = df_games['ReviewScore'].fillna(df_games['ReviewScore'].mean())

# Scaling for rating, TotalReviews, and ReviewScore
df_games[['rating', 'TotalReviews', 'ReviewScore']] = scaler.fit_transform(df_games[['rating', 'TotalReviews', 'ReviewScore']])

# Display a summary of the updated dataframe to confirm changes
df_games[['positive', 'negative', 'rating', 'TotalReviews', 'ReviewScore']].head()


# %%
df_games.iloc[:,:].info()

# %%
df_games.columns

# %%
#df_games.to_csv("dfGames.csv")

# %%
temp_df_games= df_games.copy()

# %%
df_bool = temp_df_games[['Adventure','Casual','Indie','RPG','Free To Play','Action','Strategy','Simulation',
    'Racing','Sports','Massively Multiplayer','Education','Violent','Design & Illustration',
    'Animation & Modeling','Co-op','Cross-Platform Multiplayer','Family Sharing','HDR available',
    'In-App Purchases','Multi-player','VR Support']]
to_cinv = df_bool.columns.to_list()
df_bool = temp_df_games[to_cinv].astype(bool)
temp_df_games[to_cinv] = df_bool

# %%
temp_df_games.info()

# %%
#df_games.to_csv("games_newest2.csv")

# %%


# %%


# %%
