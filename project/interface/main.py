import pandas as pd
import os











print(os.getcwd())
filepath = os.getcwd()+'/raw_data/steam_games.csv'
df = pd.read_csv(filepath, usecols=[0,1,2,3,4,5,6,7,8],dtype='unicode',low_memory=False)
print(df)
