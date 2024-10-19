import pandas as pd
from google.cloud import bigquery
from project.params import *
import os

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """

    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)

    return df

def get_data(source=True) -> pd.DataFrame:

    if source == True:
        return pd.read_csv(os.getcwd()+'/data/games_dataset.csv')
    else:
        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query()
        result = query_job.result()
        df = result.to_dataframe()
    pass


# for later use
# client = bigquery.Client()
# #
# print(client.get_table())
# #lwdaa-437406
# data = get_data()
# #full_table_name = f"{GCP_PROJECT}.{BQ_DATASET}.ai_games"
# full_table_name = "lwdaa-437406.ai-games.games_dataset"
# #data.to_gbq("ai-games.games_dataset", project_id="lwdaa-437406", if_exists="replace")
# print(full_table_name)
# job_config = bigquery.LoadJobConfig(
#     autodetect=True  # Enable schema autodetection
# )
# client.insert_rows_from_dataframe(dataframe=data,table=full_table_name,job_config=job_config)

# data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]
# truncate = True
# write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
# job_config = bigquery.LoadJobConfig(write_disposition=write_mode)
# job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
# print(get_data())
