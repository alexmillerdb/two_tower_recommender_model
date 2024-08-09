# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").first()['current_user()']
username

experiment_path = f'/Users/{username}/torchrec-instacart-two-tower-example'
 
# You will need these later
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
 
# Manually create the experiment so that you know the id and can send that to the worker nodes when you scale later.
experiment = mlflow.set_experiment(experiment_path)

# COMMAND ----------

from pyspark.sql import functions as F

out_features = 32

# get user embeddings dataframe
user_embeddings_table = f"{catalog}.{schema}.user_two_tower_embeddings_{out_features}"
user_embedding_sdf = spark.table(user_embeddings_table)

# get item embeddings dataframe
item_embeddings_table = f"{catalog}.{schema}.item_two_tower_embeddings_{out_features}"
item_embedding_sdf = spark.table(item_embeddings_table)

# get eval set from training set and join user embeddings
training_set = spark.table("training_set")

eval_set = training_set.filter("group = 'test'") \
  .join(user_embedding_sdf, on='user_id')

display(eval_set)

# COMMAND ----------

data = eval_set \
  .filter(F.col("label") == 1) \
  .groupby("user_id", 'embeddings') \
  .agg(F.collect_set("product_id").alias("product_id_list")) \
  .toPandas()

display(data)

user_ids_to_query = data['user_id'].tolist()
len(user_ids_to_query)

# COMMAND ----------

# def extract_columns(response, columns, suffix="_pred"):
#     # Extract column indices for the desired columns
#     column_indices = {col['name']: i for i, col in enumerate(response['manifest']['columns'])}

#     # Initialize the dictionary with an empty list for each desired column with the suffix
#     result = {f"{col}{suffix}": [] for col in columns}

#     # Iterate through the data_array and populate the result dictionary
#     for row in response['result']['data_array']:
#         for col in columns:
#             if f"{col}{suffix}" not in result:
#                 result[f"{col}{suffix}"] = []
#             result[f"{col}{suffix}"].append(row[column_indices[col]])

#     # No need to wrap the lists in another list
#     return result

# from databricks.vector_search.client import VectorSearchClient

# vsc = VectorSearchClient()
# vector_search_endpoint_name = "one-env-shared-endpoint-0"

# # Vector index
# vs_index = f"item_two_tower_embeddings_index_{out_features}"
# vs_index_fullname = f"{catalog}.{schema}.{vs_index}"
# index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vs_index_fullname)

# # Search with a filter.
# user_ids_to_query = data['user_id'].tolist()[0:100]
# k = 5000

# results_dict = []

# for user_id in user_ids_to_query:
#   user_query_vector = data[data['user_id'] == user_id]['embeddings'].iloc[0].tolist()
#   results = index.similarity_search(
#     query_vector=user_query_vector,
#     columns=item_embedding_sdf.columns,
#     # filters={"department_id NOT": ("17")},
#     num_results=k)
#   extract_results = extract_columns(results, columns=['product_id', 'score'])
#   results_dict.append(extract_results)

# results_dict

# COMMAND ----------

import asyncio
import aiohttp
import nest_asyncio
from databricks.vector_search.client import VectorSearchClient
from concurrent.futures import ThreadPoolExecutor

nest_asyncio.apply()

def extract_columns(response, columns, suffix="_pred"):
    column_indices = {col['name']: i for i, col in enumerate(response['manifest']['columns'])}
    result = {f"{col}{suffix}": [] for col in columns}
    for row in response['result']['data_array']:
        for col in columns:
            result[f"{col}{suffix}"].append(row[column_indices[col]])
    return result

vsc = VectorSearchClient()
vector_search_endpoint_name = "one-env-shared-endpoint-0"
vs_index = f"item_two_tower_embeddings_index_{out_features}"
vs_index_fullname = f"{catalog}.{schema}.{vs_index}"
index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vs_index_fullname)

user_ids_to_query = data['user_id'].tolist()[0:10000]
k = 100

def sync_similarity_search(user_id):
    user_query_vector = data[data['user_id'] == user_id]['embeddings'].iloc[0].tolist()
    results = index.similarity_search(
        query_vector=user_query_vector,
        columns=item_embedding_sdf.columns,
        num_results=k
    )
    return extract_columns(results, columns=['product_id', 'score'])

async def run_searches():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, sync_similarity_search, user_id) 
                 for user_id in user_ids_to_query]
        results_dict = await asyncio.gather(*tasks)
    return results_dict

# Use this if you're in a Jupyter notebook or similar environment
loop = asyncio.get_event_loop()
results_dict = loop.run_until_complete(run_searches())

print(results_dict)

# COMMAND ----------

import pandas as pd

data_filtered = data.iloc[0:10000]
results_df = pd.DataFrame(results_dict)
eval_df = pd.concat([data_filtered, results_df], axis=1)

# Ensure the product_id_list and product_id_pred columns are arrays of integers
eval_df['product_id_list'] = eval_df['product_id_list'].apply(lambda x: list(map(int, x)))
eval_df['product_id_pred'] = eval_df['product_id_pred'].apply(lambda x: list(map(int, x)))
eval_df

# COMMAND ----------

# import mlflow

# # Case 1: Evaluating a static evaluation dataset
# with mlflow.start_run() as run:
#     evaluate_results = mlflow.evaluate(
#         data=eval_df,
#         model_type="retriever",
#         targets="product_id_list",
#         predictions="product_id_pred",
#         evaluators="default",
#         evaluator_config={"retriever_k": k}
#     )


# COMMAND ----------

import mlflow

def get_latest_run_id(experiment):
    latest_run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1).iloc[0]
    return latest_run.run_id

# COMMAND ----------

import mlflow

# Replace 'your_existing_run_id' with the actual run_id of the existing run
last_run_id = get_latest_run_id(experiment)

# Case 1: Evaluating a static evaluation dataset
with mlflow.start_run(run_id=last_run_id) as run:
    evaluate_results = mlflow.evaluate(
        data=eval_df,
        model_type="retriever",
        targets="product_id_list",
        predictions="product_id_pred",
        evaluators="default",
        evaluator_config={"retriever_k": k}
    )

    # Optionally, you can log the evaluation results to the run
    mlflow.log_metrics(evaluate_results.metrics)
    # mlflow.log_params(evaluate_results.params)
    # If there are any artifacts, you can log them as well
    # for artifact in evaluate_results.artifacts:
    #     mlflow.log_artifact(artifact)

# COMMAND ----------

evaluate_results.metrics

# COMMAND ----------

metric_to_sort = f"recall_at_{k}/score"
display(evaluate_results.tables['eval_results_table'].sort_values(metric_to_sort, ascending=False)[['user_id', 'product_id_list', 'product_id_pred', metric_to_sort]])

# COMMAND ----------

from pyspark.sql import functions as F

display(training_set.groupby("product_id").count().sort(F.desc("count")).filter(F.col("product_id")==38998))

# COMMAND ----------

# MAGIC %md ### To Do:
# MAGIC - Add user/item embeddings to user/product dataframe and check embedding values

# COMMAND ----------

user_sim_df = 
