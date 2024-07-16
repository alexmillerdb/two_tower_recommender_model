# Databricks notebook source
# MAGIC %pip install -q databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow

mlflow.__version__

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

# get user embeddings dataframe
user_embeddings_table = f"{catalog}.{schema}.user_two_tower_embeddings"
user_embedding_sdf = spark.table(user_embeddings_table)

# get item embeddings dataframe
item_embeddings_table = f"{catalog}.{schema}.item_two_tower_embeddings"
item_embedding_sdf = spark.table(item_embeddings_table)

# get eval set from training set and join user embeddings
training_set = spark.table("training_set")
eval_set = training_set.filter("group = 'test'") \
  .join(user_embedding_sdf, on='user_id')
  
display(eval_set)

# COMMAND ----------

# data = eval_set.filter(F.col("user_id").isin([7])) \
#   .filter(F.col("label") == 1) \
#   .groupby("user_id", 'embeddings') \
#   .agg(F.collect_set("product_id").alias("product_id_list")) \
#   .toPandas()

# display(data)
data = eval_set \
  .filter(F.col("label") == 1) \
  .groupby("user_id", 'embeddings') \
  .agg(F.collect_set("product_id").alias("product_id_list")) \
  .toPandas()

display(data)

user_ids_to_query = data['user_id'].tolist()
len(user_ids_to_query)

# COMMAND ----------

def extract_columns(response, columns, suffix="_pred"):
    # Extract column indices for the desired columns
    column_indices = {col['name']: i for i, col in enumerate(response['manifest']['columns'])}

    # Initialize the dictionary with an empty list for each desired column with the suffix
    result = {f"{col}{suffix}": [] for col in columns}

    # Iterate through the data_array and populate the result dictionary
    for row in response['result']['data_array']:
        for col in columns:
            if f"{col}{suffix}" not in result:
                result[f"{col}{suffix}"] = []
            result[f"{col}{suffix}"].append(row[column_indices[col]])

    # No need to wrap the lists in another list
    return result

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
vector_search_endpoint_name = "one-env-shared-endpoint-0"

# Vector index
vs_index = "item_two_tower_embeddings_index"
vs_index_fullname = f"{catalog}.{schema}.{vs_index}"
index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vs_index_fullname)

# Search with a filter.
user_ids_to_query = data['user_id'].tolist()[0:100]
k = 100

results_dict = []

for user_id in user_ids_to_query:
  user_query_vector = data[data['user_id'] == user_id]['embeddings'].iloc[0].tolist()
  results = index.similarity_search(
    query_vector=user_query_vector,
    columns=item_embedding_sdf.columns,
    # filters={"department_id NOT": ("17")},
    num_results=k)
  extract_results = extract_columns(results, columns=['product_id', 'score'])
  results_dict.append(extract_results)

results_dict

# COMMAND ----------

# import pandas as pd

# results_df = pd.DataFrame(results_dict)
# results_df

# COMMAND ----------

import pandas as pd

data_filtered = data.iloc[0:100]
results_df = pd.DataFrame(results_dict)
eval_df = pd.concat([data_filtered, results_df], axis=1)

# Ensure the product_id_list and product_id_pred columns are arrays of integers
eval_df['product_id_list'] = eval_df['product_id_list'].apply(lambda x: list(map(int, x)))
eval_df['product_id_pred'] = eval_df['product_id_pred'].apply(lambda x: list(map(int, x)))
eval_df

# COMMAND ----------

import mlflow

# Case 1: Evaluating a static evaluation dataset
with mlflow.start_run() as run:
    evaluate_results = mlflow.evaluate(
        data=eval_df,
        model_type="retriever",
        targets="product_id_list",
        predictions="product_id_pred",
        evaluators="default",
        evaluator_config={"retriever_k": 100}
    )


# COMMAND ----------

evaluate_results.metrics

# COMMAND ----------

user_sim_df = 
