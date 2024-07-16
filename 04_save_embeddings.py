# Databricks notebook source
# %pip install -r ./requirements.txt
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -q --upgrade --no-deps --force-reinstall torch torchvision fbgemm-gpu torchrec --index-url https://download.pytorch.org/whl/cpu
# MAGIC %pip install -q torchmetrics==1.0.3 iopath pyre_extensions mosaicml-streaming==0.7.5 databricks-vectorsearch
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

import pandas as pd
import numpy as np
import os
from typing import List, Optional, Tuple
from mlflow import MlflowClient

import torch
from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.modules.mlp import MLP
from torchrec.datasets.utils import Batch

class TwoTower(nn.Module):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        layer_sizes: List[int],
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__()

        assert len(embedding_bag_collection.embedding_bag_configs()) == 2, "Expected two EmbeddingBags in the two tower model"
        assert embedding_bag_collection.embedding_bag_configs()[0].embedding_dim == embedding_bag_collection.embedding_bag_configs()[1].embedding_dim, "Both EmbeddingBagConfigs must have the same dimension"

        embedding_dim = embedding_bag_collection.embedding_bag_configs()[0].embedding_dim
        self._feature_names_query: List[str] = embedding_bag_collection.embedding_bag_configs()[0].feature_names
        self._candidate_feature_names: List[str] = embedding_bag_collection.embedding_bag_configs()[1].feature_names
        self.ebc = embedding_bag_collection
        self.query_proj = MLP(in_size=embedding_dim, layer_sizes=layer_sizes, device=device)
        self.candidate_proj = MLP(in_size=embedding_dim, layer_sizes=layer_sizes, device=device)

    def forward(self, kjt: KeyedJaggedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert the sparse input features into dense features
        # Input: KeyedJaggedTensor (kjt) is passed through the embedding bag collection to get the pooled embeddings
        pooled_embeddings = self.ebc(kjt)
        # Pooled embeddings for query features are concatenated along the feature dimension
        # Concatenated embeddings are passed through the query_proj MLP to produce final query embedding
        query_embedding: torch.Tensor = self.query_proj(
            torch.cat(
                [pooled_embeddings[feature] for feature in self._feature_names_query],
                dim=1,
            )
        )
        # Pooled embeddings for candidate features are concatenated along the feature dimension
        # Concatenated embeddings are passed through the candidate_proj MLP to produce final candidate embedding
        candidate_embedding: torch.Tensor = self.candidate_proj(
            torch.cat(
                [
                    pooled_embeddings[feature]
                    for feature in self._candidate_feature_names
                ],
                dim=1,
            )
        )
        return query_embedding, candidate_embedding

# Model architecture
class TwoTowerTrainTask(nn.Module):
    def __init__(self, two_tower: TwoTower) -> None:
        super().__init__()
        self.two_tower = two_tower
        # The Two Tower model uses the Binary Cross Entropy loss (you can update it as needed for your own use case and dataset)
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # batch.spare_features is KeyedJaggedTensor that is passed through TwoTower model
        # TwoTower model returns embeddings for query and candidate
        query_embedding, candidate_embedding = self.two_tower(batch.sparse_features)
        # Dot product between query and candidate embeddings is computed
        logits = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
        # # clamp the outputs here to be between 0 and 1
        # clamped_logits = torch.clamp(logits, min=0, max=1)
        # loss = self.loss_fn(clamped_logits, batch.labels.float())
        loss = self.loss_fn(logits, batch.labels.float())

        # return loss, (loss.detach(), clamped_logits.detach(), batch.labels.detach())
        return loss, (loss.detach(), logits.detach(), batch.labels.detach())

def get_latest_run_id(experiment):
    latest_run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1).iloc[0]
    return latest_run.run_id

def get_latest_artifact_path(run_id):
    client = MlflowClient()
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    artifact_paths = [i.path for i in client.list_artifacts(run_id) if "base" not in i.path]
    return artifact_paths[-1]

def get_mlflow_model(run_id, artifact_path="model_state_dict", device="cuda"):
    from mlflow import MlflowClient

    device = torch.device(device)
    run = mlflow.get_run(run_id)
    
    cat_cols = eval(run.data.params.get('cat_cols'))
    emb_counts = eval(run.data.params.get('emb_counts'))
    layer_sizes = eval(run.data.params.get('layer_sizes'))
    embedding_dim = eval(run.data.params.get('embedding_dim'))

    MlflowClient().download_artifacts(run_id, f"{artifact_path}/state_dict.pth", "/databricks/driver")
    state_dict = mlflow.pytorch.load_state_dict(f"/databricks/driver/{artifact_path}", map_location=device)
    
    # Remove the prefix "two_tower." from all of the keys in the state_dict
    state_dict = {k[10:]: v for k, v in state_dict.items()}

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=embedding_dim,
            num_embeddings=emb_counts[feature_idx],
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(cat_cols)
    ]

    embedding_bag_collection = EmbeddingBagCollection(
        tables=eb_configs,
        device=device,
    )
    two_tower_model = TwoTower(
        embedding_bag_collection=embedding_bag_collection,
        layer_sizes=layer_sizes,
        device=device,
    )

    two_tower_model.load_state_dict(state_dict)

    return two_tower_model, embedding_bag_collection, eb_configs, cat_cols, emb_counts

# COMMAND ----------

# Loading the latest model state dict from the latest run of the current experiment
latest_run_id = get_latest_run_id(experiment)
latest_artifact_path = get_latest_artifact_path(latest_run_id)
two_tower_model, embedding_bag_collection, eb_configs, cat_cols, emb_counts = get_mlflow_model(
  latest_run_id, 
  artifact_path=latest_artifact_path, 
  device="cpu")

# COMMAND ----------

two_tower_model.to('cpu')
two_tower_model.eval()

# COMMAND ----------

print(two_tower_model.ebc)

# COMMAND ----------

from pyspark.sql import functions as F

training_set = spark.table("training_set")
display(training_set.agg(
  F.min("user_id").alias("min_user_id"),
  F.max("user_id").alias("max_user_id"),
  F.min("product_id").alias("min_product_id"),
  F.max("product_id").alias("max_product_id")))

# COMMAND ----------

# MAGIC %md ### Example for getting embeddings:
# MAGIC - https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L198

# COMMAND ----------

def create_keyed_jagged_tensor(num_embeddings, cat_cols, key):

    values = torch.tensor(list(range(num_embeddings)), device=torch.device("cpu"))

    if key == 'product_id':
      lengths = torch.tensor(
            [0] * num_embeddings + [1] * num_embeddings,
            device=torch.device("cpu")
        )
    elif key == 'user_id':
      lengths = torch.tensor(
            [1] * num_embeddings + [0] * num_embeddings,
            device=torch.device("cpu")
        )

    # Create the KeyedJaggedTensor
    kjt = KeyedJaggedTensor(
        keys=cat_cols,
        values=values,
        lengths=lengths
    )

    print("KJT structure:")
    print(f"Keys: {kjt.keys()}")
    print(f"Values shape: {kjt.values().shape}")
    print(f"Lengths shape: {kjt.lengths().shape}")
    print(f"Length per key: {kjt.length_per_key()}")

    return kjt

# COMMAND ----------

def process_embeddings(two_tower_model, kjt, lookup_column):
    """
    Passes the KeyedJaggedTensor (KJT) through the EmbeddingBagCollection (EBC) to get all embeddings.

    Parameters:
    - two_tower_model: The model containing the ebc and projection methods.
    - kjt: The KeyedJaggedTensor to be processed.

    Returns:
    - item_embeddings: The embeddings for the items.
    """
    try:
        with torch.no_grad():
            lookups = two_tower_model.ebc(kjt)
            embeddings = two_tower_model.candidate_proj(lookups[lookup_column])
        
        print("Successfully processed embeddings")
        print(f"{lookup_column} embeddings shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# COMMAND ----------

product_kjt = create_keyed_jagged_tensor(emb_counts[1], cat_cols, 'product_id')
item_embeddings = process_embeddings(two_tower_model, product_kjt, 'product_id')
item_embeddings

# COMMAND ----------

# Convert KJT to dictionary
product_kjt_dict = product_kjt.to_dict()

# Get product_id values
product_id_values = product_kjt_dict['product_id'].values()

print("Product IDs:", product_id_values)

# Convert tensor to numpy array
item_embedding_array = item_embeddings.numpy()

# Create pandas DataFrame with arrays and product ids
item_embedding_df = pd.DataFrame({'embeddings': [row for row in item_embedding_array]})
item_embedding_df['product_id'] = product_id_values + 1

# Display the first few rows of the DataFrame
print(item_embedding_df.head())

# COMMAND ----------

aisles = spark.table('aisles')
departments = spark.table("departments")
products = spark.table("products")

item_embedding_sdf = spark.createDataFrame(item_embedding_df) \
  .join(products, on='product_id') \
  .join(aisles, on='aisle_id') \
  .join(departments, on='department_id')
  
display(item_embedding_sdf)

# COMMAND ----------

item_embeddings_table = f"{catalog}.{schema}.item_two_tower_embeddings"
item_embedding_sdf.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(item_embeddings_table)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
vector_search_endpoint_name = "one-env-shared-endpoint-0"

# Vector index
vs_index = "item_two_tower_embeddings_index"
vs_index_fullname = f"{catalog}.{schema}.{vs_index}"
try:
  index = vsc.get_index(vector_search_endpoint_name, vs_index_fullname)
  index.sync()
except:
  index = vsc.create_delta_sync_index_and_wait(
    endpoint_name=vector_search_endpoint_name,
    source_table_name=item_embeddings_table,
    index_name=vs_index_fullname,
    pipeline_type='TRIGGERED',
    primary_key="product_id",
    embedding_vector_column="embeddings",
    embedding_dimension=64
  )

index.describe()

# COMMAND ----------

user_kjt = create_keyed_jagged_tensor(emb_counts[0], cat_cols, 'user_id')
user_embeddings = process_embeddings(two_tower_model, user_kjt, 'user_id')

# Convert KJT to dictionary
user_kjt_dict = user_kjt.to_dict()

# Get product_id values
user_id_values = user_kjt_dict['user_id'].values()

print("User IDs:", user_id_values)

# Convert tensor to numpy array
user_embedding_array = user_embeddings.numpy()

# Create pandas DataFrame
user_embedding_df = pd.DataFrame({'embeddings': [row for row in user_embedding_array]})
user_embedding_df['user_id'] = user_id_values + 1

# Display the first few rows of the DataFrame
print(user_embedding_df.head())

# COMMAND ----------

user_embeddings_table = f"{catalog}.{schema}.user_two_tower_embeddings"
user_embedding_sdf = spark.createDataFrame(user_embedding_df)
user_embedding_sdf.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(user_embeddings_table)
display(user_embedding_sdf)

# COMMAND ----------

all_columns = item_embedding_sdf.columns
user_query_vector = user_embedding_df.iloc[0]['embeddings'].tolist()

# Search with a filter.
index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vs_index_fullname)
results = index.similarity_search(
  query_vector=user_query_vector,
  columns=all_columns,
  filters={"department_id NOT": ("17")},
  num_results=10)

results
