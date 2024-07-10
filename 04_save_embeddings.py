# Databricks notebook source
# %pip install -r ./requirements.txt
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -q --upgrade --no-deps --force-reinstall torch torchvision fbgemm-gpu torchrec --index-url https://download.pytorch.org/whl/cpu
# MAGIC %pip install -q torchmetrics==1.0.3 iopath pyre_extensions mosaicml-streaming==0.7.5
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

# DBTITLE 1,Move to Config
cat_cols = ["user_id", "product_id"]

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

from mlflow import MlflowClient
import os
from typing import List, Optional, Tuple

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

pdf = spark.table("training_set").select(*cat_cols).toPandas()

# COMMAND ----------

# MAGIC %md ### Example for getting embeddings:
# MAGIC - https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L198

# COMMAND ----------

kjt_values: List[int] = []
kjt_lengths: List[int] = []
for col_idx, col_name in enumerate(cat_cols):
    values = pdf[col_name].unique()
    for value in values:
        if value:
            kjt_values.append(
                value % emb_counts[col_idx]
            )
            kjt_lengths.append(1)
        else:
            kjt_lengths.append(0)

sparse_features = KeyedJaggedTensor.from_lengths_sync(
    cat_cols,
    torch.tensor(kjt_values, device=torch.device("cpu")),
    torch.tensor(kjt_lengths, device=torch.device("cpu"), dtype=torch.int32),
)

# COMMAND ----------

product_ids = pdf['product_id'].sort_values().unique()[0:10000]
# product_ids = pdf['product_id'].sort_values().unique()
user_ids = pdf['user_id'].sort_values().unique()[0:40000]
# user_ids = pdf['user_id'].sort_values().unique()

product_ids_tensor = torch.tensor(product_ids, device=torch.device("cpu"))
user_ids_tensor = torch.tensor(user_ids, device=torch.device("cpu"))

values = torch.cat((user_ids_tensor, product_ids_tensor))

# Create lengths tensor
user_lengths = torch.ones(len(user_ids), dtype=torch.int32)
product_lengths = torch.ones(len(product_ids), dtype=torch.int32)

# Create offsets tensor
user_offsets = torch.arange(len(user_ids) + 1, dtype=torch.int32)
product_offsets = torch.arange(len(product_ids) + 1, dtype=torch.int32)

# Count occurrences for each tensor
product_unique, product_counts = torch.unique(product_ids_tensor, return_counts=True)
user_unique, user_counts = torch.unique(user_ids_tensor, return_counts=True)

# Create lengths tensor by concatenating the counts
lengths = torch.cat((user_counts, product_counts))

kjt = KeyedJaggedTensor(
  keys=cat_cols,
  values=values,
  lengths=lengths
)

# COMMAND ----------

with torch.no_grad():
    lookups = two_tower_model.ebc(kjt)
    item_embeddings = two_tower_model.candidate_proj(lookups['product_id'])
    user_embeddings = two_tower_model.query_proj(lookups['user_id'])

# COMMAND ----------

item_embeddings

# COMMAND ----------

user_embeddings
