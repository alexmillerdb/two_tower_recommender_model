# Databricks notebook source
# DBTITLE 1,Install Required Libraries
# MAGIC %pip install -q --upgrade --no-deps --force-reinstall torch==2.5.1+cu118 torchvision==0.20.1+cu118 fbgemm-gpu==1.0.0+cu118 torchrec==1.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118 #cu122 for 12.2
# MAGIC %pip install -q torchmetrics==1.0.3 iopath pyre_extensions mosaicml-streaming==0.8.0 # mosaicml-streaming==0.8.0
# MAGIC %pip install ray[tune,train,default]
# MAGIC %pip install optuna
# MAGIC %pip install --upgrade transformers==4.47.0 # to resolve error
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

# COMMAND ----------
%run ./config/notebook_config

# COMMAND ----------
import gc
import torch

# DBTITLE 1,Calculate user_ct and product_ct
training_set = spark.table("training_set").toPandas()
user_ct = training_set['user_id'].nunique()
product_ct = training_set['product_id'].nunique()
all_product_ids = training_set['product_id'].unique().tolist()

# Taken from earlier outputs (section 1.2, cell 2)
cat_cols = ["user_id", "product_id"]
emb_counts = [user_ct, product_ct]
print(emb_counts)

# Delete dataframes to free up memory
del training_set
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, MAX_NUM_WORKER_NODES, shutdown_ray_cluster
import ray

restart = True
if restart is True:
    try:
        shutdown_ray_cluster()
    except:
        pass
    try:
        ray.shutdown()
    except:
        pass

setup_ray_cluster(
    min_worker_nodes=2,
    max_worker_nodes=2,
    num_gpus_worker_node=1, 
    num_gpus_head_node=1,     
    num_cpus_worker_node=32, 
    num_cpus_head_node=24
)

ray.init(ignore_reinit_error=True)
print(ray.cluster_resources())

# COMMAND ----------

# DBTITLE 1,Import
import mlflow
import os

from collections import defaultdict
from functools import partial

from pyspark.ml.torch.distributor import TorchDistributor

from pyspark.sql.functions import countDistinct, count, when, col, max, first, datediff, lit, row_number
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

from shutil import rmtree

import torch
import torchmetrics as metrics
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.distributed._sharded_tensor import ShardedTensor
from torch.utils.data import DataLoader
from torch import nn
from torchrec import inference as trec_infer
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.inference.state_dict_transform import (
    state_dict_gather,
    state_dict_to_device,
)
from torchrec.modules.mlp import MLP
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from tqdm import tqdm
from typing import Tuple, List, Optional

import torchmetrics as metrics
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalMRR, RetrievalMAP

from dataclasses import dataclass, field
import itertools 
import uuid
import gc

from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.air import RunConfig
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer, get_device

from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from itertools import chain
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)

# Add Ray Data imports
import ray.data
from ray.train.torch import TorchTrainer

# COMMAND ----------

# def transform_to_torchrec_batch(batch, num_embeddings_per_feature, cat_cols, dense_cols):
#     """Transform Ray Data batch to TorchRec batch"""
#     # Process sparse features
#     kjt_values = []
#     kjt_lengths = []
    
#     for col_idx, col_name in enumerate(cat_cols):
#         values = batch[col_name]
#         for value in values:
#             if value:
#                 kjt_values.append(value % num_embeddings_per_feature[col_idx])
#                 kjt_lengths.append(1)
#             else:
#                 kjt_lengths.append(0)
    
#     sparse_features = KeyedJaggedTensor.from_lengths_sync(
#         cat_cols,
#         torch.tensor(kjt_values),
#         torch.tensor(kjt_lengths, dtype=torch.int32),
#     )
    
#     # Process dense features
#     dense = [
#         batch[feature].clone().detach().requires_grad_(True)
#         for feature in dense_cols
#     ]
#     dense_features = torch.cat(dense, dim=1)
    
#     # Process labels
#     labels = batch["label"]
    
#     return Batch(
#         dense_features=dense_features,
#         sparse_features=sparse_features,
#         labels=labels
#     )
def transform_to_torchrec_batch(batch, num_embeddings_per_feature: Optional[List[int]] = None) -> Batch:
    kjt_values: List[int] = []
    kjt_lengths: List[int] = []
    for col_idx, col_name in enumerate(cat_cols):
        values = batch[col_name]
        for value in values:
            if value:
                kjt_values.append(
                    value % num_embeddings_per_feature[col_idx]
                )
                kjt_lengths.append(1)
            else:
                kjt_lengths.append(0)

    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        cat_cols,
        torch.tensor(kjt_values),
        torch.tensor(kjt_lengths, dtype=torch.int32),
    )
    # labels = torch.tensor(batch["label"], dtype=torch.int32)
    labels = batch["label"].to(torch.int32)
    assert isinstance(labels, torch.Tensor)

    return Batch(
        dense_features=torch.zeros(1),
        sparse_features=sparse_features,
        labels=labels,
    )

transform_partial = partial(transform_to_torchrec_batch, num_embeddings_per_feature=emb_counts)

# def get_dataloader_with_mosaic(path, batch_size, label):
#     # This function downloads and returns the dataloader from mosaic
#     random_uuid = uuid.uuid4()
#     local_path = f"/local_disk0/{random_uuid}"
#     # local_path = f"/tmp/dataset/mds_train_{random_uuid}"
#     print(f"Getting {label} data from UC Volumes")
#     dataset = StreamingDataset(remote=path, local=local_path, shuffle=True, batch_size=batch_size)
#     return StreamingDataLoader(dataset, batch_size=batch_size, num_workers=8, prefetch_factor=2, persistent_workers=True)

# def get_ray_dataset(path, label):
#     """Convert Mosaic dataset to Ray Dataset"""
#     import ray.data
#     print(f"Getting {label} data from UC Volumes using Ray Data")
    
#     # Read Parquet files from the path
#     ds = ray.data.read_parquet(
#         path,
#         columns=SPARSE_COLS + DENSE_COLS + ["label"]
#     )
    
#     # Preprocess the dataset
#     def preprocess_batch(batch):
#         # Convert to torch tensors
#         processed = {
#             col: torch.tensor(batch[col].values) 
#             for col in SPARSE_COLS + DENSE_COLS
#         }
#         processed["label"] = torch.tensor(batch["label"].values)
#         return processed
    
#     return ds.map_batches(preprocess_batch, batch_size=1024)

def preprocess_batch(batch):
    print("Preprocessing a new batch...")
    # Use the column data directly.
    processed = {col: torch.tensor(batch[col]) for col in SPARSE_COLS}
    processed["label"] = torch.tensor(batch["label"])
    # print("Finished processing a batch. Batch keys:", list(processed.keys()))
    return processed

# Helper function to load a Ray Dataset from Parquet and apply preprocessing.
def get_ray_dataset(path, label, batch_size=1024):
    print(f"Getting {label} data from {path} using Ray Data")
    
    ds = ray.data.read_parquet(
        path,
        columns=SPARSE_COLS + ["label"]
    )
    
    # Apply the debug preprocess; you can adjust the batch_size as needed.
    ds = ds.map_batches(preprocess_batch, batch_size=batch_size)
    return ds

# Add the test integration function here
def test_data_pipeline_integration():
    """Test the integration between Ray Dataset loading and TorchRec batch transformation"""
    
    # 1. Create a small test dataset
    test_batch_size = 4
    sample_data = {
        "user_id": [1, 2, 3, 4],
        "product_id": [10, 20, 30, 40],
        "label": [0, 1, 0, 1]
    }
    
    # 2. Save as temporary parquet file
    import tempfile
    import pandas as pd
    
    # with tempfile.TemporaryDirectory() as temp_dir:
    # Save sample data as parquet
    # pd.DataFrame(sample_data).to_parquet(f"{temp_dir}/test.parquet")
    
    try:
        # 3. Load with get_ray_dataset
        print("get_ray_dataset")
        ray_dataset = get_ray_dataset(
            path=config['output_dir_train_parquet'],
            label="train",
            batch_size=test_batch_size
        )
        
        # 4. Get first batch
        print("get first batch")
        batch_iter = ray_dataset.iter_torch_batches(
            batch_size=test_batch_size,
            dtypes={col: torch.int64 for col in cat_cols + ['label']}
        )
        first_batch = next(iter(batch_iter))
        print(f"First batch: {first_batch}")
        
        print("transform to TorchRec batch")
        # 5. Transform to TorchRec batch
        torchrec_batch = transform_to_torchrec_batch(
            first_batch,
            num_embeddings_per_feature=emb_counts
        )
        
        # 6. Validate the transformed batch
        assert isinstance(torchrec_batch, Batch), "Should return a TorchRec Batch object"
        assert isinstance(torchrec_batch.sparse_features, KeyedJaggedTensor), "Should contain KeyedJaggedTensor"
        assert torchrec_batch.labels.shape[0] == test_batch_size, f"Expected {test_batch_size} labels"
        
        # Print successful transformation
        print("✅ Data pipeline integration test passed!")
        print(f"Batch shape: {torchrec_batch.labels.shape}")
        print(f"Sparse features keys: {torchrec_batch.sparse_features.keys()}")

        return torchrec_batch
        
    except Exception as e:
        print("❌ Data pipeline integration test failed!")
        print(f"Error: {str(e)}")
        raise

torchrec_batch = test_data_pipeline_integration()

def batched(it, n):
    assert n >= 1
    for x in it:
        yield itertools.chain((x,), itertools.islice(it, n - 1))

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
        loss = self.loss_fn(logits, batch.labels.float())

        return loss, (loss.detach(), logits.detach(), batch.labels.detach())

# interaction_weights = {
#     (0,0,1): 1.0,  # for 'Favorite'
#     (0,0,0): 1.0,  # for 'RequestListingInfoClicked'
#     (0,1,0): 1.0,  # for 'Share'
# }

# class TwoTower(nn.Module):
#     def __init__(
#         self,
#         embedding_bag_collection: EmbeddingBagCollection,
#         layer_sizes: List[List[int]],
#         device: Optional[torch.device] = None
#     ) -> None:
#         super().__init__()

#         # embedding_dim[0] is dimension of user features input
#         # embedding_dim[1] is dimension of item features input
#         embedding_dim = [53, 55]

#         # Update with sparse feature names for user and item features
#         self._feature_names_user: List[str] = ["client_index", "hour", "dayOfWeek"]
#         # Add zipcode and submarket below
#         self._feature_names_item: List[str] = ["item_index"]

#         # Use dense_cols to identify ordering of user vs. item features
#         self.dense_index = 9

#         self.ebc = embedding_bag_collection
#         self.user_proj = MLP(in_size=embedding_dim[0], layer_sizes=layer_sizes[0], device=device)
#         self.item_proj = MLP(in_size=embedding_dim[1], layer_sizes=layer_sizes[1], device=device)

#         self.device = device or torch.device('cuda' if torch.cuda.is_available() else None)

#     def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
#         pooled_embeddings = self.ebc(batch.sparse_features)

#         # Concatenate the sparse features
#         sparse_user_cat = torch.cat(
#             [pooled_embeddings[feature].float() for feature in self._feature_names_user],
#             dim=1,
#         )
#         sparse_item_cat = torch.cat(
#             [pooled_embeddings[feature].float() for feature in self._feature_names_item],
#             dim=1,
#         )

#         # Split dense features tensor based on item/user
#         dense_user = batch.dense_features[:, :self.dense_index].clone().detach().requires_grad_(True)
#         dense_item = batch.dense_features[:, self.dense_index:].clone().detach().requires_grad_(True)

#         # Pass concatenated tensor through MLP architecture (layers)
#         query_embedding: torch.Tensor = self.user_proj(
#             torch.cat(
#                 [sparse_user_cat, dense_user],
#                 dim=1,
#             ).float()
#         )
#         candidate_embedding: torch.Tensor = self.item_proj(
#             torch.cat(
#                 [sparse_item_cat, dense_item],
#                 dim=1,
#             ).float()
#         )

#         return query_embedding, candidate_embedding

# class WeightedBCELoss(nn.Module):
#     def __init__(self, weights):
#         super(WeightedBCELoss, self).__init__()
#         self.weights = weights
#         self.bce_loss = nn.BCELoss(reduction='none')
    
#     def forward(self, outputs, targets, interaction_types):
#         loss = self.bce_loss(outputs, targets)
#         weights = torch.tensor([self.weights[tuple(it.tolist())] for it in interaction_types], device=loss.device)
#         weighted_loss = loss * weights
#         return weighted_loss.mean()

# class TwoTowerTrainTask(nn.Module):
#     def __init__(self, two_tower: TwoTower) -> None:
#         super().__init__()
#         self.two_tower = two_tower
#         # The BCEWithLogitsLoss combines a sigmoid layer and binary cross entropy loss
#         # self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()
#         self.loss_fn: nn.Module = WeightedBCELoss(interaction_weights)

#     def forward(self, batch: Batch) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
#         query_embedding, candidate_embedding = self.two_tower(batch)
#         logits = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
#         probs = torch.sigmoid(logits) # ADDITION
#         # Modify batch.dense_features[:, :4] if placement of eventTypeOHE changes in training batch
#         loss = self.loss_fn(probs, batch.labels.float(), batch.dense_features[:, :3].clone().detach().requires_grad_(True))
#         return loss, (loss.detach(), logits.detach(), batch.labels.detach(), batch.sparse_features["client_index"].values().long().clone().detach())

@dataclass
class Args:
    """
    Training arguments.
    """
    epochs: int = 3  # Training for one Epoch
    embedding_dim: int = 128  # Embedding dimension is 128
    layer_sizes: List[int] = field(default_factory=lambda: [128, 64]) # The layers for the two tower model are 128, 64 (with the final embedding size for the outputs being 64)
    learning_rate: float = 0.01
    batch_size: int = 1024 # Set a larger batch size due to the large size of dataset
    print_sharding_plan: bool = True
    print_lr: bool = False  # Optional, prints the learning rate at each iteration step
    validation_freq: int = None  # Optional, determines how often during training you want to run validation (# of training steps)
    limit_train_batches: int = None  # Optional, limits the number of training batches
    limit_val_batches: int = None  # Optional, limits the number of validation batches
    limit_test_batches: int = None  # Optional, limits the number of test batches

# Note: define your sparse and dense column ordering and embedding counts globally.
# SPARSE_COLS = ["client_index", "item_index", "hour", "dayOfWeek"]
# DENSE_COLS = [
#     "eventTypeOHE", "deviceTypeIdOHE", "scaled_user_x", "scaled_user_y", "scaled_user_z",
#     "scaled_baths", "scaled_beds", "scaled_currentPrice", "scaled_item_x", "scaled_item_y",
#     "scaled_item_z", "scaled_listDate_diff", "propertyTypeOHE", "scaled_squareFeet_LotSize",
#     "scaled_yearBuild_diff"
# ]
# EMB_COUNTS = [1683153 + 1, 3070828 + 1, 24, 7]
EMB_COUNTS = emb_counts
SPARSE_COLS = cat_cols

def evaluate(args, pipeline, eval_dataloader, stage, transform_partial, device):

    pipeline._model.eval()

    iterator = itertools.islice(iter(eval_dataloader), args.limit_val_batches)

    # We are using the AUROC for binary classification
    auroc = metrics.AUROC(task="binary").to(device)
    # Add components: NDCG@k, MAP@k, AP@k
    # map25 = RetrievalMAP(top_k=25).to(device)
    # ndcg25 = RetrievalNormalizedDCG(top_k=25).to(device)
    # recall25 = RetrievalRecall(top_k=25).to(device)
    # mrr = RetrievalMRR().to(device)

    total_loss = torch.tensor(0.0, device=device)
    total_samples = 0

    with torch.no_grad():
        while True:
            try:
                _loss, logits, labels, indices = pipeline.progress(map(transform_partial, iterator))
                logits = logits.to(device)
                labels = labels.to(device)
                # Calculating AUROC
                preds = torch.sigmoid(logits)
                auroc(preds, labels)
                # Calculating additional metrics
                # map25(preds, labels, indices)
                # ndcg25(preds, labels.detach().long(), indices)
                # recall25(preds, labels.detach().long(), indexes=indices)
                # mrr(preds, labels.detach().long(), indexes=indices)
                # Calculating loss
                total_loss += _loss.detach().to(device)  # Detach _loss to prevent gradients from being calculated
                total_samples += len(labels)
            except StopIteration:
                break

    auroc_result = auroc.compute().item()
    # Loss
    average_loss = total_loss / total_samples if total_samples > 0 else torch.tensor(0.0).to(device)
    average_loss_value = average_loss.item()

    return average_loss_value, auroc_result

def train_one_epoch(args, pipeline, train_dataloader, transform_partial, epoch, device, print_lr=False):
    pipeline._model.train()

    # Get the first `limit_train_batches` batches
    iterator = itertools.islice(iter(train_dataloader), args.limit_train_batches)

    is_rank_zero = (dist.get_rank() == 0) if dist.is_initialized() else True
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Epoch {epoch}",
            total=len(train_dataloader),
            disable=False,
        )
    # TorchRec's pipeline paradigm is unique as it takes in an iterator of batches for training.
    start_it = 0
    n = args.validation_freq if args.validation_freq else len(train_dataloader)

    for batched_iterator in batched(iterator, n):
        for it in itertools.count(start_it):
            try:
                if is_rank_zero and print_lr:
                    for i, g in enumerate(pipeline._optimizer.param_groups):
                        print(f"lr: {it} {i} {g['lr']:.6f}")
                pipeline.progress(map(transform_partial, batched_iterator))
                if is_rank_zero:
                    pbar.update(1)
            except StopIteration:
                if is_rank_zero:
                    print("Total number of iterations:", it)
                start_it = it
                break

        # # If you are validating frequently, use the evaluation function
        # if args.validation_freq and start_it % args.validation_freq == 0:
        #     evaluate(args.limit_val_batches, pipeline, val_dataloader, "val")
        #     pipeline._model.train()

    # for batch in itertools.islice(iter(train_dataloader), args.limit_train_batches or 100):
    #     processed = transform_partial(batch)
    #     _ = pipeline.progress([processed])
    #     if is_rank_zero and args.print_lr:
    #         for i, g in enumerate(pipeline._optimizer.param_groups):
    #             print(f"Epoch {epoch} - Group {i} LR: {g['lr']:.6f}")
    # Optionally, run validation at freq
    val_loss, val_auroc = evaluate(args, pipeline, train_dataloader, "val", transform_partial, device)
    return val_loss, val_auroc

def train_val_test(args, model, optimizer, device, train_dl, val_dl, test_dl, transform_partial):
    pipeline = TrainPipelineSparseDist(model, optimizer, device)
    # initial validation before training
    val_loss, val_auroc = evaluate(args, pipeline, val_dl, "val", transform_partial, device)
    for epoch in range(args.epochs):
        train_loss, train_auroc = train_one_epoch(args, pipeline, train_dl, transform_partial, epoch, device)
        val_loss, val_auroc = evaluate(args, pipeline, val_dl, "val", transform_partial, device)
        # tune.report(epoch=epoch, val_loss=val_loss, val_auroc=val_auroc)
    test_loss, test_auroc = evaluate(args, pipeline, test_dl, "test", transform_partial, device)
    return {"test_loss": test_loss, "test_auroc": test_auroc}

# COMMAND ----------

def train_func(config):
    # If "train_loop_config" exists, use it; otherwise, use config directly.
    config = config.get("train_loop_config", config)
    
    # Get the device from Ray Train.
    device = get_device()
    
    # Create our Args object from the config.
    args = Args(
        epochs=config.get("epochs"),
        layer_sizes=config.get("layer_sizes"),
        mlp_lr=config.get("mlp_lr"),
        ebc_lr=config.get("ebc_lr"),
        batch_size=config.get("batch_size")
    )
    
    # Get Ray datasets
    train_ds = get_ray_dataset(config["train_dir"], "train")
    val_ds = get_ray_dataset(config["val_dir"], "val")
    test_ds = get_ray_dataset(config["test_dir"], "test")
    
    # Convert to DataLoaders using Ray Train
    train_dl = train_ds.iter_torch_batches(
        batch_size=args.batch_size,
        dtypes={col: torch.int64 for col in SPARSE_COLS},
    )
    val_dl = val_ds.iter_torch_batches(
        batch_size=args.batch_size,
        dtypes={col: torch.int64 for col in SPARSE_COLS},
    )
    test_dl = test_ds.iter_torch_batches(
        batch_size=args.batch_size,
        dtypes={col: torch.int64 for col in SPARSE_COLS},
    )

    # Create the embedding tables
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=emb_counts[feature_idx],
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(cat_cols)
    ]
    
    # Rest of the training setup remains the same
    # eb_configs = [
    #     EmbeddingBagConfig(name="t_client_index", embedding_dim=36, num_embeddings=EMB_COUNTS[0],
    #                        feature_names=['client_index']),
    #     EmbeddingBagConfig(name="t_item_index", embedding_dim=36, num_embeddings=EMB_COUNTS[1],
    #                        feature_names=['item_index']),
    #     EmbeddingBagConfig(name="t_hour", embedding_dim=4, num_embeddings=EMB_COUNTS[2],
    #                        feature_names=['hour']),
    #     EmbeddingBagConfig(name="t_dayOfWeek", embedding_dim=4, num_embeddings=EMB_COUNTS[3],
    #                        feature_names=['dayOfWeek']),
    # ]
    embedding_bag_collection = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))
    two_tower_model = TwoTower(embedding_bag_collection=embedding_bag_collection,
                               layer_sizes=args.layer_sizes, device=device)
    train_task = TwoTowerTrainTask(two_tower_model)
    apply_optimizer_in_backward(RowWiseAdagrad, two_tower_model.ebc.parameters(), {"lr": args.ebc_lr})
    
    # Create sharding plan if using multiple GPUs.
    if device.type == "cuda":
        local_world_size = int(os.environ.get("WORLD_SIZE", 1))
        topology = Topology(local_world_size=local_world_size,
                            world_size=local_world_size,
                            compute_device=device.type)
        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=args.batch_size,
            storage_reservation=HeuristicalStorageReservation(percentage=0.05)
        )
        from torchrec.distributed.model_parallel import get_default_sharders
        plan = planner.collective_plan(two_tower_model, get_default_sharders(), dist.group.WORLD)
    
    model = DistributedModelParallel(module=train_task, device=device)

    transform_partial = partial(transform_to_torchrec_batch,
                                num_embeddings_per_feature=EMB_COUNTS)
    
    # transform_partial = partial(transform_to_torchrec_batch,
    #                             num_embeddings_per_feature=EMB_COUNTS,
    #                             cat_cols=SPARSE_COLS,
    #                             dense_cols=DENSE_COLS)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)
    
    results = train_val_test(args, model, optimizer, device, train_dl, val_dl, test_dl, transform_partial)
    
    # Report the final metrics from the trial.
    tune.report(**results)

# COMMAND ----------
train_loop_config = {
    "epochs": 1,                           # Number of epochs to run
    "layer_sizes": [128, 64],             # Architecture of your MLP layers
    "mlp_lr": 1e-3,                         # Learning rate for the MLP part
    "ebc_lr": 5e-3,                         # Learning rate for the embedding bag collection optimizer
    "batch_size": 512,                       # Batch size used for training
    "train_dir": config['output_dir_train_parquet'],
    "val_dir": config['output_dir_validation_parquet'],
    "test_dir": config['output_dir_test_parquet'],              # Directory path for test data
    "embedding_dim": 32                     # Embedding dimension used in your Args object
}

trainer = TorchTrainer(
    train_func,
    train_loop_config=train_loop_config,
    scaling_config=ScalingConfig(
        num_workers=1, 
        use_gpu=True
    ),
    run_config=RunConfig(
        storage_path="/dbfs/tmp/ray/",
        name="two_tower_tune"
    )
)

result = trainer.fit()

# COMMAND ----------

# Define candidate layer configurations as tuples to avoid Optuna warnings.
layer_configs = [
    # Architecture 1
    [
        (53, 128, 64),
        (55, 128, 64)
    ],
    # Architecture 2
    [
        (128, 64, 32),
        (128, 64, 32)
    ],
    # # Architecture 3
    # [
    #     (64, 128, 64, 32),
    #     (64, 128, 64, 32)
    # ],
    # # Architecture 4
    # [
    #     (53, 32),
    #     (55, 32)
    # ],
    # # Architecture 5
    # [
    #     (64, 32),
    #     (64, 32)
    # ]
]

# Define candidate layer configurations as tuples to avoid Optuna warnings.
layer_configs = [
    # Architecture 1
    [
        (128, 64),
        (128, 64)
    ],
    # Architecture 2
    [
        (64, 32),
        (64, 32)
    ]
]

# Update search space paths to point to Parquet files
search_space = {
    "epochs": 1,
    "layer_sizes": tune.choice(layer_configs),
    "mlp_lr": tune.loguniform(5e-6, 1e-2),
    "ebc_lr": tune.loguniform(5e-5, 1e-1),
    "batch_size": tune.choice([1024]),
    # Update paths to point to Parquet files
    "train_dir": config['output_dir_train_parquet'],
    "val_dir": config['output_dir_validation_parquet'],
    "test_dir": config['output_dir_test_parquet']
}

# Create the TorchTrainer with your training function.
trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(
        num_workers=12, 
        use_gpu=True,
        resources_per_worker={
            "CPU": 12,
            "GPU": 1
        }),
    run_config=RunConfig(
        storage_path="/dbfs/tmp/ray/",
        name="two_tower_tune"
    )
)

# Now pass the search space nested under "train_loop_config".
tuner = tune.Tuner(
    trainer,
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
        search_alg=OptunaSearch(),
        num_samples=1,
        max_concurrent_trials=1
    ),
    param_space={"train_loop_config": search_space}
)

results = tuner.fit()

# Print the best configuration.
best_config = results.get_best_result(metric="test_loss", mode="min").config
print("Best config is:", best_config)