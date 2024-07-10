# Databricks notebook source
# MAGIC %md Using 15.0 ML g5.24xlarge [A10] cluster with 1 worker node. Does not work with > 1 worker node (Eng is looking into it)

# COMMAND ----------

# %pip install -r ./requirements.txt
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -q --upgrade --no-deps --force-reinstall torch torchvision fbgemm-gpu torchrec --index-url https://download.pytorch.org/whl/cu118
# MAGIC %pip install -q torchmetrics==1.0.3 iopath pyre_extensions mosaicml-streaming==0.7.5
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

import gc
import itertools
import os
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple
from enum import Enum

import pandas as pd
import torch
import mlflow
from pyspark.ml.torch.distributor import TorchDistributor

# COMMAND ----------

# DBTITLE 1,Calculate user_ct and product_ct
training_set = spark.table("training_set").toPandas()
user_ct = training_set['user_id'].nunique()
product_ct = training_set['product_id'].nunique()

# Taken from earlier outputs (section 1.2, cell 2)
cat_cols = ["user_id", "product_id"]
emb_counts = [user_ct, product_ct]

# Delete dataframes to free up memory
del training_set
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md ### Base Dataclass for Training inputs
# MAGIC Feel free to modify any of the variables mentioned here, but note that the first layer for layer_sizes should be equivalent to embedding_dim.

# COMMAND ----------

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

class TrainingMethod(str, Enum):
    SNSG = "Single Node Single GPU Training"
    SNMG = "Single Node Multi GPU Training"
    MNMG = "Multi Node Multi GPU Training"

# TODO: Specify what level of distribution will be used for training. The Single-Node Multi-GPU and Multi-Node Multi-GPU arrangements will use the TorchDistributor for training.
training_method = TrainingMethod.MNMG

output_dir_train = config['output_dir_train']
output_dir_validation = config['output_dir_validation']
output_dir_test = config['output_dir_test']

# COMMAND ----------

# MAGIC %md ## Helper functions for model training

# COMMAND ----------

import os
import uuid
from typing import Tuple, List, Optional
from tqdm import tqdm
from collections import defaultdict
from functools import partial

from streaming import StreamingDataset, StreamingDataLoader

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.distributed._sharded_tensor import ShardedTensor

from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.inference.state_dict_transform import (
    state_dict_gather,
    state_dict_to_device,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.utils import Batch
from torchrec.modules.mlp import MLP
import torchmetrics as metrics
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)

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
    labels = torch.tensor(batch["label"], dtype=torch.int32)
    assert isinstance(labels, torch.Tensor)

    return Batch(
        dense_features=torch.zeros(1),
        sparse_features=sparse_features,
        labels=labels,
    )

transform_partial = partial(transform_to_torchrec_batch, num_embeddings_per_feature=emb_counts)

def get_dataloader_with_mosaic(path, batch_size, label):
    random_uuid = uuid.uuid4()
    local_path = f"/local_disk0/{random_uuid}"
    print(f"Getting {label} data from UC Volumes at {path} and saving to {local_path}")
    # dataset = StreamingDataset(remote=path, local=local_path, shuffle=True, batch_size=batch_size)
    dataset = StreamingDataset(local=path, shuffle=True, batch_size=batch_size)
    return StreamingDataLoader(dataset, batch_size=batch_size)
    # return DataLoader(dataset, batch_size=batch_size, pin_memory=True, drop_last=True)

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
      
# Store the results in mlflow
def get_relevant_fields(args, cat_cols, emb_counts):
    fields_to_save = ["epochs", "embedding_dim", "layer_sizes", "learning_rate", "batch_size"]
    result = { key: getattr(args, key) for key in fields_to_save }
    # add dense cols
    result["cat_cols"] = cat_cols
    result["emb_counts"] = emb_counts
    return result
  
def batched(it, n):
    assert n >= 1
    for x in it:
        yield itertools.chain((x,), itertools.islice(it, n - 1))

# Two Tower and TorchRec use special tensors called ShardedTensors.
# This code localizes them and puts them in the same rank that is saved to MLflow.
def gather_and_get_state_dict(model):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    state_dict = model.state_dict()
    gathered_state_dict = {}

    # Iterate over all items in the state_dict
    for fqn, tensor in state_dict.items():
        if isinstance(tensor, ShardedTensor):
            # Collect all shards of the tensor across ranks
            full_tensor = None
            if rank == 0:
                full_tensor = torch.zeros(tensor.size()).to(tensor.device)
            tensor.gather(0, full_tensor)
            if rank == 0:
                gathered_state_dict[fqn] = full_tensor
        else:
            # Directly add non-sharded tensors to the new state_dict
            if rank == 0:
                gathered_state_dict[fqn] = tensor

    return gathered_state_dict

def log_state_dict_to_mlflow(model, artifact_path) -> None:
    # All ranks participate in gathering
    state_dict = gather_and_get_state_dict(model)
    # Only rank 0 logs the state dictionary
    if dist.get_rank() == 0 and state_dict:
        mlflow.pytorch.log_state_dict(state_dict, artifact_path=artifact_path)

def evaluate(
    limit_batches: Optional[int],
    pipeline: TrainPipelineSparseDist,
    eval_dataloader: DataLoader,
    stage: str,
    transform_partial: partial) -> Tuple[float, float]:
    """
    Evaluates model. Computes and prints AUROC, accuracy, and average loss. Helper function for train_val_test.

    Args:
        limit_batches (Optional[int]): Limits the dataloader to the first `limit_batches` batches.
        pipeline (TrainPipelineSparseDist): data pipeline.
        eval_dataloader (DataLoader): Dataloader for either the validation set or test set.
        stage (str): "val" or "test".

    Returns:
        Tuple[float, float]: a tuple of (average loss, accuracy)
    """
    pipeline._model.eval()
    device = pipeline._device

    iterator = itertools.islice(iter(eval_dataloader), limit_batches)

    # We are using the AUROC for binary classification
    auroc = metrics.AUROC(task="binary").to(device)

    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Evaluating {stage} set",
            total=len(eval_dataloader),
            disable=False,
        )
    
    total_loss = torch.tensor(0.0).to(device)  # Initialize total_loss as a tensor on the same device as _loss
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        while True:
            try:
                _loss, logits, labels = pipeline.progress(map(transform_partial, iterator))
                # Calculating AUROC
                preds = torch.sigmoid(logits)
                auroc(preds, labels)
                # Calculating loss
                total_loss += _loss.detach()  # Detach _loss to prevent gradients from being calculated
                # total_correct += (logits.round() == labels).sum().item()  # Count the number of correct predictions
                total_samples += len(labels)
                if is_rank_zero:
                    pbar.update(1)
            except StopIteration:
                break
    
    auroc_result = auroc.compute().item()
    average_loss = total_loss / total_samples if total_samples > 0 else torch.tensor(0.0).to(device)
    average_loss_value = average_loss.item()

    if is_rank_zero:
        print(f"Average loss over {stage} set: {average_loss_value:.4f}.")
        print(f"AUROC over {stage} set: {auroc_result}")
    
    return average_loss_value, auroc_result
  
def train(
    pipeline: TrainPipelineSparseDist,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epoch: int,
    print_lr: bool,
    validation_freq: Optional[int],
    limit_train_batches: Optional[int],
    limit_val_batches: Optional[int],
    transform_partial: partial) -> None:
    """
    Trains model for 1 epoch. Helper function for train_val_test.

    Args:
        pipeline (TrainPipelineSparseDist): data pipeline.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        epoch (int): The number of complete passes through the training set so far.
        print_lr (bool): Whether to print the learning rate every training step.
        validation_freq (Optional[int]): The number of training steps between validation runs within an epoch.
        limit_train_batches (Optional[int]): Limits the training set to the first `limit_train_batches` batches.
        limit_val_batches (Optional[int]): Limits the validation set to the first `limit_val_batches` batches.

    Returns:
        None.
    """
    pipeline._model.train()

    # Get the first `limit_train_batches` batches
    iterator = itertools.islice(iter(train_dataloader), limit_train_batches)

    # Only print out the progress bar on rank 0
    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Epoch {epoch}",
            total=len(train_dataloader),
            disable=False,
        )

    # TorchRec's pipeline paradigm is unique as it takes in an iterator of batches for training.
    start_it = 0 
    n = validation_freq if validation_freq else len(train_dataloader)
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

        # If you are validating frequently, use the evaluation function
        if validation_freq and start_it % validation_freq == 0:
            evaluate(limit_val_batches, pipeline, val_dataloader, "val", transform_partial)
            pipeline._model.train()

def train_val_test(args, model, optimizer, device, train_dataloader, val_dataloader, test_dataloader, transform_partial) -> None:
    """
    Train/validation/test loop.

    Args:
        args (Args): args for training.
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device to use.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        test_dataloader (DataLoader): Test set's dataloader.

    Returns:
        TrainValTestResults.
    """
    pipeline = TrainPipelineSparseDist(model, optimizer, device)

    # Getting base auroc and saving it to mlflow
    val_loss, val_auroc = evaluate(args.limit_val_batches, pipeline, val_dataloader, "val", transform_partial)
    if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('val_loss', val_loss)
        mlflow.log_metric('val_auroc', val_auroc)

    # Running a training loop
    for epoch in range(args.epochs):
        train(
            pipeline,
            train_dataloader,
            val_dataloader,
            epoch,
            args.print_lr,
            args.validation_freq,
            args.limit_train_batches,
            args.limit_val_batches,
            transform_partial
        )

        # Evaluate after each training epoch
        val_loss, val_auroc = evaluate(args.limit_val_batches, pipeline, val_dataloader, "val", transform_partial)
        if int(os.environ["RANK"]) == 0:
            mlflow.log_metric('val_loss', val_loss)
            mlflow.log_metric('val_auroc', val_auroc)

        # Save the underlying model and results to mlflow
        log_state_dict_to_mlflow(pipeline._model.module, artifact_path=f"model_state_dict_{epoch}")
    
    # Evaluate on the test set after training loop finishes
    test_loss, test_auroc = evaluate(args.limit_test_batches, pipeline, test_dataloader, "test", transform_partial)

    if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('test_loss', test_loss)
        mlflow.log_metric('test_auroc', test_auroc)
    return test_auroc

# COMMAND ----------

# MAGIC %md ## Scratch Code moved to utils - keep until tested

# COMMAND ----------

# def transform_to_torchrec_batch(batch, num_embeddings_per_feature: Optional[List[int]] = None) -> Batch:
#     kjt_values: List[int] = []
#     kjt_lengths: List[int] = []
#     for col_idx, col_name in enumerate(cat_cols):
#         values = batch[col_name]
#         for value in values:
#             if value:
#                 kjt_values.append(
#                     value % num_embeddings_per_feature[col_idx]
#                 )
#                 kjt_lengths.append(1)
#             else:
#                 kjt_lengths.append(0)

#     sparse_features = KeyedJaggedTensor.from_lengths_sync(
#         cat_cols,
#         torch.tensor(kjt_values),
#         torch.tensor(kjt_lengths, dtype=torch.int32),
#     )
#     labels = torch.tensor(batch["label"], dtype=torch.int32)
#     assert isinstance(labels, torch.Tensor)

#     return Batch(
#         dense_features=torch.zeros(1),
#         sparse_features=sparse_features,
#         labels=labels,
#     )

# transform_partial = partial(transform_to_torchrec_batch, num_embeddings_per_feature=emb_counts)

# import uuid

# def get_dataloader_with_mosaic(path, batch_size, label):
#     print(f"Getting {label} data from UC Volumes")
#     random_uuid = uuid.uuid4()
#     dataset = StreamingDataset(remote=path, local=f"/local_disk0/{random_uuid}", shuffle=True, batch_size=batch_size)
#     # dataset = StreamingDataset(local=path, shuffle=True, batch_size=batch_size)
#     return StreamingDataLoader(dataset, batch_size=batch_size)
#     # return DataLoader(dataset, batch_size=batch_size, pin_memory=True, drop_last=True)

# COMMAND ----------

# MAGIC %md ### Creating the Relevant TorchRec code for Training
# MAGIC This section contains all of the training and evaluation code.

# COMMAND ----------

# MAGIC %md ### Two Tower Model Definition
# MAGIC This is taken directly from the torchrec example's page. Note that the loss is the Binary Cross Entropy loss, which requires labels to be within the values {0, 1}.

# COMMAND ----------

# import torch.nn.functional as F

# class TwoTower(nn.Module):
#     def __init__(
#         self,
#         embedding_bag_collection: EmbeddingBagCollection,
#         layer_sizes: List[int],
#         device: Optional[torch.device] = None
#     ) -> None:
#         super().__init__()

#         assert len(embedding_bag_collection.embedding_bag_configs()) == 2, "Expected two EmbeddingBags in the two tower model"
#         assert embedding_bag_collection.embedding_bag_configs()[0].embedding_dim == embedding_bag_collection.embedding_bag_configs()[1].embedding_dim, "Both EmbeddingBagConfigs must have the same dimension"

#         embedding_dim = embedding_bag_collection.embedding_bag_configs()[0].embedding_dim
#         self._feature_names_query: List[str] = embedding_bag_collection.embedding_bag_configs()[0].feature_names
#         self._candidate_feature_names: List[str] = embedding_bag_collection.embedding_bag_configs()[1].feature_names
#         self.ebc = embedding_bag_collection
#         self.query_proj = MLP(in_size=embedding_dim, layer_sizes=layer_sizes, device=device)
#         self.candidate_proj = MLP(in_size=embedding_dim, layer_sizes=layer_sizes, device=device)

#     def forward(self, kjt: KeyedJaggedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Convert the sparse input features into dense features
#         # Input: KeyedJaggedTensor (kjt) is passed through the embedding bag collection to get the pooled embeddings
#         pooled_embeddings = self.ebc(kjt)
#         # Pooled embeddings for query features are concatenated along the feature dimension
#         # Concatenated embeddings are passed through the query_proj MLP to produce final query embedding
#         query_embedding: torch.Tensor = self.query_proj(
#             torch.cat(
#                 [pooled_embeddings[feature] for feature in self._feature_names_query],
#                 dim=1,
#             )
#         )
#         # Pooled embeddings for candidate features are concatenated along the feature dimension
#         # Concatenated embeddings are passed through the candidate_proj MLP to produce final candidate embedding
#         candidate_embedding: torch.Tensor = self.candidate_proj(
#             torch.cat(
#                 [
#                     pooled_embeddings[feature]
#                     for feature in self._candidate_feature_names
#                 ],
#                 dim=1,
#             )
#         )
#         return query_embedding, candidate_embedding


# class TwoTowerTrainTask(nn.Module):
#     def __init__(self, two_tower: TwoTower) -> None:
#         super().__init__()
#         self.two_tower = two_tower
#         # The Two Tower model uses the Binary Cross Entropy loss (you can update it as needed for your own use case and dataset)
#         self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

#     def forward(self, batch: Batch) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
#         # batch.spare_features is KeyedJaggedTensor that is passed through TwoTower model
#         # TwoTower model returns embeddings for query and candidate
#         query_embedding, candidate_embedding = self.two_tower(batch.sparse_features)
#         # Dot product between query and candidate embeddings is computed
#         logits = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
#         # # clamp the outputs here to be between 0 and 1
#         # clamped_logits = torch.clamp(logits, min=0, max=1)
#         # loss = self.loss_fn(clamped_logits, batch.labels.float())
#         loss = self.loss_fn(logits, batch.labels.float())

#         # return loss, (loss.detach(), clamped_logits.detach(), batch.labels.detach())
#         return loss, (loss.detach(), logits.detach(), batch.labels.detach())

# COMMAND ----------

# MAGIC %md ### Training and Evaluation Helper Functions:
# MAGIC
# MAGIC Helper Functions for Distributed Model Saving, Distributed Model Training, and Evaluation

# COMMAND ----------

# def batched(it, n):
#     assert n >= 1
#     for x in it:
#         yield itertools.chain((x,), itertools.islice(it, n - 1))

# # Two Tower and TorchRec use special tensors called ShardedTensors.
# # This code localizes them and puts them in the same rank that is saved to MLflow.
# def gather_and_get_state_dict(model):
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     state_dict = model.state_dict()
#     gathered_state_dict = {}

#     # Iterate over all items in the state_dict
#     for fqn, tensor in state_dict.items():
#         if isinstance(tensor, ShardedTensor):
#             # Collect all shards of the tensor across ranks
#             full_tensor = None
#             if rank == 0:
#                 full_tensor = torch.zeros(tensor.size()).to(tensor.device)
#             tensor.gather(0, full_tensor)
#             if rank == 0:
#                 gathered_state_dict[fqn] = full_tensor
#         else:
#             # Directly add non-sharded tensors to the new state_dict
#             if rank == 0:
#                 gathered_state_dict[fqn] = tensor

#     return gathered_state_dict

# def log_state_dict_to_mlflow(model, artifact_path) -> None:
#     # All ranks participate in gathering
#     state_dict = gather_and_get_state_dict(model)
#     # Only rank 0 logs the state dictionary
#     if dist.get_rank() == 0 and state_dict:
#         mlflow.pytorch.log_state_dict(state_dict, artifact_path=artifact_path)

# def evaluate(
#     limit_batches: Optional[int],
#     pipeline: TrainPipelineSparseDist,
#     eval_dataloader: DataLoader,
#     stage: str,
#     transform_partial: partial) -> Tuple[float, float]:
#     """
#     Evaluates model. Computes and prints AUROC, accuracy, and average loss. Helper function for train_val_test.

#     Args:
#         limit_batches (Optional[int]): Limits the dataloader to the first `limit_batches` batches.
#         pipeline (TrainPipelineSparseDist): data pipeline.
#         eval_dataloader (DataLoader): Dataloader for either the validation set or test set.
#         stage (str): "val" or "test".

#     Returns:
#         Tuple[float, float]: a tuple of (average loss, accuracy)
#     """
#     pipeline._model.eval()
#     device = pipeline._device

#     iterator = itertools.islice(iter(eval_dataloader), limit_batches)

#     # We are using the AUROC for binary classification
#     auroc = metrics.AUROC(task="binary").to(device)

#     is_rank_zero = dist.get_rank() == 0
#     if is_rank_zero:
#         pbar = tqdm(
#             iter(int, 1),
#             desc=f"Evaluating {stage} set",
#             total=len(eval_dataloader),
#             disable=False,
#         )
    
#     total_loss = torch.tensor(0.0).to(device)  # Initialize total_loss as a tensor on the same device as _loss
#     total_correct = 0
#     total_samples = 0
#     with torch.no_grad():
#         while True:
#             try:
#                 _loss, logits, labels = pipeline.progress(map(transform_partial, iterator))
#                 # Calculating AUROC
#                 preds = torch.sigmoid(logits)
#                 auroc(preds, labels)
#                 # Calculating loss
#                 total_loss += _loss.detach()  # Detach _loss to prevent gradients from being calculated
#                 # total_correct += (logits.round() == labels).sum().item()  # Count the number of correct predictions
#                 total_samples += len(labels)
#                 if is_rank_zero:
#                     pbar.update(1)
#             except StopIteration:
#                 break
    
#     auroc_result = auroc.compute().item()
#     average_loss = total_loss / total_samples if total_samples > 0 else torch.tensor(0.0).to(device)
#     average_loss_value = average_loss.item()

#     if is_rank_zero:
#         print(f"Average loss over {stage} set: {average_loss_value:.4f}.")
#         print(f"AUROC over {stage} set: {auroc_result}")
    
#     return average_loss_value, auroc_result

# COMMAND ----------

# def train(
#     pipeline: TrainPipelineSparseDist,
#     train_dataloader: DataLoader,
#     val_dataloader: DataLoader,
#     epoch: int,
#     print_lr: bool,
#     validation_freq: Optional[int],
#     limit_train_batches: Optional[int],
#     limit_val_batches: Optional[int],
#     transform_partial: partial) -> None:
#     """
#     Trains model for 1 epoch. Helper function for train_val_test.

#     Args:
#         pipeline (TrainPipelineSparseDist): data pipeline.
#         train_dataloader (DataLoader): Training set's dataloader.
#         val_dataloader (DataLoader): Validation set's dataloader.
#         epoch (int): The number of complete passes through the training set so far.
#         print_lr (bool): Whether to print the learning rate every training step.
#         validation_freq (Optional[int]): The number of training steps between validation runs within an epoch.
#         limit_train_batches (Optional[int]): Limits the training set to the first `limit_train_batches` batches.
#         limit_val_batches (Optional[int]): Limits the validation set to the first `limit_val_batches` batches.

#     Returns:
#         None.
#     """
#     pipeline._model.train()

#     # Get the first `limit_train_batches` batches
#     iterator = itertools.islice(iter(train_dataloader), limit_train_batches)

#     # Only print out the progress bar on rank 0
#     is_rank_zero = dist.get_rank() == 0
#     if is_rank_zero:
#         pbar = tqdm(
#             iter(int, 1),
#             desc=f"Epoch {epoch}",
#             total=len(train_dataloader),
#             disable=False,
#         )

#     # TorchRec's pipeline paradigm is unique as it takes in an iterator of batches for training.
#     start_it = 0 
#     n = validation_freq if validation_freq else len(train_dataloader)
#     for batched_iterator in batched(iterator, n):
#         for it in itertools.count(start_it):
#             try:
#                 if is_rank_zero and print_lr:
#                     for i, g in enumerate(pipeline._optimizer.param_groups):
#                         print(f"lr: {it} {i} {g['lr']:.6f}")
#                 pipeline.progress(map(transform_partial, batched_iterator))
#                 if is_rank_zero:
#                     pbar.update(1)
#             except StopIteration:
#                 if is_rank_zero:
#                     print("Total number of iterations:", it)
#                 start_it = it
#                 break

#         # If you are validating frequently, use the evaluation function
#         if validation_freq and start_it % validation_freq == 0:
#             evaluate(limit_val_batches, pipeline, val_dataloader, "val", transform_partial)
#             pipeline._model.train()

# def train_val_test(args, model, optimizer, device, train_dataloader, val_dataloader, test_dataloader, transform_partial) -> None:
#     """
#     Train/validation/test loop.

#     Args:
#         args (Args): args for training.
#         model (torch.nn.Module): model to train.
#         optimizer (torch.optim.Optimizer): optimizer to use.
#         device (torch.device): device to use.
#         train_dataloader (DataLoader): Training set's dataloader.
#         val_dataloader (DataLoader): Validation set's dataloader.
#         test_dataloader (DataLoader): Test set's dataloader.

#     Returns:
#         TrainValTestResults.
#     """
#     pipeline = TrainPipelineSparseDist(model, optimizer, device)

#     # Getting base auroc and saving it to mlflow
#     val_loss, val_auroc = evaluate(args.limit_val_batches, pipeline, val_dataloader, "val", transform_partial)
#     if int(os.environ["RANK"]) == 0:
#         mlflow.log_metric('val_loss', val_loss)
#         mlflow.log_metric('val_auroc', val_auroc)

#     # Running a training loop
#     for epoch in range(args.epochs):
#         train(
#             pipeline,
#             train_dataloader,
#             val_dataloader,
#             epoch,
#             args.print_lr,
#             args.validation_freq,
#             args.limit_train_batches,
#             args.limit_val_batches,
#             transform_partial
#         )

#         # Evaluate after each training epoch
#         val_loss, val_auroc = evaluate(args.limit_val_batches, pipeline, val_dataloader, "val", transform_partial)
#         if int(os.environ["RANK"]) == 0:
#             mlflow.log_metric('val_loss', val_loss)
#             mlflow.log_metric('val_auroc', val_auroc)

#         # Save the underlying model and results to mlflow
#         log_state_dict_to_mlflow(pipeline._model.module, artifact_path=f"model_state_dict_{epoch}")
    
#     # Evaluate on the test set after training loop finishes
#     test_loss, test_auroc = evaluate(args.limit_test_batches, pipeline, test_dataloader, "test", transform_partial)

#     if int(os.environ["RANK"]) == 0:
#         mlflow.log_metric('test_loss', test_loss)
#         mlflow.log_metric('test_auroc', test_auroc)
#     return test_auroc

# COMMAND ----------

# MAGIC %md ## The main function
# MAGIC
# MAGIC This function trains the Two Tower recommendation model. For more information, see the following guides/docs/code:
# MAGIC
# MAGIC - https://pytorch.org/torchrec/
# MAGIC - https://github.com/pytorch/torchrec
# MAGIC - https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L75

# COMMAND ----------

def main(args: Args):
    import logging
    import torch
    import mlflow

    import streaming.base.util as util

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch import distributed as dist

    from torchrec.distributed.comm import get_local_size
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner.storage_reservations import (
        HeuristicalStorageReservation,
    )
    from torchrec.distributed.model_parallel import (
        DistributedModelParallel,
        get_default_sharders,
    )
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.modules.embedding_modules import EmbeddingBagCollection
    from torchrec.optim.keyed import KeyedOptimizerWrapper
    from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    from torch.distributed.optim import (
        _apply_optimizer_in_backward as apply_optimizer_in_backward,
    )

    # Some preliminary torch setup
    torch.jit._state.disable()
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    # os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
    device = torch.device(f"cuda:{local_rank}")
    backend = "nccl"
    torch.cuda.set_device(device)

    # Start MLflow
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token
    experiment = mlflow.set_experiment(experiment_path)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Start distributed process group
    dist.init_process_group(backend=backend)

    # Loading the data
    # util.clean_stale_shared_memory()

    # Loading the data
    try:
        train_dataloader = get_dataloader_with_mosaic(output_dir_train, args.batch_size, "train")
        val_dataloader = get_dataloader_with_mosaic(output_dir_validation, args.batch_size, "val")
        test_dataloader = get_dataloader_with_mosaic(output_dir_test, args.batch_size, "test")
    except RuntimeError as e:
        logger.error(f"NCCL initialization failed: {str(e)}")

    # Save parameters to MLflow
    if global_rank == 0:
        param_dict = get_relevant_fields(args, cat_cols, emb_counts)
        mlflow.log_params(param_dict)

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

    # Create the Two Tower model
    embedding_bag_collection = EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("meta"),
    )
    two_tower_model = TwoTower(
        embedding_bag_collection=embedding_bag_collection,
        layer_sizes=args.layer_sizes,
        device=device,
    )
    two_tower_train_task = TwoTowerTrainTask(two_tower_model)
    apply_optimizer_in_backward(
        RowWiseAdagrad,
        two_tower_train_task.two_tower.ebc.parameters(),
        {"lr": args.learning_rate},
    )

    # Create a plan to shard the embedding tables across the GPUs and creating a distributed model
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=args.batch_size,
        # If you get an out-of-memory error, increase the percentage. See
        # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
    )
    plan = planner.collective_plan(
        two_tower_model, get_default_sharders(), dist.GroupMember.WORLD
    )
    model = DistributedModelParallel(
        module=two_tower_train_task,
        device=device,
    )

    # Print out the sharding information to see how the embedding tables are sharded across the GPUs
    if global_rank == 0 and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for table_name, plan in plans.items():
                print(table_name, "\n", plan, "\n")
    
    log_state_dict_to_mlflow(model.module.two_tower, artifact_path="model_state_dict_base")

    optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.Adam(params, lr=args.learning_rate),
    )

    results = train_val_test(
        args,
        model,
        optimizer,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        transform_partial
    )

    # Destroy the process group
    dist.destroy_process_group()

# COMMAND ----------

# MAGIC %md ### Setup MLflow

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()['current_user()']
username

experiment_path = f'/Users/{username}/torchrec-instacart-two-tower-example'
 
# You will need these later
db_host = os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
 
# Manually create the experiment so that you know the id and can send that to the worker nodes when you scale later.
experiment = mlflow.set_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md ### Single Node + Single GPU Training
# MAGIC Here, you set the environment variables to run training over the sample set of 26M data points (stored in Volumes in Unity Catalog and collected using Mosaic StreamingDataset). You can expect each epoch to take ~16 minutes.

# COMMAND ----------

if training_method == TrainingMethod.SNSG:
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    args = Args()
    main(args)
else:
    print(f"`training_method` was set to {repr(training_method)[1:-1]}. Set `training_method` to {repr(TrainingMethod.SNSG)[1:-1]} to run training on this cell.")

# COMMAND ----------

# MAGIC %md ### Single Node - Multi GPU Training
# MAGIC
# MAGIC This notebook uses TorchDistributor to handle training on a g5.24xlarge instance with 4 A10 GPUs. You can view the sharding plan in the output logs to see what tables are located on what GPUs. This takes ~8 minutes to run per epoch.
# MAGIC
# MAGIC Note: There may be cases where you receive unexpected errors (like the Python Kernel crashing or segmentation faults). This is a transient error and the easiest way to overcome it is to skip the single node single GPU training code before you run any distributed code (single node multi GPU or multi node multi GPU).
# MAGIC
# MAGIC Note: If you see any errors that are associated with Mosaic Data Loading, these are transient errors that can be overcome by rerunning the failed cell.

# COMMAND ----------

if training_method == TrainingMethod.SNMG:
    args = Args()
    TorchDistributor(num_processes=4, local_mode=True, use_gpu=True).run(main, args)
else:
    print(f"`training_method` was set to {repr(training_method)[1:-1]}. Set `training_method` to {repr(TrainingMethod.SNMG)[1:-1]} to run training on this cell.")

# COMMAND ----------

# MAGIC %md ### Multi Node + Multi GPU Training
# MAGIC
# MAGIC This is tested with a g5.24xlarge instance with 4 A10 GPUs as a worker. You can view the sharding plan in the output logs to see what tables are located on what GPUs. This takes ~6 minutes to run per epoch.
# MAGIC
# MAGIC Note: There may be cases where you receive unexpected errors (like the Python Kernel crashing or segmentation faults). This is a transient error and the easiest way to overcome it is to skip the single node single GPU training code before you run any distributed code (single node multi GPU or multi node multi GPU).
# MAGIC
# MAGIC Note: If you see any errors that are associated with Mosaic Data Loading, these are transient errors that can be overcome by rerunning the failed cell.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import os

cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

w = WorkspaceClient(host=db_host, token=db_token)
cluster = w.clusters.get(cluster_id=cluster_id)
cluster_memory_mb = cluster.cluster_memory_mb
driver_node_type = cluster.driver_node_type_id
worker_node_type = cluster.node_type_id
num_workers = cluster.num_workers
spark_version = cluster.spark_version
cluster_cores = cluster.cluster_cores
cluster_params = {
  "cluster_memory_mb": cluster.cluster_memory_mb,
  "driver_node_type": cluster.driver_node_type_id,
  "worker_node_type": cluster.node_type_id,
  "num_workers": cluster.num_workers,
  "spark_version": cluster.spark_version,
  "cluster_cores": cluster.cluster_cores
}
cluster_params

# COMMAND ----------

import os
# NCCL failure workaround: https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html#nccl-failure-ncclinternalerror-internal-check-failed
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
# os.environ["NCCL_SOCKET_IFNAME"] = "eth"
# os.environ["NCCL_DEBUG"] = "info"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

embedding_dim = 128
layer_sizes = [128, 64]
args = Args(
  epochs=3, 
  embedding_dim=embedding_dim, 
  layer_sizes=layer_sizes, 
  learning_rate=0.01, 
  batch_size=1024*2, 
  print_lr=False
)

# assumes your driver node GPU count is the same as your worker nodes
device_count = torch.cuda.device_count()
worker_nodes = cluster.num_workers

if training_method == TrainingMethod.MNMG:
    # args = Args()
    TorchDistributor(num_processes=device_count * worker_nodes, local_mode=False, use_gpu=True).run(main, args)
else:
    print(f"`training_method` was set to {repr(training_method)[1:-1]}. Set `training_method` to {repr(TrainingMethod.MNMG)[1:-1]} to run training on this cell.")

# COMMAND ----------

# MAGIC %md ## Move code to separate notebook

# COMMAND ----------

# MAGIC %md ### Create Two Tower model from saved state_dict

# COMMAND ----------

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

def get_latest_run_id(experiment):
    latest_run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1).iloc[0]
    return latest_run.run_id

def get_latest_artifact_path(run_id):
    client = MlflowClient()
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    artifact_paths = [i.path for i in client.list_artifacts(run_id) if "base" not in i.path]
    return artifact_paths[-1]

def get_mlflow_model(run_id, artifact_path="model_state_dict"):
    from mlflow import MlflowClient

    device = torch.device("cuda")
    run = mlflow.get_run(run_id)
    
    cat_cols = eval(run.data.params.get('cat_cols'))
    emb_counts = eval(run.data.params.get('emb_counts'))
    layer_sizes = eval(run.data.params.get('layer_sizes'))
    embedding_dim = eval(run.data.params.get('embedding_dim'))

    MlflowClient().download_artifacts(run_id, f"{artifact_path}/state_dict.pth", "/databricks/driver")
    state_dict = mlflow.pytorch.load_state_dict(f"/databricks/driver/{artifact_path}")
    
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

# Loading the latest model state dict from the latest run of the current experiment
latest_run_id = get_latest_run_id(experiment)
latest_artifact_path = get_latest_artifact_path(latest_run_id)
two_tower_model, embedding_bag_collection, eb_configs, cat_cols, emb_counts = get_mlflow_model(latest_run_id, artifact_path=latest_artifact_path)

# COMMAND ----------

def transform_test(batch, cat_cols, emb_counts):
    kjt_values: List[int] = []
    kjt_lengths: List[int] = []
    for col_idx, col_name in enumerate(cat_cols):
        values = batch[col_name]
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
        torch.tensor(kjt_values),
        torch.tensor(kjt_lengths, dtype=torch.int32),
    )
    return sparse_features

# COMMAND ----------

# MAGIC %md ### Test inference on test data using Mosaic Streaming

# COMMAND ----------

num_batches = 5 # Number of batches to print out at a time 
batch_size = 1 # Print out each individual row

test_dataloader = iter(get_dataloader_with_mosaic(input_dir_test, batch_size, "test"))

# COMMAND ----------

labels = []
user_ids = []
product_ids = []
user_embeddings = []
item_embeddings = []

for _ in range(num_batches):
    device = torch.device("cuda:0")
    two_tower_model.to(device)
    two_tower_model.eval()

    next_batch = next(test_dataloader)
    expected_result = next_batch["label"][0]

    user_ids.append(next_batch["user_id"][0].item())
    product_ids.append(next_batch["product_id"][0].item())
    
    sparse_features = transform_test(next_batch, cat_cols, emb_counts)
    sparse_features = sparse_features.to(device)
    
    query_embedding, candidate_embedding = two_tower_model(kjt=sparse_features)
    user_embeddings.append(query_embedding.detach().cpu().numpy())
    item_embeddings.append(candidate_embedding.detach().cpu().numpy())

    actual_result = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
    sigmoid_result = torch.sigmoid(actual_result)
    # print(f"Expected Result: {expected_result}; Actual Result: {actual_result.round().item()}; Sigmoid Result: {actual_result}")
    print(f"Expected Result: {expected_result}; Rounded Result: {sigmoid_result.round().item()}; Sigmoid Result: {sigmoid_result}; Dot Product Result: {actual_result}")

# COMMAND ----------

# MAGIC %md ### Create embedding datasets

# COMMAND ----------

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TwoTowerDataset(Dataset):
    def __init__(self, dataframe, user_id_column='user_id', product_id_column='product_id', label_column='label', user_id_index_column='user_id_index'):
        self.user_id_column = user_id_column
        self.product_id_column = product_id_column
        self.label_column = label_column
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user_id = torch.tensor(self.dataframe.iloc[idx][self.user_id_column], dtype=torch.int32)
        product_id = torch.tensor(self.dataframe.iloc[idx][self.product_id_column], dtype=torch.int32)
        label = torch.tensor(self.dataframe.iloc[idx][self.label_column], dtype=torch.int32)

        return {
            'label': label,
            'product_id': product_id,
            'user_id': user_id
        }

# COMMAND ----------

import torch
import pandas as pd

num_batches = 1
batch_size = 1024

# Create the dataset and dataloader
df = spark.table("training_set").limit(10000)
pdf = df.toPandas()
dataset = TwoTowerDataset(pdf)
test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

labels = []
user_ids = []
product_ids = []
user_embeddings = []
item_embeddings = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
two_tower_model.to(device)
two_tower_model.eval()

for batch_idx, batch in enumerate(test_dataloader):

  batch = {key: value.to(device) for key, value in batch.items()}
  expected_result = batch["label"][0]

  # user_ids.append(next_batch["user_id"][0].item())
  user_ids.extend(batch["user_id"].tolist())
  product_ids.extend(batch["product_id"].tolist())
  
  sparse_features = transform_test(batch, cat_cols, emb_counts)
  sparse_features = sparse_features.to(device)
  
  with torch.no_grad():
    query_embedding, candidate_embedding = two_tower_model(kjt=sparse_features)
    user_embeddings.extend(query_embedding.detach().cpu().numpy().tolist())
    item_embeddings.extend(candidate_embedding.detach().cpu().numpy().tolist())

# Create a DataFrame
user_df = pd.DataFrame({
    'user_id': user_ids,
    'user_embedding': user_embeddings,
})

# Create a DataFrame
item_df = pd.DataFrame({
    'product_id': product_ids,
    'product_embedding': item_embeddings,
})

# Print or save the DataFrame
print(user_df.head())


# COMMAND ----------

user_df.shape
