# Databricks notebook source
# MAGIC %pip install -r ./requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# %pip install -q --upgrade --no-deps --force-reinstall torch torchvision fbgemm-gpu torchrec --index-url https://download.pytorch.org/whl/cu118
# %pip install -q torchmetrics==1.0.3 iopath pyre_extensions mosaicml-streaming==0.7.5
# dbutils.library.restartPython()

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

# negative_samples = []
# user_ids = [0,0,0,1,2]
# product_ids = [0,1,5,1,3]
# labels = [1,1,1,1,1]
# user_ids_tensor = torch.tensor(user_ids, dtype=torch.int32)
# product_ids_tensor = torch.tensor(product_ids, dtype=torch.int32)
# labels_tensor = torch.tensor(labels, dtype=torch.int32)

# for user_id in list(set(user_ids)):
#     user_product_ids = product_ids_tensor[user_ids_tensor == user_id].tolist()
#     num_negatives = len(user_product_ids)
#     neg_products = [prod for prod in all_product_ids if prod not in user_product_ids]
#     sampled_neg_products = random.sample(neg_products, min(num_negatives, len(neg_products)))
#     negative_samples.extend([(user_id, neg_product, 0) for neg_product in sampled_neg_products])

# # Combine positive and negative samples
# positive_samples = list(zip(user_ids_tensor.tolist(), product_ids_tensor.tolist(), labels_tensor.tolist()))
# combined_samples = positive_samples + negative_samples

# # Convert combined samples to tensors
# combined_user_ids, combined_product_ids, combined_labels = zip(*combined_samples)
# combined_user_ids_tensor = torch.tensor(combined_user_ids, dtype=torch.int32)
# combined_product_ids_tensor = torch.tensor(combined_product_ids, dtype=torch.int32)
# combined_labels_tensor = torch.tensor(combined_labels, dtype=torch.int32)
# {"user_id": combined_user_ids_tensor,
#   "product_id": combined_product_ids_tensor,
#   "label": combined_labels_tensor}

# COMMAND ----------

# import torch
# from typing import Optional, Sequence, Callable, Union
# from torch.utils.data import Dataset
# from streaming import StreamingDataset, Stream
# import random

# class TabularDataset(StreamingDataset):
#     def __init__(self, 
#                  streams: Optional[Sequence[Stream]] = None,
#                  remote: Optional[str] = None,
#                  local: Optional[str] = None, 
#                  shuffle: Optional[bool] = False, 
#                  batch_size: Optional[int] = False, 
#                  all_product_ids: Optional[list] = None  # List of all product IDs
#                  ) -> None:
        
#         super().__init__(
#             streams=streams,
#             remote=remote, 
#             local=local, 
#             shuffle=shuffle, 
#             batch_size=batch_size
#             )
        
#         self.all_product_ids = all_product_ids if all_product_ids is not None else []
    
#     def __getitem__(self, idx):
#         obj = super().__getitem__(idx)  # Fetch the data object
#         user_ids = obj['user_id']
#         product_ids = obj['product_id']
#         labels = obj['label']
        
#         # Ensure user_ids, product_ids, and labels are lists or iterable
#         if not isinstance(user_ids, list):
#             user_ids = [user_ids]
#         if not isinstance(product_ids, list):
#             product_ids = [product_ids]
#         if not isinstance(labels, list):
#             labels = [labels]
        
#         user_ids_tensor = torch.tensor(user_ids, dtype=torch.int32)
#         product_ids_tensor = torch.tensor(product_ids, dtype=torch.int32)
#         labels_tensor = torch.tensor(labels, dtype=torch.int32)
        
#         # # Debugging prints
#         # print(f"user_ids_tensor: {user_ids_tensor}")
#         # print(f"product_ids_tensor: {product_ids_tensor}")
#         # print(f"labels_tensor: {labels_tensor}")
        
#         # Ensure tensors are of the same length
#         assert user_ids_tensor.shape == product_ids_tensor.shape == labels_tensor.shape, \
#             "Tensors must be of the same shape"
        
#         # Add random negative samples
#         negative_samples = []
#         unique_user_ids = torch.unique(user_ids_tensor)
        
#         for user_id in unique_user_ids:
#             user_product_ids = product_ids_tensor[user_ids_tensor == user_id].tolist()
#             num_negatives = len(user_product_ids)
#             neg_products = [prod for prod in self.all_product_ids if prod not in user_product_ids]
#             sampled_neg_products = random.sample(neg_products, min(num_negatives, len(neg_products)))
#             negative_samples.extend([(user_id.item(), neg_product, 0) for neg_product in sampled_neg_products])
        
#         # Combine positive and negative samples
#         positive_samples = list(zip(user_ids_tensor.tolist(), product_ids_tensor.tolist(), labels_tensor.tolist()))
#         combined_samples = positive_samples + negative_samples
        
#         # Convert combined samples to tensors
#         if combined_samples:
#             combined_user_ids, combined_product_ids, combined_labels = zip(*combined_samples)
#             combined_user_ids_tensor = torch.tensor(combined_user_ids, dtype=torch.int32)
#             combined_product_ids_tensor = torch.tensor(combined_product_ids, dtype=torch.int32)
#             combined_labels_tensor = torch.tensor(combined_labels, dtype=torch.int32)
#         else:
#             combined_user_ids_tensor = torch.tensor([], dtype=torch.int32)
#             combined_product_ids_tensor = torch.tensor([], dtype=torch.int32)
#             combined_labels_tensor = torch.tensor([], dtype=torch.int32)
        
#         return {"user_id": combined_user_ids_tensor,
#                 "product_id": combined_product_ids_tensor,
#                 "label": combined_labels_tensor}

# # Example usage
# from shutil import rmtree

# rmtree("/local_disk0/test")
# util.clean_stale_shared_memory()
# dataset = TabularDataset(remote=output_dir_train, local="/local_disk0/test", batch_size=32, shuffle=True, all_product_ids=all_product_ids)
# # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# dataloader = StreamingDataLoader(dataset, batch_size=32)

# # Iterate through the first 4 batches
# for i, batch in enumerate(dataloader):
#     if i < 4:
#         print(batch)
#     else:
#         break

# COMMAND ----------

# import random
# import torch
# from typing import List, Optional


# def transform_to_torchrec_batch(
#     batch, 
#     num_embeddings_per_feature: Optional[List[int]] = None,
#     num_negative_samples: int = 1,
#     num_items: int = 100000  # Total number of items in your dataset
# ) -> Batch:
#     kjt_values: List[int] = []
#     kjt_lengths: List[int] = []
#     user_ids = []
#     positive_item_ids = []

#     for col_idx, col_name in enumerate(cat_cols):
#         values = batch[col_name]
#         for value in values:
#             if value:
#                 kjt_values.append(value % num_embeddings_per_feature[col_idx])
#                 kjt_lengths.append(1)
#                 if col_name == 'user_id':
#                     user_ids.append(value)
#                 elif col_name == 'item_id':
#                     positive_item_ids.append(value)
#             else:
#                 kjt_lengths.append(0)

#     # Generate negative samples
#     negative_item_ids = []
#     for user_id in user_ids:
#         for _ in range(num_negative_samples):
#             negative_item = random.randint(0, num_items - 1)
#             while negative_item in positive_item_ids:
#                 negative_item = random.randint(0, num_items - 1)
#             negative_item_ids.append(negative_item)
#             kjt_values.append(negative_item % num_embeddings_per_feature[cat_cols.index('item_id')])
#             kjt_lengths.append(1)

#     # Duplicate user_ids for negative samples
#     for user_id in user_ids:
#         for _ in range(num_negative_samples):
#             kjt_values.append(user_id % num_embeddings_per_feature[cat_cols.index('user_id')])
#             kjt_lengths.append(1)

#     sparse_features = KeyedJaggedTensor.from_lengths_sync(
#         cat_cols * (num_negative_samples + 1),
#         torch.tensor(kjt_values),
#         torch.tensor(kjt_lengths, dtype=torch.int32),
#     )

#     # Create labels (1 for positive, 0 for negative)
#     positive_labels = torch.ones(len(user_ids), dtype=torch.float32)
#     negative_labels = torch.zeros(len(user_ids) * num_negative_samples, dtype=torch.float32)
#     labels = torch.cat([positive_labels, negative_labels])

#     return Batch(
#         dense_features=torch.zeros(1),
#         sparse_features=sparse_features,
#         labels=labels,
#     )

# # Example dataset
# example_batch = {
#     "user_id": [1, 2, 3, 4],
#     "item_id": [10, 20, 30, 40],
#     "label": [1, 1, 1, 1]
# }

# # Define categorical columns and number of embeddings per feature
# cat_cols = ["user_id", "item_id"]
# num_embeddings_per_feature = [100, 100]

# # Transform the example batch
# transformed_batch = transform_to_torchrec_batch(
#     example_batch, 
#     num_embeddings_per_feature=num_embeddings_per_feature,
#     num_negative_samples=1,  # Number of negative samples per user
#     num_items=50  # Total number of items in the dataset
# )

# # Print the transformed batch
# print("Sparse Features:", transformed_batch.sparse_features)
# print("Labels:", transformed_batch.labels)

# COMMAND ----------

import os
import uuid
from typing import Tuple, List, Optional, Sequence, Callable, Union
from tqdm import tqdm
from collections import defaultdict
from functools import partial
import random

from streaming import StreamingDataset, StreamingDataLoader, StreamingDataset, Stream
import streaming.base.util as util

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
    # labels = torch.tensor(batch["label"], dtype=torch.int32)
    labels = batch["label"].clone().detach().to(torch.int32)
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
    dataset = StreamingDataset(remote=path, local=local_path, shuffle=True, batch_size=batch_size)
    # dataset = StreamingDataset(local=path, shuffle=True, batch_size=batch_size)
    return StreamingDataLoader(dataset, batch_size=batch_size)

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

# MAGIC %md ## The main function
# MAGIC
# MAGIC This function trains the Two Tower recommendation model. For more information, see the following guides/docs/code:
# MAGIC
# MAGIC - https://pytorch.org/torchrec/
# MAGIC - https://github.com/pytorch/torchrec
# MAGIC - https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L75

# COMMAND ----------

# MAGIC %md #### To Do:
# MAGIC - Log entire model to MLflow (only logging state dict)

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
    util.clean_stale_shared_memory()

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

embedding_dim = 128
layer_sizes = [128, 64]
args = Args(
  epochs=3, 
  embedding_dim=embedding_dim, 
  layer_sizes=layer_sizes, 
  learning_rate=0.001, 
  batch_size=1024, 
  print_lr=False
)

os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

args = Args()
main(args)

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
# os.environ["NCCL_DEBUG"] = "info"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

embedding_dim = 128
layer_sizes = [128, 64, 32]
# test more layer_sizes
# test more than 3 epochs
# learning rate = 0.0001
args = Args(
  epochs=3, 
  embedding_dim=embedding_dim, 
  layer_sizes=layer_sizes, 
  learning_rate=0.001, 
  batch_size=1024, 
  print_lr=False
)

# assumes your driver node GPU count is the same as your worker nodes
gpu_per_node = 4
worker_nodes = cluster.num_workers

util.clean_stale_shared_memory()

if training_method == TrainingMethod.MNMG:
    # args = Args()
    TorchDistributor(num_processes=gpu_per_node * worker_nodes, local_mode=False, use_gpu=True).run(main, args)
else:
    print(f"`training_method` was set to {repr(training_method)[1:-1]}. Set `training_method` to {repr(TrainingMethod.MNMG)[1:-1]} to run training on this cell.")

# COMMAND ----------

# MAGIC %md ### Save learned embeddings from TwoTower model to delta tables and create VectorSearch index on the item embeddings to retrieve the most similar products based on the user embedding
# MAGIC - TorchRec example: https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L198

# COMMAND ----------

import pandas as pd
import numpy as np
import os
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
  
def create_keyed_jagged_tensor(num_embeddings, cat_cols, key, device=None):
    # Determine the device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    values = torch.tensor(list(range(num_embeddings)), device=device)

    if key == 'product_id':
        lengths = torch.tensor(
            [0] * num_embeddings + [1] * num_embeddings,
            device=device
        )
    elif key == 'user_id':
        lengths = torch.tensor(
            [1] * num_embeddings + [0] * num_embeddings,
            device=device
        )
    else:
        raise ValueError(f"Unsupported key: {key}")

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
            if lookup_column == 'product_id':
                lookups = two_tower_model.ebc(kjt)
                embeddings = two_tower_model.candidate_proj(lookups[lookup_column])
            elif lookup_column == 'user_id':
                lookups = two_tower_model.ebc(kjt)
                embeddings = two_tower_model.query_proj(lookups[lookup_column])
        
        print("Successfully processed embeddings")
        print(f"{lookup_column} embeddings shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# COMMAND ----------

# Loading the latest model state dict from the latest run of the current experiment
latest_run_id = get_latest_run_id(experiment)
latest_artifact_path = get_latest_artifact_path(latest_run_id)
two_tower_model, embedding_bag_collection, eb_configs, cat_cols, emb_counts = get_mlflow_model(
  latest_run_id, 
  artifact_path=latest_artifact_path, 
  device="cpu")

two_tower_model.to('cpu')
two_tower_model.eval()

# COMMAND ----------

latest_run_id

# COMMAND ----------

query_out_features = two_tower_model.query_proj._mlp[-1]._linear.out_features
candidate_out_features = two_tower_model.candidate_proj._mlp[-1]._linear.out_features
assert (query_out_features == candidate_out_features), "query_out_features != candidate_out_features"

# COMMAND ----------

# MAGIC %md ### Extract item embeddings from two tower model

# COMMAND ----------

product_kjt = create_keyed_jagged_tensor(emb_counts[1], cat_cols, 'product_id', device="cpu")
item_embeddings = process_embeddings(two_tower_model, product_kjt, 'product_id')
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

aisles = spark.table('aisles')
departments = spark.table("departments")
products = spark.table("products")

item_embedding_sdf = spark.createDataFrame(item_embedding_df) \
  .join(products, on='product_id') \
  .join(aisles, on='aisle_id') \
  .join(departments, on='department_id')

item_embeddings_table = f"{catalog}.{schema}.item_two_tower_embeddings_{candidate_out_features}"
item_embedding_sdf.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(item_embeddings_table)
display(item_embedding_sdf)

# COMMAND ----------

# MAGIC %md ### Add item embeddings to Vector Search index so we can retrieve similar products 

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
vector_search_endpoint_name = "one-env-shared-endpoint-0"

# Vector index
vs_index = f"item_two_tower_embeddings_index_{candidate_out_features}"
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
    embedding_dimension=candidate_out_features
  )

index.describe()

# COMMAND ----------

# MAGIC %md ### Extract user embeddings from two tower model and save to delta table

# COMMAND ----------

user_kjt = create_keyed_jagged_tensor(emb_counts[0], cat_cols, 'user_id', device="cpu")
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

user_embeddings_table = f"{catalog}.{schema}.user_two_tower_embeddings_{query_out_features}"
user_embedding_sdf = spark.createDataFrame(user_embedding_df)
user_embedding_sdf.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(user_embeddings_table)
display(user_embedding_sdf)
