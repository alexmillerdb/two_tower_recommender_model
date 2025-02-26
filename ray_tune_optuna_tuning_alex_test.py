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
    num_cpus_worker_node=16, 
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

import streaming.base.util as util

from streaming import StreamingDataset, StreamingDataLoader
from streaming.base.converters import dataframe_to_mds
from streaming.base import MDSWriter

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

# COMMAND ----------

def transform_to_torchrec_batch(batch, num_embeddings_per_feature, cat_cols, dense_cols=None):
    # Process sparse features
    kjt_values = []
    kjt_lengths = []
    for col_idx, col_name in enumerate(cat_cols):
        values = batch[col_name]
        for value in values:
            if value:
                kjt_values.append(value % num_embeddings_per_feature[col_idx])
                kjt_lengths.append(1)
            else:
                kjt_lengths.append(0)
    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        cat_cols,
        torch.tensor(kjt_values),
        torch.tensor(kjt_lengths, dtype=torch.int32),
    )
    
    # Process dense features
    if dense_cols is not None:
        dense = [
            batch[feature].clone().detach().requires_grad_(True).unsqueeze(1)
            if batch[feature].dim() == 1 else batch[feature].clone().detach().requires_grad_(True)
            for feature in dense_cols
        ]
        dense_features = torch.cat(dense, dim=1)
    else:
        # If no dense columns specified, use a zero tensor
        # dense_features = torch.zeros((len(batch[cat_cols[0]]), 1), requires_grad=True)
        dense_features = torch.zeros(1)
        
    labels = torch.tensor(batch["label"], dtype=torch.int32)
    return Batch(dense_features=dense_features, sparse_features=sparse_features, labels=labels)

def get_dataloader_with_mosaic(path, batch_size, label, num_workers=4, prefetch_factor=2, persistent_workers=False):
    # This function downloads and returns the dataloader from mosaic
    random_uuid = uuid.uuid4()
    local_path = f"/local_disk0/{random_uuid}"
    # local_path = f"/tmp/dataset/mds_train_{random_uuid}"
    print(f"Getting {label} data from UC Volumes")
    dataset = StreamingDataset(remote=path, local=local_path, shuffle=True, batch_size=batch_size)
    
    # Create dataloader kwargs dict with only provided values
    dataloader_kwargs = {"batch_size": batch_size}
    if num_workers is not None:
        dataloader_kwargs["num_workers"] = num_workers
    if prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    if persistent_workers is not None:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        
    return StreamingDataLoader(dataset, **dataloader_kwargs)

def batched(it, n):
    assert n >= 1
    for x in it:
        yield itertools.chain((x,), itertools.islice(it, n - 1))

interaction_weights = {
    (0,0,1): 1.0,  # for 'Favorite'
    (0,0,0): 1.0,  # for 'RequestListingInfoClicked'
    (0,1,0): 1.0,  # for 'Share'
}

class TwoTower(nn.Module):
    """Two Tower neural network model for recommendation systems.

    This class implements a two-tower architecture where one tower processes user features
    and the other processes item features. The model can handle both sparse and dense features,
    and allows for flexible configuration of layer sizes and embedding dimensions for each tower.

    Parameters
    ----------
    embedding_bag_collection : EmbeddingBagCollection
        Collection of embedding bags for processing sparse features
    layer_sizes : Union[List[List[int]], List[int]]
        Neural network layer sizes. Can be either:
        - Single list of integers: Same architecture applied to both towers
        - List of two lists: Different architectures for user and item towers
    embedding_dim : Union[List[int], int]
        Embedding dimensions. Can be either:
        - Single integer: Same dimension used for both towers
        - List of two integers: Different dimensions for user and item towers
    feature_names_user : List[str]
        Names of features used for the user tower
    feature_names_item : List[str]
        Names of features used for the item tower
    dense_index : Optional[int], default=None
        Index to split dense features between user and item features.
        If None, model uses only sparse features
    device : Optional[torch.device], default=None
        Device to run the model on. If None, uses CUDA if available, else CPU

    Methods
    -------
    forward(batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]
        Forward pass of the model.
        Returns user (query) and item (candidate) embeddings.

    Notes
    -----
    The model supports both sparse and dense features:
    - Sparse features are processed through embedding bags
    - Dense features (if provided) are concatenated with sparse features
    - Each tower processes its features through an MLP architecture
    """
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        layer_sizes: List[List[int]] | List[int],
        embedding_dim: List[int] | int,
        feature_names_user: List[str],
        feature_names_item: List[str], 
        dense_index: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__()

        self._feature_names_user = feature_names_user
        self._feature_names_item = feature_names_item
        self.dense_index = dense_index
        self.has_dense = dense_index is not None

        # Handle single list or list of lists for layer_sizes
        if not any(isinstance(x, list) for x in layer_sizes):
            # Single list - use same architecture for both towers
            user_layers = layer_sizes
            item_layers = layer_sizes
        else:
            # List of lists - use different architecture for each tower
            user_layers = layer_sizes[0]
            item_layers = layer_sizes[1]

        # Handle single int or list for embedding_dim
        if isinstance(embedding_dim, int):
            # Single int - use same dim for both towers
            user_dim = embedding_dim
            item_dim = embedding_dim
        else:
            # List - use different dims for each tower
            user_dim = embedding_dim[0]
            item_dim = embedding_dim[1]

        self.ebc = embedding_bag_collection
        self.user_proj = MLP(in_size=user_dim, layer_sizes=user_layers, device=device)
        self.item_proj = MLP(in_size=item_dim, layer_sizes=item_layers, device=device)

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled_embeddings = self.ebc(batch.sparse_features)

        # Concatenate the sparse features
        sparse_user_cat = torch.cat(
            [pooled_embeddings[feature].float() for feature in self._feature_names_user],
            dim=1,
        )
        sparse_item_cat = torch.cat(
            [pooled_embeddings[feature].float() for feature in self._feature_names_item],
            dim=1,
        )

        if self.has_dense:
            # Split dense features tensor based on item/user
            dense_user = batch.dense_features[:, :self.dense_index].clone().detach().requires_grad_(True)
            dense_item = batch.dense_features[:, self.dense_index:].clone().detach().requires_grad_(True)

            # Pass concatenated tensor through MLP architecture (layers)
            query_embedding: torch.Tensor = self.user_proj(
                torch.cat(
                    [sparse_user_cat, dense_user],
                    dim=1,
                ).float()
            )
            candidate_embedding: torch.Tensor = self.item_proj(
                torch.cat(
                    [sparse_item_cat, dense_item],
                    dim=1,
                ).float()
            )
        else:
            # Use only sparse features
            query_embedding: torch.Tensor = self.user_proj(sparse_user_cat.float())
            candidate_embedding: torch.Tensor = self.item_proj(sparse_item_cat.float())

        return query_embedding, candidate_embedding

class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def forward(self, outputs, targets, interaction_types):
        loss = self.bce_loss(outputs, targets)
        weights = torch.tensor([self.weights[tuple(it.tolist())] for it in interaction_types], device=loss.device)
        weighted_loss = loss * weights
        return weighted_loss.mean()

class TwoTowerTrainTask(nn.Module):
    """Two Tower Training Task Module for recommendation systems.

    This module implements a PyTorch training task for a two-tower recommendation model.
    It handles both standard and weighted loss calculations and provides flexible output options.

    Args:
        two_tower (TwoTower): The two-tower model architecture to be trained.
        loss_fn (nn.Module, optional): Loss function to be used for training. 
            Defaults to BCEWithLogitsLoss if not provided.
        return_sparse (bool, optional): Whether to return sparse features in forward pass.
            Defaults to True.
        sparse_feature_names (List[str], optional): Names of sparse features to return.
            Defaults to ["client_index"] if not provided.

    Attributes:
        two_tower (TwoTower): The two-tower model instance.
        loss_fn (nn.Module): Loss function used for training.
        using_weighted_loss (bool): Flag indicating if weighted loss is being used.
        return_sparse (bool): Flag controlling sparse feature return behavior.
        sparse_feature_names (List[str]): List of sparse feature names to return.

    Returns:
        Tuple containing:
            - loss (torch.Tensor): The computed loss value
            - tuple: Contains detached tensors of (loss, logits, labels) if return_sparse=False,
                    or (loss, logits, labels, sparse_values) if return_sparse=True
                    where sparse_values is a dict of specified sparse features.
    """
    def __init__(self, two_tower: TwoTower, loss_fn: Optional[nn.Module] = None, return_sparse: bool = True, sparse_feature_names: Optional[List[str]] = None) -> None:
        super().__init__()
        self.two_tower = two_tower
        # Default to BCEWithLogitsLoss if no custom loss provided
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()
        self.using_weighted_loss = isinstance(self.loss_fn, WeightedBCELoss)
        self.return_sparse = return_sparse
        # Default to client_index if no sparse features specified
        self.sparse_feature_names = sparse_feature_names or ["client_index"]

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        query_embedding, candidate_embedding = self.two_tower(batch)
        logits = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
        probs = torch.sigmoid(logits)

        if self.using_weighted_loss:
            # Use interaction types for weighted loss
            loss = self.loss_fn(probs, batch.labels.float(), batch.dense_features[:, :3])
        else:
            # Use standard loss function
            loss = self.loss_fn(logits, batch.labels.float())

        if self.return_sparse:
            sparse_values = {name: batch.sparse_features[name].values().long().clone().detach() 
                           for name in self.sparse_feature_names}
            return loss, (loss.detach(), logits.detach(), batch.labels.detach(), sparse_values)
        else:
            return loss, (loss.detach(), logits.detach(), batch.labels.detach())

@dataclass
class Args:
    epochs: int
    embedding_dim: list = field(default_factory=lambda: [53, 55])
    layer_sizes: list = field(default_factory=list)
    mlp_lr: float = 1e-3
    ebc_lr: float = 1e-2
    batch_size: int = 1024
    print_sharding_plan: bool = True
    print_lr: bool = False
    validation_freq: int = None
    limit_train_batches: int = None
    limit_val_batches: int = None
    limit_test_batches: int = None

# Note: define your sparse and dense column ordering and embedding counts globally.
SPARSE_COLS = ["client_index", "item_index", "hour", "dayOfWeek"]
DENSE_COLS = [
    "eventTypeOHE", "deviceTypeIdOHE", "scaled_user_x", "scaled_user_y", "scaled_user_z",
    "scaled_baths", "scaled_beds", "scaled_currentPrice", "scaled_item_x", "scaled_item_y",
    "scaled_item_z", "scaled_listDate_diff", "propertyTypeOHE", "scaled_squareFeet_LotSize",
    "scaled_yearBuild_diff"
]
EMB_COUNTS = [1683153 + 1, 3070828 + 1, 24, 7]

def evaluate(args, pipeline, eval_dataloader, stage, transform_partial, device):
    pipeline._model.eval()

    iterator = itertools.islice(iter(eval_dataloader), args.limit_val_batches)

    # We are using the AUROC for binary classification 
    auroc = metrics.AUROC(task="binary").to(device)

    total_loss = torch.tensor(0.0, device=device)
    total_samples = 0

    with torch.no_grad():
        while True:
            try:
                # Handle both return types from pipeline.progress
                progress_output = pipeline.progress(map(transform_partial, iterator))
                if len(progress_output) == 4:
                    _loss, logits, labels, indices = progress_output
                else:
                    _loss, logits, labels = progress_output
                    indices = None
                    
                logits = logits.to(device)
                labels = labels.to(device)
                
                # Calculating AUROC
                preds = torch.sigmoid(logits)
                auroc(preds, labels)
                
                # Calculating loss
                total_loss += _loss.detach().to(device)
                total_samples += len(labels)
                
            except StopIteration:
                break

    auroc_result = auroc.compute().item()
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
    config_dirs = {
        "train_dir": config["train_dir"],
        "val_dir": config["val_dir"],
        "test_dir": config["test_dir"],
    }
    # Get dataloaders.
    train_dl = get_dataloader_with_mosaic(config_dirs["train_dir"], args.batch_size, "train")
    val_dl = get_dataloader_with_mosaic(config_dirs["val_dir"], args.batch_size, "val")
    test_dl = get_dataloader_with_mosaic(config_dirs["test_dir"], args.batch_size, "test")
    
    # Dynamically create embedding bag configs
    eb_configs = []
    embedding_dims = {
        'client_index': 36,
        'item_index': 36,
        'hour': 4,
        'dayOfWeek': 4
    }
    
    for col_name, count in zip(SPARSE_COLS, EMB_COUNTS):
        eb_configs.append(
            EmbeddingBagConfig(
                name=f"t_{col_name}",
                embedding_dim=embedding_dims[col_name],
                num_embeddings=count,
                feature_names=[col_name]
            )
        )

    embedding_bag_collection = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))
    two_tower_model = TwoTower(embedding_bag_collection=embedding_bag_collection,
                              layer_sizes=args.layer_sizes, 
                              embedding_dim=args.embedding_dim,
                              feature_names_user=SPARSE_COLS[:1],  # client_index
                              feature_names_item=SPARSE_COLS[1:],  # item_index and rest
                              dense_index=9,  # Split dense features after eventTypeOHE
                              device=device)
    train_task = TwoTowerTrainTask(two_tower_model)
    apply_optimizer_in_backward(RowWiseAdagrad, two_tower_model.ebc.parameters(), {"lr": args.ebc_lr})
    
    # Create sharding plan if using multiple GPUs.
    if device.type == "cuda":
        # Get world size from Ray Train's worker group
        local_world_size = int(os.environ.get("WORLD_SIZE", train.get_context().get_world_size()))
        topology = Topology(local_world_size=local_world_size,
                          world_size=local_world_size,
                          compute_device=device.type)
        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=args.batch_size,
            storage_reservation=HeuristicalStorageReservation(percentage=0.05)
        )
        if dist.is_initialized():
            plan = planner.collective_plan(two_tower_model, get_default_sharders(), dist.group.WORLD)
        else:
            plan = planner.collective_plan(two_tower_model, get_default_sharders())
    
    model = DistributedModelParallel(module=train_task, device=device)
    
    transform_partial = partial(transform_to_torchrec_batch,
                                num_embeddings_per_feature=EMB_COUNTS,
                                cat_cols=SPARSE_COLS,
                                dense_cols=DENSE_COLS)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)
    
    results = train_val_test(args, model, optimizer, device, train_dl, val_dl, test_dl, transform_partial)
    
    # Report the final metrics from the trial.
    tune.report(**results)

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

# Define search space as before.
search_space = {
    # "epochs": tune.choice([4,6]),
    "epochs": 1,
    "layer_sizes": tune.choice(layer_configs),
    "mlp_lr": tune.loguniform(5e-6, 1e-2),
    "ebc_lr": tune.loguniform(5e-5, 1e-1),
    "batch_size": tune.choice([1024]),
    "train_dir": "/Volumes/residential_dvm/recsys/weighted_activity_recsys/bakeoff/ts/mds_train",
    "val_dir": "/Volumes/residential_dvm/recsys/weighted_activity_recsys/bakeoff/ts/mds_validation",
    "test_dir": "/Volumes/residential_dvm/recsys/weighted_activity_recsys/bakeoff/ts/mds_test"
}

# Create the TorchTrainer with your training function.
trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(
        num_workers=3, 
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
        num_samples=3,
        max_concurrent_trials=3
    ),
    param_space={"train_loop_config": search_space}
)

results = tuner.fit()

# Print the best configuration.
best_config = results.get_best_result(metric="test_loss", mode="min").config
print("Best config is:", best_config)
