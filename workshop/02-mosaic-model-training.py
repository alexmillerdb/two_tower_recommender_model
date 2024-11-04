# Databricks notebook source
# MAGIC %md # Two Tower Model Training (using TorchRec + TorchDistributor + StreamingDataset)
# MAGIC
# MAGIC This notebook illustrates how to create a distributed Two Tower recommendation model. This notebook was tested on `g4dn.12xlarge` instances (one instance as the driver, one instance as the worker) using the Databricks Runtime for ML 14.3 LTS. For more insight into the Two Tower recommendation model, you can view the following resources:
# MAGIC - Hopsworks definition: https://www.hopsworks.ai/dictionary/two-tower-embedding-model
# MAGIC - TorchRec's training implementation: https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L75

# COMMAND ----------

# DBTITLE 1,Torchrec Installation
# MAGIC %pip install -q --upgrade --no-deps --force-reinstall torch torchvision fbgemm-gpu torchrec --index-url https://download.pytorch.org/whl/cu118
# MAGIC %pip install -q torchmetrics==1.0.3 iopath pyre_extensions mosaicml-streaming==0.7.5
# MAGIC %pip install --upgrade mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Step 0. Defining Variables
# MAGIC
# MAGIC The following cell contains variables that are relevant to this notebook. If this notebook was imported from the marketplace, make sure to update the variables and paths as needed to point to the correct catalogs, volumes, etc.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
from enum import Enum
from typing import List
import dataclasses

# TODO: Specify what UC Volumes path contain the persisted MDS-formatted data
input_dir_train = config['output_dir_train']
input_dir_validation = config['output_dir_validation']
input_dir_test = config['output_dir_test']

# input_dir_train = "/Volumes/databricks_two_tower_recommendation_model_training_am/rec_datasets/learning_from_sets_data/mds_train/"
# input_dir_validation = "/Volumes/databricks_two_tower_recommendation_model_training_am/rec_datasets/learning_from_sets_data/mds_validation/"
# input_dir_test = "/Volumes/databricks_two_tower_recommendation_model_training_am/rec_datasets/learning_from_sets_data/mds_test/"

class TrainingMethod(str, Enum):
    SNSG = "Single Node Single GPU Training"
    SNMG = "Single Node Multi GPU Training"
    MNMG = "Multi Node Multi GPU Training"

# TODO: Specify what level of distribution will be used for training. The Single-Node Multi-GPU and Multi-Node Multi-GPU arrangements will use the TorchDistributor for training.
training_method = TrainingMethod.SNMG

# TODO: Update the training hyperparameters as needed. View the associated comments for more information. 
@dataclasses.dataclass
class Args:
    """
    Training arguments.
    """
    epochs: int = 3  # Training for one Epoch
    embedding_dim: int = 128  # Embedding dimension is 128 (should be equivalent to the first layer size in `layer_sizes`)
    layer_sizes: List[int] = dataclasses.field(default_factory=lambda: [128, 64]) # The layers for the two tower model are 128, 64 (with the final embedding size for the outputs being 64)
    learning_rate: float = 0.01
    batch_size: int = 1024 # Set a larger batch size due to the large size of dataset
    print_sharding_plan: bool = True
    print_lr: bool = False  # Optional, prints the learning rate at each iteration step
    validation_freq: int = None  # Optional, determines how often during training you want to run validation (# of training steps)
    limit_train_batches: int = None  # Optional, limits the number of training batches
    limit_val_batches: int = None  # Optional, limits the number of validation batches
    limit_test_batches: int = None  # Optional, limits the number of test batches

# TODO: Update the following variable with the URL for your Databricks workspace
db_host = os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# MAGIC %md The following cell contains all of the imports that are necessary for training the Two Tower model.

# COMMAND ----------

import os
from typing import List, Optional
from streaming import StreamingDataset, StreamingDataLoader

import torch
import torchmetrics as metrics
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.utils.data import DataLoader
from torch import nn
from torchrec import inference as trec_infer
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import DistributedModelParallel
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
from torch.distributed._sharded_tensor import ShardedTensor

from collections import defaultdict
from functools import partial
import mlflow

from typing import Tuple, List, Optional
from torchrec.modules.mlp import MLP

from pyspark.ml.torch.distributor import TorchDistributor
from tqdm import tqdm
import torchmetrics as metrics
import itertools

# COMMAND ----------

# MAGIC %md ## Step 1. Helper Functions for Recommendation Data Loading
# MAGIC
# MAGIC This step contains various helper functions for loading data for the downstream recommendation model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1. Helper functions for Converting to Pipelineable DataType
# MAGIC
# MAGIC Using TorchRec pipelines requires a pipelineable data type (which is `Batch` in this case). In this step, you create a helper function that takes each batch from the StreamingDataset and passes it through a transformation function to convert it into a pipelineable batch.
# MAGIC
# MAGIC For further context, see https://github.com/pytorch/torchrec/blob/main/torchrec/datasets/utils.py#L28.

# COMMAND ----------

# These values are specific to the "Learning From Sets of Items" dataset (and are visible in the output logs of the Dataset Curation notebook)
cat_cols = ["userId", "movieId"]
emb_counts = [193, 9740]

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2. Helper Function for DataLoading using Mosaic's StreamingDataset
# MAGIC
# MAGIC This utilizes Mosaic's StreamingDataset and Mosaic's StreamingDataLoader for efficient data loading. For more information, view this [documentation](https://docs.mosaicml.com/projects/streaming/en/stable/distributed_training/fast_resumption.html#saving-and-loading-state). 

# COMMAND ----------

def get_dataloader_with_mosaic(path, batch_size, label):
    print(f"Getting {label} data from UC Volumes")
    dataset = StreamingDataset(local=path, shuffle=True, batch_size=batch_size)
    return StreamingDataLoader(dataset, batch_size=batch_size)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Creating the Relevant TorchRec code for Training
# MAGIC
# MAGIC This section contains all of the training and evaluation code.

# COMMAND ----------

# Helper function to parse the training args to later save them in mlflow
def get_relevant_fields(args, cat_cols, emb_counts):
    fields_to_save = ["epochs", "embedding_dim", "layer_sizes", "learning_rate", "batch_size"]
    result = { key: getattr(args, key) for key in fields_to_save }
    # add dense cols
    result["cat_cols"] = cat_cols
    result["emb_counts"] = emb_counts
    return result

# COMMAND ----------

# MAGIC %md ### 2.1. Two Tower Model Definition
# MAGIC
# MAGIC This is taken directly from the [torchrec example's page](https://sourcegraph.com/github.com/pytorch/torchrec@2d62bdef24d144eaabeb0b8aa9376ded4a89e9ee/-/blob/examples/retrieval/modules/two_tower.py?L38:7-38:15). Note that the loss is the Binary Cross Entropy loss, which requires labels to be within the values {0, 1}.

# COMMAND ----------

import torch.nn.functional as F

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
        pooled_embeddings = self.ebc(kjt)
        query_embedding: torch.Tensor = self.query_proj(
            torch.cat(
                [pooled_embeddings[feature] for feature in self._feature_names_query],
                dim=1,
            )
        )
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


class TwoTowerTrainTask(nn.Module):
    def __init__(self, two_tower: TwoTower) -> None:
        super().__init__()
        self.two_tower = two_tower
        # The BCEWithLogitsLoss combines a sigmoid layer and binary cross entropy loss
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        query_embedding, candidate_embedding = self.two_tower(batch.sparse_features)
        logits = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
        loss = self.loss_fn(logits, batch.labels.float())
        return loss, (loss.detach(), logits.detach(), batch.labels.detach())

# COMMAND ----------

# MAGIC %md ### 2.2. Training and Evaluation Helper Functions

# COMMAND ----------

def batched(it, n):
    assert n >= 1
    for x in it:
        yield itertools.chain((x,), itertools.islice(it, n - 1))

# COMMAND ----------

# MAGIC %md #### 2.2.1. Helper Functions for Distributed Model Saving

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.2. Helper Functions for Distributed Model Training and Evaluation

# COMMAND ----------

import torchmetrics as metrics

def evaluate(
    limit_batches: Optional[int],
    pipeline: TrainPipelineSparseDist,
    eval_dataloader: DataLoader,
    stage: str) -> Tuple[float, float]:
    """
    Evaluates model. Computes and prints AUROC and average loss. Helper function for train_val_test.

    Args:
        limit_batches (Optional[int]): Limits the dataloader to the first `limit_batches` batches.
        pipeline (TrainPipelineSparseDist): data pipeline.
        eval_dataloader (DataLoader): Dataloader for either the validation set or test set.
        stage (str): "val" or "test".

    Returns:
        Tuple[float, float]: a tuple of (average loss, auroc)
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

# COMMAND ----------

def train(
    pipeline: TrainPipelineSparseDist,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epoch: int,
    print_lr: bool,
    validation_freq: Optional[int],
    limit_train_batches: Optional[int],
    limit_val_batches: Optional[int]) -> None:
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
            evaluate(limit_val_batches, pipeline, val_dataloader, "val")
            pipeline._model.train()

# COMMAND ----------

def train_val_test(args, model, optimizer, device, train_dataloader, val_dataloader, test_dataloader) -> None:
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
    val_loss, val_auroc = evaluate(args.limit_val_batches, pipeline, val_dataloader, "val")
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
        )

        # Evaluate after each training epoch
        val_loss, val_auroc = evaluate(args.limit_val_batches, pipeline, val_dataloader, "val")
        if int(os.environ["RANK"]) == 0:
            mlflow.log_metric('val_loss', val_loss)
            mlflow.log_metric('val_auroc', val_auroc)

        # Save the underlying model and results to mlflow
        log_state_dict_to_mlflow(pipeline._model.module, artifact_path=f"model_state_dict_{epoch}")
    
    # Evaluate on the test set after training loop finishes
    test_loss, test_auroc = evaluate(args.limit_test_batches, pipeline, test_dataloader, "test")
    if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('test_loss', test_loss)
        mlflow.log_metric('test_auroc', test_auroc)
    return test_auroc

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3. The Main Function
# MAGIC
# MAGIC This function trains the Two Tower recommendation model. For more information, see the following guides/docs/code:
# MAGIC
# MAGIC - https://pytorch.org/torchrec/
# MAGIC - https://github.com/pytorch/torchrec
# MAGIC - https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L75

# COMMAND ----------

from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)

def main(args: Args):
    import torch
    import mlflow
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    # Some preliminary torch setup
    torch.jit._state.disable()
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    backend = "nccl"
    torch.cuda.set_device(device)

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

    # Start MLflow
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token
    experiment = mlflow.set_experiment(experiment_path)

    # Save parameters to MLflow
    if global_rank == 0:
        param_dict = get_relevant_fields(args, cat_cols, emb_counts)
        mlflow.log_params(param_dict)

    # Start distributed process group
    dist.init_process_group(backend=backend)

    import streaming.base.util as util

    util.clean_stale_shared_memory()

    # Loading the data
    train_dataloader = get_dataloader_with_mosaic(input_dir_train, args.batch_size, "train")
    val_dataloader = get_dataloader_with_mosaic(input_dir_validation, args.batch_size, "val")
    test_dataloader = get_dataloader_with_mosaic(input_dir_test, args.batch_size, "test")

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

    # Start the training loop
    results = train_val_test(
        args,
        model,
        optimizer,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )

    # Destroy the process group
    dist.destroy_process_group()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4. Setting up MLflow
# MAGIC
# MAGIC **Note:** You must update the route for `db_host` to the URL of your Databricks workspace.

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()['current_user()']
username

experiment_path = f'/Users/{username}/torchrec-learning-from-sets-example'
 
# db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
 
# Manually create the experiment so that you know the id and can send that to the worker nodes when you scale later.
experiment = mlflow.set_experiment(experiment_path)

# COMMAND ----------

from databricks.sdk import WorkspaceClient

cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

w = WorkspaceClient(host=db_host, token=db_token)
gpus_per_node = torch.cuda.device_count()
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
  "cluster_cores": cluster.cluster_cores,
  "gpus_per_node": gpus_per_node
}
cluster_params

# COMMAND ----------

# MAGIC %md ## Step 3. Single Node + Single GPU Training
# MAGIC
# MAGIC Here, you set the environment variables to run training over the sample set of 100,000 data points (stored in Volumes in Unity Catalog and collected using Mosaic StreamingDataset). You can expect each epoch to take ~16 minutes.

# COMMAND ----------

if training_method == TrainingMethod.SNSG:
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    args = Args(
        epochs=1,
        embedding_dim=128,
        layer_sizes=[128, 64],
        learning_rate=0.01,
        batch_size=1024
    )
    main(args)
else:
    print(f"`training_method` was set to {repr(training_method)[1:-1]}. Set `training_method` to {repr(TrainingMethod.SNSG)[1:-1]} to run training on this cell.")

# COMMAND ----------

# MAGIC %md ## Step 4. Single Node - Multi GPU Training
# MAGIC
# MAGIC This notebook uses TorchDistributor to handle training on a `g4dn.12xlarge` instance with 4 T4 GPUs. You can view the sharding plan in the output logs to see what tables are located on what GPUs. This takes ~8 minutes to run per epoch.
# MAGIC
# MAGIC **Note**: There may be cases where you receive unexpected errors (like the Python Kernel crashing or segmentation faults). This is a transient error and the easiest way to overcome it is to skip the single node single GPU training code before you run any distributed code (single node multi GPU or multi node multi GPU).
# MAGIC
# MAGIC **Note**: If you see any errors that are associated with Mosaic Data Loading, these are transient errors that can be overcome by rerunning the failed cell.

# COMMAND ----------

if training_method == TrainingMethod.SNMG:
    args = Args(
        epochs=1,
        embedding_dim=128,
        layer_sizes=[128, 64],
        learning_rate=0.01,
        batch_size=1024
    )
    TorchDistributor(num_processes=gpus_per_node, local_mode=True, use_gpu=True).run(main, args)
else:
    print(f"`training_method` was set to {repr(training_method)[1:-1]}. Set `training_method` to {repr(TrainingMethod.SNMG)[1:-1]} to run training on this cell.")

# COMMAND ----------

# MAGIC %md ## Step 5. Multi Node + Multi GPU Training
# MAGIC
# MAGIC This is tested with a `g4dn.12xlarge` instance as a worker (with 4 T4 GPUs). You can view the sharding plan in the output logs to see what tables are located on what GPUs. This takes ~6 minutes to run per epoch.
# MAGIC
# MAGIC **Note**: There may be cases where you receive unexpected errors (like the Python Kernel crashing or segmentation faults). This is a transient error and the easiest way to overcome it is to skip the single node single GPU training code before you run any distributed code (single node multi GPU or multi node multi GPU).
# MAGIC
# MAGIC **Note**: If you see any errors that are associated with Mosaic Data Loading, these are transient errors that can be overcome by rerunning the failed cell.

# COMMAND ----------

training_method = TrainingMethod.MNMG

if training_method == TrainingMethod.MNMG:
    args = Args(
        epochs=1,
        embedding_dim=128,
        layer_sizes=[128, 64],
        learning_rate=0.01,
        batch_size=1024
    )
    TorchDistributor(num_processes=gpus_per_node*num_workers, local_mode=False, use_gpu=True).run(main, args)
else:
    print(f"`training_method` was set to {repr(training_method)[1:-1]}. Set `training_method` to {repr(TrainingMethod.MNMG)[1:-1]} to run training on this cell.")

# COMMAND ----------

# MAGIC %md ## Step 6. Inference
# MAGIC
# MAGIC Because the Two Tower Model's `state_dict`s are logged to MLflow, you can use the following code to load any of the saved `state_dict`s and create the associated Two Tower model with it. You can further expand this by 1) saving the loaded model to mlflow for inference or 2) doing batch inference using a UDF.
# MAGIC
# MAGIC Note: The saving code and loading code is used for loading the entire Two Tower model on one node and is useful as an example. In real world use cases, the expected model size could be significant (as the embedding tables can scale with the number of users or the number of products and items). It might be worthwhile to consider distributed inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1. Creating the Two Tower model from saved `state_dict`
# MAGIC
# MAGIC **Note:** You must update this with the correct `run_id` and path to the MLflow artifact.

# COMMAND ----------

from mlflow import MlflowClient

def get_latest_run_id(experiment):
    latest_run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1).iloc[0]
    return latest_run.run_id

def get_latest_artifact_path(run_id):
    client = MlflowClient()
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    artifact_paths = [i.path for i in client.list_artifacts(run_id) if "model_state_dict" in i.path and "base" not in i.path]
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

    if key == 'movieId':
        lengths = torch.tensor(
            [0] * num_embeddings + [1] * num_embeddings,
            device=device
        )
    elif key == 'userId':
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
            if lookup_column == 'movieId':
                lookups = two_tower_model.ebc(kjt)
                embeddings = two_tower_model.candidate_proj(lookups[lookup_column])
            elif lookup_column == 'userId':
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

# MAGIC %md
# MAGIC ### 6.2. Helper Function to Transform Dataloader to Two Tower Inputs
# MAGIC
# MAGIC The inputs that Two Tower expects are: `sparse_features`, so this section reuses aspects of the code from Section 3.4.2. The code shown here is verbose for clarity.

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

# MAGIC %md
# MAGIC ### 6.3. Getting the Data

# COMMAND ----------

num_batches = 5 # Number of batches to print out at a time 
batch_size = 1 # Print out each individual row

test_dataloader = iter(get_dataloader_with_mosaic(input_dir_test, batch_size, "test"))

# COMMAND ----------

# MAGIC %md ### 6.4. Running Tests
# MAGIC
# MAGIC In this example, you ran training for 3 epochs. The results were reasonable. Running a larger number of epochs would likely lead to optimal performance.

# COMMAND ----------

for _ in range(num_batches):
    device = torch.device("cuda:0")
    two_tower_model.to(device)
    two_tower_model.eval()

    next_batch = next(test_dataloader)
    expected_result = next_batch["label"][0]
    
    sparse_features = transform_test(next_batch, cat_cols, emb_counts)
    sparse_features = sparse_features.to(device)
    
    query_embedding, candidate_embedding = two_tower_model(kjt=sparse_features)
    actual_result = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
    actual_result = torch.sigmoid(actual_result)
    print(f"Expected Result: {expected_result}; Actual Result: {actual_result.round().item()}")

# COMMAND ----------

# MAGIC %md ## Step 7. Model Serving and Vector Search
# MAGIC
# MAGIC For information about how to serve the model, see the Databricks Model Serving documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)).
# MAGIC
# MAGIC Also, the Two Tower model is unique as it generates a `query` and `candidate` embedding, and therefore, allows you to create a vector index of movies, and then allows you to find the K movies that a user (given their generated vector) would most likely give a high rating. For more information, view the code [here](https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L198) for how to create your own FAISS Index. You can also take a similar approach with Databricks Vector Search ([AWS](https://docs.databricks.com/en/generative-ai/vector-search.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/vector-search/)).

# COMMAND ----------

# MAGIC %md ### Extract item embeddings from two tower model

# COMMAND ----------

import pandas as pd

two_tower_model.to('cpu')
two_tower_model.eval()
product_kjt = create_keyed_jagged_tensor(emb_counts[1], cat_cols, 'movieId', device="cpu")
item_embeddings = process_embeddings(two_tower_model, product_kjt, 'movieId')
# Convert KJT to dictionary
item_kjt_dict = product_kjt.to_dict()

# Get product_id values
movie_id_values = item_kjt_dict['movieId'].values()

print("Product IDs:", movie_id_values)

# Convert tensor to numpy array
item_embedding_array = item_embeddings.numpy()

# Create pandas DataFrame with arrays and product ids
item_embedding_df = pd.DataFrame({'embeddings': [row for row in item_embedding_array]})
item_embedding_df['movie_id'] = movie_id_values + 1

item_embedding_sdf = spark.createDataFrame(item_embedding_df)

# item_embeddings_table = f"{catalog}.{schema}.item_two_tower_embeddings_{candidate_out_features}"
# item_embedding_sdf.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(item_embeddings_table)
display(item_embedding_sdf)

# COMMAND ----------

# MAGIC %md ### Add item embeddings to Vector Search index so we can retrieve similar products

# COMMAND ----------

# from databricks.vector_search.client import VectorSearchClient

# vsc = VectorSearchClient()
# vector_search_endpoint_name = "one-env-shared-endpoint-0" # TO DO: entire in VS Endpoint

# # Vector index
# vs_index = f"item_two_tower_embeddings_index_{candidate_out_features}"
# vs_index_fullname = f"{catalog}.{schema}.{vs_index}"
# try:
#   index = vsc.get_index(vector_search_endpoint_name, vs_index_fullname)
#   index.sync()
# except:
#   index = vsc.create_delta_sync_index_and_wait(
#     endpoint_name=vector_search_endpoint_name,
#     source_table_name=item_embeddings_table,
#     index_name=vs_index_fullname,
#     pipeline_type='TRIGGERED',
#     primary_key="product_id",
#     embedding_vector_column="embeddings",
#     embedding_dimension=candidate_out_features
#   )

# index.describe()

# COMMAND ----------

# MAGIC %md ### Register your model to MLflow for serving and batch inference

# COMMAND ----------

def load_model_from_mlflow(run_id: str, model_class: torch.nn.Module, model_config: dict) -> Optional[torch.nn.Module]:
    """
    Load PyTorch model from MLflow model_state_dict.
    
    Args:
        run_id: MLflow run ID
        model_class: PyTorch model class to instantiate
        model_config: Configuration dictionary for model initialization
        
    Returns:
        Loaded PyTorch model or None if loading fails
    """
    try:
        # List artifacts to find model state dict
        artifacts = mlflow.artifacts.list_artifacts(
            run_id=run_id,
            artifact_path="model"  # Default MLflow model artifact path
        )
        
        # Find state dict file
        state_dict_files = [
            artifact.path 
            for artifact in artifacts 
            if 'state_dict' in artifact.path
        ]
        
        if not state_dict_files:
            print("No model state dict found in artifacts")
            return None
            
        # Download state dict file
        state_dict_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=state_dict_files[0]
        )
        
        print(f"Downloaded state dict from: {state_dict_files[0]}")
        
        # Initialize model
        model = model_class(config=model_config)
        
        # Load state dict
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        
        # Move model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        print(f"Successfully loaded model and moved to {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# COMMAND ----------

import mlflow
import torch
import pandas as pd
import numpy as np
from mlflow.pyfunc import PythonModel, PythonModelContext
from typing import Dict, List
from mlflow.models.signature import infer_signature

class TwoTowerWrapper(PythonModel):
    """
    MLflow PythonModel wrapper for TwoTower model that handles dictionary input and returns list outputs.
    """
    def __init__(self, two_tower_model, device):
        self.two_tower_model = two_tower_model.to(device)
        self.device = device

    def _calculate_logits_from_batch(self, batch):
        batch = self._transform_to_torchrec_batch(batch)
        query_embedding, candidate_embedding = self.two_tower_model(batch.sparse_features)
        logits = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
        return logits
    
    def _transform_to_torchrec_batch(self, batch, num_embeddings_per_feature: Optional[List[int]] = emb_counts) -> Batch:
        kjt_values: List[int] = []
        kjt_lengths: List[int] = []
        
        for col_idx, col_name in enumerate(cat_cols):
            values = batch[col_name]
            for value in values:
                if value is not None:
                    kjt_values.append(
                        value % num_embeddings_per_feature[col_idx]
                    )
                    kjt_lengths.append(1)
                else:
                    kjt_lengths.append(0)

        sparse_features = KeyedJaggedTensor.from_lengths_sync(
            cat_cols,
            torch.tensor(kjt_values).to(self.device),
            torch.tensor(kjt_lengths, dtype=torch.int32).to(self.device),
        )
        
        # Handle labels
        if "label" in batch:
            labels = torch.tensor(batch["label"], dtype=torch.int32).to(self.device)
        else:
            labels = torch.zeros(len(next(iter(batch.values()))), dtype=torch.int32).to(self.device)

        return Batch(
            dense_features=torch.zeros(1).to(self.device),
            sparse_features=sparse_features,
            labels=labels,
        )

    def predict(self, context, model_input: Dict[str, List]) -> List[float]:
        """
        Make predictions using the model.
        
        Args:
            context: MLflow context (required by PyFunc)
            model_input: Dictionary with feature lists
            
        Returns:
            List of prediction probabilities
        """
        try:
            # Convert input to tensors and move to device
            batch = {
                key: torch.tensor(value).to(self.device)
                for key, value in model_input.items()
            }
            
            # Run inference
            with torch.no_grad():
                logits = self._calculate_logits_from_batch(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                
            # Ensure output is always a list
            if isinstance(probs, (float, np.float32, np.float64)):
                return [float(probs)]
            elif isinstance(probs, np.ndarray):
                return probs.tolist()
            else:
                return list(probs)
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise

# COMMAND ----------

test = TwoTowerWrapper(two_tower_model, device="cpu")
test.predict(context="", model_input={"label": np.array([0,1]), "movieId": np.array([893,900]), "userId": np.array([54, 55])})

# COMMAND ----------

test.predict(context="", model_input=next_batch)

# COMMAND ----------

def get_package_versions() -> List[str]:
    """
    Get current versions of required packages dynamically.
    """
    import torch
    import torchrec
    import torchmetrics
    
    requirements = [
        f"torch=={torch.__version__}",
        f"torchrec=={torchrec.__version__}",
        f"torchmetrics=={torchmetrics.__version__}",
        "numpy",
        "pandas"
    ]
    
    return requirements

def clean_package_versions(requirements: List[str]) -> List[str]:
    """
    Clean package requirements by removing CUDA tags after '+'.
    
    Args:
        requirements: List of package requirements
        
    Returns:
        Cleaned requirements list
    """
    cleaned = []
    for req in requirements:
        if '=' in req:
            # Split package and version
            package, version = req.split('==')
            # Remove anything after '+'
            if '+' in version:
                version = version.split('+')[0]
            cleaned.append(f"{package}=={version}")
        else:
            cleaned.append(req)
            
    return cleaned

# COMMAND ----------

def log_pyfunc_model_to_mlflow(
    model: torch.nn.Module,
    run_id: str,
    model_name: str,
    input_example: Dict[str, List] = None
) -> None:
    """
    Log PyFunc model to MLflow and register in Unity Catalog with proper signature.
    
    Args:
        model: PyTorch model to log
        run_id: MLflow run ID to log model to
        model_name: Name to register model as in Unity Catalog (format: catalog.schema.model_name)
        input_example: Optional example input for model signature
    """
    try:
        # Create sample input and output for signature
        sample_input = {
            "label": np.array([0]), 
            "movieId": np.array([893]), 
            "userId": np.array([54])
            }
        # Getting the model signature and logging the model
        pyfunc_two_tower_model = TwoTowerWrapper(model, device="cpu")
        predictions = pyfunc_two_tower_model.predict(context="", model_input=sample_input)
        # current_output = {"prediction": np.array(predictions)}
        signature = infer_signature(
          model_input=sample_input, 
          model_output=predictions)
        
        pip_requirements = get_package_versions()
        clean_pip_requirements = clean_package_versions(pip_requirements)
            
        # Set MLflow registry to Unity Catalog
        mlflow.set_registry_uri("databricks-uc")
        
        # Start run in existing context
        with mlflow.start_run(run_id=run_id):
            # Log model as PyTorch flavor with signature
            model_info = mlflow.pyfunc.log_model(
              artifact_path="two_tower_pyfunc", 
              python_model=pyfunc_two_tower_model, 
              signature=signature, 
              input_example=sample_input,
              registered_model_name=model_name,
              extra_pip_requirements=clean_pip_requirements
              )
            
            print(f"Model logged to: {model_info.model_uri}")

        return model_info
            
    except Exception as e:
        print(f"Error logging model: {str(e)}")
        raise

# Example usage:
model_name = f"{catalog}.{schema}.learning_from_sets_two_tower"

# Log and register model
model_info = log_pyfunc_model_to_mlflow(
    model=two_tower_model,
    run_id=latest_run_id,
    model_name=model_name
)

# COMMAND ----------

from mlflow.models import validate_serving_input

# model_uri = 'runs:/c6f9a3f649334b708fed196bb1357595/two_tower_pyfunc'
model_uri = model_info.model_uri

# The model is logged with an input example. MLflow converts
# it into the serving payload format for the deployed model endpoint,
# and saves it to 'serving_input_payload.json'
serving_payload = """{
  "inputs": {
    "label": [
      0
    ],
    "movieId": [
      893
    ],
    "userId": [
      54
    ]
  }
}"""

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)

# COMMAND ----------

# MAGIC %md ### Deploy model to model serving endpoint using API

# COMMAND ----------

from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
    ServedModelInputWorkloadType,
    ServingEndpointDetailed,
    AutoCaptureConfigInput
)
from datetime import timedelta

# Create model serving configurations using Databricks SDK
wc = WorkspaceClient()

endpoint_name = "two-tower-torchrec-endpoint"

# for more information see https://databricks-sdk-py.readthedocs.io/en/latest/dbdataclasses/serving.html#databricks.sdk.service.serving
served_models = [
    ServedModelInput(
        model_name=model_name,
        model_version=model_info.registered_model_version,
        workload_type=ServedModelInputWorkloadType.GPU_SMALL,
        workload_size=ServedModelInputWorkloadSize.SMALL,
        scale_to_zero_enabled=True,
    )
]
auto_capture_config = AutoCaptureConfigInput(
        catalog_name=catalog,
        enabled=True,  # Enable inference tables
        schema_name=schema,
    )

endpoint_config = EndpointCoreConfigInput(served_models=served_models, auto_capture_config=auto_capture_config)

try:
    print(f"Creating endpoint {endpoint_name} with latest version...")
    wc.serving_endpoints.create_and_wait(
        endpoint_name, config=endpoint_config
    )
except Exception as e:
    if "already exists" in str(e):
        print(f"Endpoint exists, updating with latest model version...")
        wc.serving_endpoints.update_config_and_wait(
            endpoint_name, served_models=served_models, auto_capture_config=auto_capture_config
        )
    else:
        raise e

# COMMAND ----------

import requests
import json

sample_input = {
    "label": [0,1], 
    "movieId": [893,899], 
    "userId": [54,55]
}

# Get the API endpoint and token for the current notebook context
API_ROOT = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
)
API_TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

data = {"inputs": sample_input}
headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(
    url=f"{API_ROOT}/serving-endpoints/{endpoint_name}/invocations",
    json=data,
    headers=headers,
)

print(json.dumps(response.json()))
