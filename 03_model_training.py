# Databricks notebook source
# MAGIC %md Using 15.0 ML g5.24xlarge [A10] cluster with 1 worker node

# COMMAND ----------

# MAGIC %pip install -q --upgrade --no-deps --force-reinstall torch torchvision fbgemm-gpu torchrec --index-url https://download.pytorch.org/whl/cu118
# MAGIC %pip install -q torchmetrics==1.0.3 iopath pyre_extensions mosaicml-streaming
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

import os
import uuid
from typing import List, Optional
from streaming import StreamingDataset

import torch
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

# COMMAND ----------

# DBTITLE 1,Streaming paths
# Specify where the data will be stored
output_dir_train = config['output_dir_train']
output_dir_validation = config['output_dir_validation']
output_dir_test = config['output_dir_test']

# COMMAND ----------

# MAGIC %md ### Helper functions used throughout notebook
# MAGIC
# MAGIC Using TorchRec pipelines requires a pipelineable data type (which is Batch in this case). In this step, you create a helper function that takes each batch from the StreamingDataset and passes it through a transformation function to convert it into a pipelineable batch.
# MAGIC
# MAGIC For further context, see https://github.com/pytorch/torchrec/blob/main/torchrec/datasets/utils.py#L28.

# COMMAND ----------

# DBTITLE 1,Calculate user_ct and product_ct
import pandas as pd
import gc

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

def transpose_batch(batch):
    result = defaultdict(list)
    for d in batch:
        for k, v in d.items():
            result[k].append(v)
    return result    

def transform_to_torchrec_batch(batch, num_embeddings_per_feature: Optional[List[int]] = None) -> Batch:
    batch = transpose_batch(batch)

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

def add_random_string_to_path(base_path, length=8):
    # Generate a random string
    random_string = str(uuid.uuid4()).replace('-', '')[:length]
    
    # Extract the directory and file name from the base path
    directory, filename = os.path.split(base_path)
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    
    # Create the new filename with the random string
    new_filename = f"{name}_{random_string}{ext}"
    
    # Combine the directory and the new filename to form the new path
    new_path = os.path.join(directory, new_filename)
    
    return new_path

def get_dataloader_with_mosaic(remote_path, local_path, batch_size, label):
    print(f"Getting {label} data from UC Volumes")
    # dataset = StreamingDataset(local=path, split=None, shuffle=False, allow_unsafe_types=True, batch_size=batch_size)
    dataset = StreamingDataset(remote=remote_path, local=local_path, split=None, shuffle=False, allow_unsafe_types=True, batch_size=batch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=lambda x: x,
        drop_last=True
    )

transform_partial = partial(transform_to_torchrec_batch, num_embeddings_per_feature=emb_counts)

# COMMAND ----------

# MAGIC %md ## Creating the Relevant TorchRec code for Training
# MAGIC This section contains all of the training and evaluation code.

# COMMAND ----------

# MAGIC %md ### Two Tower Model Definition
# MAGIC This is taken directly from the torchrec example's page. Note that the loss is the Binary Cross Entropy loss, which requires labels to be within the values {0, 1}.

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

# COMMAND ----------

# MAGIC %md ### Base Dataclass for Training inputs
# MAGIC Feel free to modify any of the variables mentioned here, but note that the first layer for layer_sizes should be equivalent to embedding_dim.

# COMMAND ----------

from dataclasses import dataclass, field
import itertools 
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

# Store the results in mlflow
def get_relevant_fields(args, cat_cols, emb_counts):
    fields_to_save = ["epochs", "embedding_dim", "layer_sizes", "learning_rate", "batch_size"]
    result = { key: getattr(args, key) for key in fields_to_save }
    # add dense cols
    result["cat_cols"] = cat_cols
    result["emb_counts"] = emb_counts
    return result

# COMMAND ----------

# MAGIC %md ### Training and Evaluation Helper Functions:
# MAGIC
# MAGIC Helper Functions for Distributed Model Saving, Distributed Model Training, and Evaluation

# COMMAND ----------

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
                total_loss += _loss.detach()  # Detach _loss to prevent gradients from being calculated
                total_correct += (logits.round() == labels).sum().item()  # Count the number of correct predictions
                total_samples += len(labels)
                if is_rank_zero:
                    pbar.update(1)
            except StopIteration:
                break

    average_loss = total_loss / total_samples if total_samples > 0 else torch.tensor(0.0).to(device)
    average_loss_value = average_loss.item()

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    if is_rank_zero:
        print(f"Average loss over {stage} set: {average_loss_value:.4f}.")
        print(f"Accuracy over {stage} set: {accuracy*100:.2f}%")
    
    return average_loss_value, accuracy

# COMMAND ----------

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
    val_loss, val_accuracy = evaluate(args.limit_val_batches, pipeline, val_dataloader, "val", transform_partial)
    if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('val_loss', val_loss)
        mlflow.log_metric('val_accuracy', val_accuracy)

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
        val_loss, val_accuracy = evaluate(args.limit_val_batches, pipeline, val_dataloader, "val", transform_partial)
        if int(os.environ["RANK"]) == 0:
            mlflow.log_metric('val_loss', val_loss)
            mlflow.log_metric('val_accuracy', val_accuracy)

        # Save the underlying model and results to mlflow
        log_state_dict_to_mlflow(pipeline._model.module, artifact_path=f"model_state_dict_{epoch}")
    
    # Evaluate on the test set after training loop finishes
    test_loss, test_accuracy = evaluate(args.limit_test_batches, pipeline, test_dataloader, "test", transform_partial)
    if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('test_loss', test_loss)
        mlflow.log_metric('test_accuracy', test_accuracy)
    return test_loss

# COMMAND ----------

# MAGIC %md ### The main function
# MAGIC
# MAGIC This function trains the Two Tower recommendation model. For more information, see the following guides/docs/code:
# MAGIC
# MAGIC - https://pytorch.org/torchrec/
# MAGIC - https://github.com/pytorch/torchrec
# MAGIC - https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L75

# COMMAND ----------

output_dir_train = config['output_dir_train']
output_dir_validation = config['output_dir_validation']
output_dir_test = config['output_dir_test']

local_dir_train = "/local_disk0/" + output_dir_train.split("/")[-1] + "/"
local_dir_test = "/local_disk0/" + output_dir_test.split("/")[-1] + "/"
local_dir_validation = "/local_disk0/" + output_dir_validation.split("/")[-1] + "/"

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
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
    device = torch.device(f"cuda:{local_rank}")
    backend = "nccl"
    torch.cuda.set_device(device)

    # Start MLflow
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token
    experiment = mlflow.set_experiment(experiment_path)

    # Start distributed process group
    dist.init_process_group(backend=backend)

    # Loading the data
    import streaming.base.util as util

    util.clean_stale_shared_memory()
    local_dir_train_update = add_random_string_to_path(local_dir_train, length=8)
    local_dir_validation_update = add_random_string_to_path(local_dir_validation, length=8)
    local_dir_test_update = add_random_string_to_path(local_dir_test, length=8)
    try:
        train_dataloader = get_dataloader_with_mosaic(output_dir_train, local_dir_train_update, args.batch_size, "train")
        val_dataloader = get_dataloader_with_mosaic(output_dir_validation, local_dir_validation_update, args.batch_size, "val")
        test_dataloader = get_dataloader_with_mosaic(output_dir_test, local_dir_test_update, args.batch_size, "test")
    except FileExistsError:
        local_dir_train_update = add_random_string_to_path(local_dir_train, length=10)
        local_dir_validation_update = add_random_string_to_path(local_dir_validation, length=10)
        local_dir_test_update = add_random_string_to_path(local_dir_test, length=10)
        train_dataloader = get_dataloader_with_mosaic(output_dir_train, local_dir_train_update, args.batch_size, "train")
        val_dataloader = get_dataloader_with_mosaic(output_dir_validation, local_dir_validation_update, args.batch_size, "val")
        test_dataloader = get_dataloader_with_mosaic(output_dir_test, local_dir_test_update, args.batch_size, "test")

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
        storage_reservation=HeuristicalStorageReservation(percentage=0.15),
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
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
 
# Manually create the experiment so that you know the id and can send that to the worker nodes when you scale later.
experiment = mlflow.set_experiment(experiment_path)

# Log system metrics while training loop is running
mlflow.enable_system_metrics_logging()

# Automatically log per-epoch parameters, metrics, and checkpoint weights
# mlflow.pytorch.autolog(checkpoint_save_best_only = False)

# COMMAND ----------

# MAGIC %md ### Single Node + Single GPU Training
# MAGIC Here, you set the environment variables to run training over the sample set of 26M data points (stored in Volumes in Unity Catalog and collected using Mosaic StreamingDataset). You can expect each epoch to take ~16 minutes.

# COMMAND ----------

# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "29500"

# args = Args()
# main(args)

# COMMAND ----------

# MAGIC %md ### Single Node - Multi GPU Training
# MAGIC
# MAGIC This notebook uses TorchDistributor to handle training on a g5.24xlarge instance with 4 A10 GPUs. You can view the sharding plan in the output logs to see what tables are located on what GPUs. This takes ~8 minutes to run per epoch.
# MAGIC
# MAGIC Note: There may be cases where you receive unexpected errors (like the Python Kernel crashing or segmentation faults). This is a transient error and the easiest way to overcome it is to skip the single node single GPU training code before you run any distributed code (single node multi GPU or multi node multi GPU).
# MAGIC
# MAGIC Note: If you see any errors that are associated with Mosaic Data Loading, these are transient errors that can be overcome by rerunning the failed cell.

# COMMAND ----------

# args = Args(epochs=3, embedding_dim=128, layer_sizes=[128, 64], learning_rate=0.01, batch_size=1024, print_lr=True, train_on_sample=True)
# TorchDistributor(num_processes=4, local_mode=True, use_gpu=True).run(main, args)

# COMMAND ----------

# MAGIC %md ### Multi Node + Multi GPU Training
# MAGIC
# MAGIC This is tested with a g5.24xlarge instance with 4 A10 GPUs as a worker. You can view the sharding plan in the output logs to see what tables are located on what GPUs. This takes ~6 minutes to run per epoch.
# MAGIC
# MAGIC Note: There may be cases where you receive unexpected errors (like the Python Kernel crashing or segmentation faults). This is a transient error and the easiest way to overcome it is to skip the single node single GPU training code before you run any distributed code (single node multi GPU or multi node multi GPU).
# MAGIC
# MAGIC Note: If you see any errors that are associated with Mosaic Data Loading, these are transient errors that can be overcome by rerunning the failed cell.

# COMMAND ----------

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

# To Do: add try and except or retry logic for random spark failures
args = Args(
  epochs=10, 
  embedding_dim=128, 
  layer_sizes=[128, 64], 
  learning_rate=0.01, 
  batch_size=1024, 
  print_lr=False,
  validation_freq=1000
  )
  # limit_train_batches=100
  # )
TorchDistributor(num_processes=4, local_mode=False, use_gpu=True).run(main, args)
