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

def get_dataloader_with_mosaic(path, batch_size, label):
    print(f"Getting {label} data from UC Volumes")
    random_uuid = uuid.uuid4()
    dataset = StreamingDataset(remote=path, local=f"/local_disk0/{random_uuid}", shuffle=True, batch_size=batch_size)
    # dataset = StreamingDataset(local=path, shuffle=True, batch_size=batch_size)
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