# Databricks notebook source
# %pip install -r ./requirements.txt
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install mosaicml==0.26.0 mosaicml-streaming==0.7.5 torchmetrics databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Standard library imports
import os
import argparse
import shutil
import tempfile
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import gc
from dataclasses import dataclass, field
from enum import Enum

# Third-party machine learning and numerical libraries
import numpy as np
import mlflow

# PyTorch core
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler
from streaming import StreamingDataset, StreamingDataLoader, StreamingDataset, Stream
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.distributed._sharded_tensor import ShardedTensor
import torchmetrics

# Composer framework
import composer.models
from composer import Trainer
from composer.loggers import MLFlowLogger, InMemoryLogger
from composer.devices import Device
from composer.utils import get_device
from composer.callbacks import SpeedMonitor, SystemMetricsMonitor, ExportForInferenceCallback
from composer.optim import DecoupledAdamW

# Streaming data handling
import streaming
from streaming import StreamingDataset, Stream, StreamingDataLoader
import streaming.base.util as util

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

training_set = spark.table("training_set").toPandas()
user_ct = training_set['user_id'].nunique()
product_ct = training_set['product_id'].nunique()

# embedding columns and counts
cat_cols = ["user_id", "product_id"]
emb_counts = [user_ct, product_ct]
emb_counts = [user_ct, 49688]
print(emb_counts)

# Delete dataframes to free up memory
del training_set
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

training_set['product_id'].max()

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()['current_user()']
print(username)

experiment_path = f'/Users/{username}/pytorch-two-tower/'

# We will need these later
db_host = os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# We manually create the experiment so that we know the id and can send that to the worker nodes when we scale
experiment = mlflow.set_experiment(experiment_path)

# COMMAND ----------

import uuid

def get_dataloader_with_mosaic(path, batch_size, label, write_to_local=True):

    if write_to_local:
        random_uuid = uuid.uuid4()
        local_path = f"/local_disk0/mds/{label}/{random_uuid}"
        print(f"Getting {label} data from UC Volumes at {path} and saving to {local_path}")
        dataset = StreamingDataset(remote=path, local=local_path, shuffle=False, batch_size=batch_size)
    else:
        dataset = StreamingDataset(local=path, shuffle=False, batch_size=batch_size)
    return StreamingDataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=4)

# COMMAND ----------

test = get_dataloader_with_mosaic(config['output_dir_train'], batch_size=100, label="train")

# COMMAND ----------

for i, batch in enumerate(test):
  print(batch)
  if i > 2:
    break

# COMMAND ----------

def log_checkpoints_to_mlflow(run_id, local_checkpoint_dir):
    import os
    import mlflow
    # Set up MLflow client
    client = mlflow.tracking.MlflowClient()

    # Get all .pt files
    pt_files = [f for f in os.listdir(local_checkpoint_dir) if f.endswith('.pt')]

    # Log file sizes (optional)
    for f in pt_files:
        file_path = os.path.join(local_checkpoint_dir, f)
        file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
        print(f"File: {f}, Size: {file_size_gb:.2f} GB")

    # Check if there's an active run with the given run_id
    try:
        if mlflow.active_run():
            run_context = mlflow.active_run()
        else:
            mlflow.start_run(run_id=run_id)
    except Exception as e:
        print(f"Error checking active run: {e}")
        print(f"Starting new run with ID: {run_id}")
        run_context = mlflow.start_run(run_id=run_id)

    # Log artifacts to MLflow
    with run_context:
        for f in pt_files:
            local_path = os.path.join(local_checkpoint_dir, f)
            artifact_path = "checkpoints"
            client.log_artifact(run_id, local_path, artifact_path)
            print(f"Logged {f} to MLflow artifacts")

    print("Finished logging checkpoints to MLflow")

def get_run_id_by_name(experiment_id, run_name):

    import mlflow
    
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")
    if runs:
        return runs[0].info.run_id
    else:
        return None

# COMMAND ----------

# from dataclasses import dataclass
# from typing import Dict, Optional, List

# @dataclass
# class ModelConfig:
#     """Configuration for TwoTower model architecture"""
#     num_users: int
#     num_items: int
#     embedding_dim: int = 128
#     hidden_dims: List[int] = (128, 64)
#     dropout_rate: float = 0.2
#     user_key: str = 'user_id'
#     item_key: str = 'product_id'
#     label_key: str = 'label'

# # class TowerModule(nn.Module):
# class TowerModule(composer.models.ComposerModel):
#     """Generic tower architecture for embedding and processing"""
#     def __init__(self, num_items: int, embedding_dim: int, hidden_dims: List[int], dropout_rate: float):
#         super().__init__()
        
        # self.embedding = nn.Embedding(
        #     num_embeddings=num_items,
        #     embedding_dim=embedding_dim,
        #     sparse=False
        # )
        
        # layers = []
        # input_dim = embedding_dim
        
        # for hidden_dim in hidden_dims:
        #     layers.extend([
        #         nn.Linear(input_dim, hidden_dim),
        #         nn.ReLU(inplace=False),
        #         nn.BatchNorm1d(hidden_dim),
        #         nn.Dropout(p=dropout_rate, inplace=False)
        #     ])
        #     input_dim = hidden_dim
            
        # self.network = nn.Sequential(*layers)
    
#     def forward(self, x):
#         # Ensure input is on the same device as the embedding layer
#         x = x.to(self.embedding.weight.device)
#         x = self.embedding(x)
#         return self.network(x)

# # class TwoTowerBase(nn.Module):
# class TwoTowerBase(composer.models.ComposerModel):
#     """Base two-tower architecture"""
#     def __init__(self, config: ModelConfig):
#         super().__init__()
        
#         self.config = config
        
#         # Create towers
#         self.user_tower = TowerModule(
#             config.num_users,
#             config.embedding_dim,
#             config.hidden_dims,
#             config.dropout_rate
#         )
        
#         self.item_tower = TowerModule(
#             config.num_items,
#             config.embedding_dim,
#             config.hidden_dims,
#             config.dropout_rate
#         )
        
#         # Final prediction layer
#         final_dim = config.hidden_dims[-1]
#         self.predictor = nn.Linear(2 * final_dim, 1)

#     def to(self, device):
#         # Explicitly move all components to the specified device
#         self.user_tower = self.user_tower.to(device)
#         self.item_tower = self.item_tower.to(device)
#         self.predictor = self.predictor.to(device)
#         return self
    
#     def _get_batch_data(self, batch):
#         # Ensure batch data is on the correct device
#         device = next(self.parameters()).device
#         return (
#             batch[self.config.user_key].to(device),
#             batch[self.config.item_key].to(device),
#             batch[self.config.label_key].to(device)
#         )
    
#     def forward(self, batch):
#         users, items, _ = self._get_batch_data(batch)
        
#         user_embedding = self.user_tower(users)
#         item_embedding = self.item_tower(items)
        
#         combined = torch.cat([user_embedding, item_embedding], dim=1)
#         return self.predictor(combined).squeeze(-1)

# class TwoTowerComposerModel(composer.models.ComposerModel):
#     """Composer wrapper for TwoTower model"""
#     def __init__(
#         self,
#         config: ModelConfig,
#         metrics: Optional[Dict] = None
#     ):
#         super().__init__()
        
#         self.model = TwoTowerBase(config)
#         self.loss_fn = nn.BCEWithLogitsLoss()
#         self._init_metrics(metrics)
    
#     def _init_metrics(self, metrics):
#         if metrics is None:
#             metrics = {
#                 'auroc': torchmetrics.AUROC(task='binary'),
#                 'f1': torchmetrics.F1Score(task='binary')
#             }
        
#         self.train_metrics = torchmetrics.MetricCollection(
#             {f'train_{k}': v.clone() for k, v in metrics.items()}
#         )
#         self.val_metrics = torchmetrics.MetricCollection(
#             {f'val_{k}': v.clone() for k, v in metrics.items()}
#         )
    
#     def forward(self, batch):
#         return self.model(batch)
    
#     def loss(self, outputs, batch):
#         _, _, labels = self.model._get_batch_data(batch)
#         return self.loss_fn(outputs, labels.float())
    
#     def update_metric(self, batch, outputs, metric):
#         _, _, labels = self.model._get_batch_data(batch)
#         outputs = torch.sigmoid(outputs.detach())
#         metric.update(outputs.float(), labels.float())
    
#     def get_metrics(self, is_train=False):
#         return self.train_metrics if is_train else self.val_metrics


# COMMAND ----------

# from dataclasses import dataclass
# from typing import Dict, Optional, List, Union
# import torch
# import torch.nn as nn
# import torchmetrics
# from composer.models import ComposerModel

# @dataclass
# class ModelConfig:
#     """Configuration for TwoTower model architecture"""
#     num_users: int
#     num_items: int
#     embedding_dim: int = 128
#     hidden_dim: int = 128
#     dropout_rate: float = 0.2
#     user_key: str = 'user_id'
#     item_key: str = 'product_id'
#     label_key: str = 'label'

# class TwoTowerComposerModel(ComposerModel):
#     def __init__(
#         self,
#         config: Union[ModelConfig, Dict],
#         metrics: Optional[Dict] = None
#     ):
#         super().__init__()
        
#         # Convert dict to ModelConfig if necessary
#         if isinstance(config, dict):
#             config = ModelConfig(**config)
        
#         # Store configuration
#         self.config = config
        
#         # Embedding layers (no sparse gradients)
#         self.user_embedding = nn.Embedding(
#             num_embeddings=config.num_users,
#             embedding_dim=config.embedding_dim,
#             sparse=False
#         )
#         self.item_embedding = nn.Embedding(
#             num_embeddings=config.num_items,
#             embedding_dim=config.embedding_dim,
#             sparse=False
#         )
        
#         # Hidden layers
#         self.fc1 = nn.Linear(2 * config.embedding_dim, config.hidden_dim)
#         self.fc2 = nn.Linear(config.hidden_dim, 1)
        
#         # Dropout and activation (avoid inplace operations)
#         self.dropout = nn.Dropout(p=config.dropout_rate, inplace=False)
#         self.relu = nn.ReLU(inplace=False)
        
#         # Loss function
#         self.loss_fn = nn.BCEWithLogitsLoss()

#         # Initialize metrics
#         self._init_metrics(metrics)

#     def _init_metrics(self, metrics):
#         if metrics is None:
#             # Default metrics
#             metrics = {
#                 'accuracy': torchmetrics.Accuracy(task='binary'),
#                 'auroc': torchmetrics.AUROC(task='binary'),
#                 'f1': torchmetrics.F1Score(task='binary')
#             }
        
#         # Create separate collections for train and validation
#         self.train_metrics = torchmetrics.MetricCollection(
#             {f'train_{k}': v.clone() for k, v in metrics.items()}
#         )
#         self.val_metrics = torchmetrics.MetricCollection(
#             {f'val_{k}': v.clone() for k, v in metrics.items()}
#         )

#     def _get_batch_data(self, batch):
#         """Extract data safely"""
#         users = batch[self.config.user_key]
#         items = batch[self.config.item_key]
#         labels = batch[self.config.label_key]
#         return users, items, labels

#     def forward(self, batch):
#         try:
#             users, items, _ = self._get_batch_data(batch)
            
#             # Get embeddings
#             user_embedded = self.user_embedding(users)
#             item_embedded = self.item_embedding(items)
            
#             # Concatenate embeddings (create new tensor)
#             combined = torch.cat([user_embedded, item_embedded], dim=1)
            
#             # Forward pass (avoid inplace operations)
#             x = self.fc1(combined)
#             x = self.relu(x)  # Non-inplace ReLU
#             x = self.dropout(x)  # Non-inplace dropout
#             logits = self.fc2(x).squeeze(-1)
            
#             return logits
            
#         except Exception as e:
#             print(f"Error in forward pass: {str(e)}")
#             raise

#     def loss(self, outputs, batch):
#         _, _, labels = self._get_batch_data(batch)
#         return self.loss_fn(outputs, labels.float())
    
#     def update_metric(self, batch, outputs, metric):
#         try:
#             _, _, labels = self._get_batch_data(batch)
            
#             # Ensure outputs are probabilities
#             if outputs.requires_grad:
#                 outputs = torch.sigmoid(outputs.detach())
            
#             metric.update(outputs, labels)
            
#         except Exception as e:
#             print(f"Error in update_metric: {str(e)}")
#             print(f"Outputs: min={outputs.min()}, max={outputs.max()}, requires_grad={outputs.requires_grad}")
#             print(f"Labels: min={labels.min()}, max={labels.max()}, dtype={labels.dtype}")
#             raise

#     def get_metrics(self, is_train=False):
#         return self.train_metrics if is_train else self.val_metrics

# COMMAND ----------

from dataclasses import dataclass
from typing import Dict, Optional, List, Union
import torch
import torch.nn as nn
import torchmetrics
from composer.models import ComposerModel

@dataclass
class ModelConfig:
    """Configuration for TwoTower model architecture"""
    num_users: int
    num_items: int
    embedding_dim: int = 128
    hidden_dims: List[int] = (128, 64)  # List of hidden dimensions
    dropout_rate: float = 0.2
    user_key: str = 'user_id'
    item_key: str = 'product_id'
    label_key: str = 'label'

class TwoTowerComposerModel(ComposerModel):
    def __init__(
        self,
        config: Union[ModelConfig, Dict],
        metrics: Optional[Dict] = None
    ):
        super().__init__()
        
        # Convert dict to ModelConfig if necessary
        if isinstance(config, dict):
            config = ModelConfig(**config)
        
        # Store configuration
        self.config = config
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Build network architecture
        self.layers = self._build_network()
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize metrics
        self._init_metrics(metrics)

    def _init_embeddings(self):
        """Initialize embedding layers"""
        self.user_embedding = nn.Embedding(
            num_embeddings=self.config.num_users,
            embedding_dim=self.config.embedding_dim,
            sparse=False
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=self.config.num_items,
            embedding_dim=self.config.embedding_dim,
            sparse=False
        )

    def _build_network(self) -> nn.ModuleList:
        """Build dynamic network architecture"""
        layers = nn.ModuleList()
        input_dim = 2 * self.config.embedding_dim  # Combined embedding dimension
        
        # Build hidden layers
        for hidden_dim in self.config.hidden_dims:
            block = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=False),
                nn.Dropout(p=self.config.dropout_rate, inplace=False)
            )
            layers.append(block)
            input_dim = hidden_dim
        
        # Add final output layer
        layers.append(nn.Linear(input_dim, 1))
        
        return layers

    def _init_metrics(self, metrics):
        """Initialize model metrics"""
        if metrics is None:
            metrics = {
                'accuracy': torchmetrics.Accuracy(task='binary'),
                'auroc': torchmetrics.AUROC(task='binary'),
                'f1': torchmetrics.F1Score(task='binary')
            }
        
        self.train_metrics = torchmetrics.MetricCollection(
            {f'train_{k}': v.clone() for k, v in metrics.items()}
        )
        self.val_metrics = torchmetrics.MetricCollection(
            {f'val_{k}': v.clone() for k, v in metrics.items()}
        )

    def _get_batch_data(self, batch):
        """Extract and validate batch data"""
        try:
            users = batch[self.config.user_key]
            items = batch[self.config.item_key]
            labels = batch[self.config.label_key]
            
            # Validate indices
            if torch.any(users >= self.config.num_users) or torch.any(users < 0):
                raise ValueError("User indices out of bounds")
            if torch.any(items >= self.config.num_items) or torch.any(items < 0):
                raise ValueError("Item indices out of bounds")
                
            return users, items, labels
        except KeyError as e:
            raise KeyError(f"Missing key in batch: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error processing batch data: {str(e)}")

    def forward(self, batch):
        try:
            users, items, _ = self._get_batch_data(batch)
            
            # Get embeddings
            user_embedded = self.user_embedding(users)
            item_embedded = self.item_embedding(items)
            
            # Concatenate embeddings
            x = torch.cat([user_embedded, item_embedded], dim=1)
            
            # Forward pass through hidden layers
            for layer in self.layers[:-1]:
                x = layer(x)
            
            # Final layer
            logits = self.layers[-1](x).squeeze(-1)
            
            return logits
            
        except Exception as e:
            raise RuntimeError(f"Forward pass error: {str(e)}")

    def loss(self, outputs, batch):
        try:
            _, _, labels = self._get_batch_data(batch)
            return self.loss_fn(outputs, labels.float())
        except Exception as e:
            raise RuntimeError(f"Loss computation error: {str(e)}")
    
    def update_metric(self, batch, outputs, metric):
        try:
            _, _, labels = self._get_batch_data(batch)
            
            # Ensure outputs are probabilities
            if outputs.requires_grad:
                outputs = torch.sigmoid(outputs.detach())
            
            # Convert labels to float
            labels = labels.float()
            
            # Update metric
            metric.update(outputs, labels)
            
        except Exception as e:
            raise RuntimeError(f"Metric update error: {str(e)}")

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

# COMMAND ----------

def main_fn(model_config):
    import mlflow
    import torch.distributed as dist

    import streaming.base.util as util
    from composer.utils import get_device
    import streaming
    from streaming import StreamingDataset, Stream, StreamingDataLoader

    # set environment variables for Databricks and TMPDIR for mlflow_logger
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token
    experiment = mlflow.set_experiment(experiment_path)

    print("Running distributed training")
    # Initialize distributed training
    dist.init_process_group("nccl")
    # local_rank = int(os.environ["LOCAL_RANK"])
    # NUM_GPUS_PER_NODE = torch.cuda.device_count()
    # os.environ["LOCAL_WORLD_SIZE"] = str(NUM_GPUS_PER_NODE)
    # device = torch.device(f"cuda:{local_rank}")
    # torch.cuda.set_device(device)

    # mosaic streaming recommendations
    util.clean_stale_shared_memory()
    composer.utils.dist.initialize_dist(get_device(None))

    # get dataloaders
    train_dataloader = get_dataloader_with_mosaic(
      '/Volumes/main/alex_m/instacart_data/two_tower/mds_train_update', batch_size=32, label="train", write_to_local=False)
    eval_dataloader = get_dataloader_with_mosaic(
      '/Volumes/main/alex_m/instacart_data/two_tower/mds_validation_update', batch_size=32, label="validation", write_to_local=False)
    
    # Create model and move to GPU
    model = TwoTowerComposerModel(config=model_config)

    # Optimizer and scheduler
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
        betas=(0.9, 0.95)
    )

    linear_lr_decay = LinearLR(
        optimizer, start_factor=1.0,
        end_factor=0, total_iters=150
    )
    
    mlflow_logger = composer.loggers.MLFlowLogger(
        experiment_name=experiment_path,
        synchronous=True,
        resume=True,
        tracking_uri="databricks"
    )

    save_folder = '/local_disk0/composer-training/checkpoints'

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        schedulers=linear_lr_decay,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration='1ep',
        eval_interval='100ba',
        train_subset_num_batches=20,
        eval_subset_num_batches=10,
        save_folder=save_folder,
        save_overwrite=True,
        device="gpu",
        loggers=[mlflow_logger]
    )

    # Start training
    trainer.fit()

    # get run_name from mlflow_logger and log checkpoints to mlflow experiment/run
    rank_of_gpu = os.environ["RANK"]
    run_name = mlflow_logger.run_name
    # run_id = get_run_id_by_name(experiment.experiment_id, run_name)

    if rank_of_gpu == "0":
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")
        if runs:
            run_id = runs[0].info.run_id
            # Get all .pt files
            pt_files = [f for f in os.listdir(save_folder) if f.endswith('.pt')]
            for f in pt_files:
                local_path = os.path.join(save_folder, f)
                artifact_path = "checkpoints"
                client.log_artifact(run_id, local_path, artifact_path)
                print(f"Logged {f} to MLflow artifacts")

        print("Finished logging checkpoints to MLflow")

    return trainer.saved_checkpoints


# COMMAND ----------

# MAGIC %md ### Run single-node/local training

# COMMAND ----------

# Usage example
model_config = ModelConfig(
    num_users=user_ct,
    num_items=product_ct,
    embedding_dim=128,
    hidden_dim=128,
    dropout_rate=0.2
)

model = TwoTowerComposerModel(config=model_config)

# Optimizer and scheduler
optimizer = DecoupledAdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5,
    betas=(0.9, 0.95)
)

linear_lr_decay = LinearLR(
    optimizer, start_factor=1.0,
    end_factor=0, total_iters=150
)

util.clean_stale_shared_memory()
composer.utils.dist.initialize_dist(get_device(None))

# get dataloaders
train_dataloader = get_dataloader_with_mosaic(config['output_dir_train'], batch_size=32, label="train")
eval_dataloader = get_dataloader_with_mosaic(config['output_dir_train'], batch_size=32, label="validation")

mlflow_logger = composer.loggers.MLFlowLogger(
  experiment_name=experiment_path,
  log_system_metrics=True,
  resume=True,
  tracking_uri="databricks")

# checkpoint paths
save_overwrite=True
save_folder='/local_disk0/composer-training/checkpoints'

# Initialize trainer
trainer = Trainer(
    model=model,
    optimizers=optimizer,
    schedulers=linear_lr_decay,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    eval_interval='100ba',
    train_subset_num_batches=1,
    eval_subset_num_batches=1,
    save_folder=save_folder,
    save_overwrite=save_overwrite,
    device="gpu",
    loggers=[mlflow_logger]
)

# Start training
trainer.fit()

# COMMAND ----------

# MAGIC %md ### Run single-node + multi-GPU

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

import os
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"

model_config = ModelConfig(
    num_users=user_ct,
    num_items=49688,
    embedding_dim=128,
    hidden_dims=[128],
    dropout_rate=0.2
)

trainer_checkpoints = TorchDistributor(num_processes=4, local_mode=True, use_gpu=True).run(main_fn, model_config)

# COMMAND ----------

# MAGIC %md ### Run multi-node + multi-GPU

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

# Usage example
model_config = ModelConfig(
    num_users=user_ct,
    num_items=product_ct,
    embedding_dim=128,
    hidden_dims=[128, 64],
    dropout_rate=0.2
)


trainer_checkpoints = TorchDistributor(num_processes=8, local_mode=False, use_gpu=True).run(main_fn, model_config)
