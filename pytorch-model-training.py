# Databricks notebook source
# %pip install -r ./requirements.txt
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install mosaicml==0.26.0 mosaicml-streaming==0.9.0 torchmetrics databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Standard library imports
import os
import uuid
import gc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

# Third-party machine learning and numerical libraries
import numpy as np
import mlflow

# PyTorch core
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.distributed.optim import _apply_optimizer_in_backward as apply_optimizer_in_backward
from torch.distributed._sharded_tensor import ShardedTensor
import torchmetrics

# PySpark
from pyspark.ml.torch.distributor import TorchDistributor

# Composer framework
import composer.models
from composer import Trainer
from composer.loggers import MLFlowLogger, InMemoryLogger
from composer.devices import Device
from composer.utils import get_device
from composer.callbacks import SpeedMonitor, SystemMetricsMonitor, ExportForInferenceCallback
from composer.optim import DecoupledAdamW
from composer.models import ComposerModel

# Streaming data handling
import streaming
from streaming import StreamingDataset, Stream, StreamingDataLoader
import streaming.base.util as util

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

training_set = spark.table("training_set").toPandas()
user_ct = training_set['user_id'].nunique()
product_ct = training_set['product_id'].max() if training_set['product_id'].max() > training_set['product_id'].nunique() else training_set['product_id'].nunique()

# embedding columns and counts
cat_cols = ["user_id", "product_id"]
emb_counts = [user_ct, product_ct]
print(emb_counts)

# Delete dataframes to free up memory
del training_set
gc.collect()
torch.cuda.empty_cache()

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

# DBTITLE 1,Utility Functions
def get_dataloader_with_mosaic(path, batch_size, label, write_to_local=True, num_workers=4, shuffle=True):

    if write_to_local:
        random_uuid = uuid.uuid4()
        local_path = f"/local_disk0/mds/{label}/{random_uuid}"
        print(f"Getting {label} data from UC Volumes at {path} and saving to {local_path}")
        dataset = StreamingDataset(remote=path, local=local_path, shuffle=shuffle, batch_size=batch_size)
    else:
        dataset = StreamingDataset(local=path, shuffle=shuffle, batch_size=batch_size)
    return StreamingDataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers)

def get_run_id(experiment_id: str, run_name: str) -> Optional[str]:
    """
    Get MLflow run ID using experiment ID and run name.
    
    Args:
        experiment_id: MLflow experiment ID
        run_name: Name of the run to search for
        
    Returns:
        Optional[str]: Run ID if found, None otherwise
    """
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    
    return runs[0].info.run_id if runs else None

def log_checkpoints(save_folder: str, run_id: str) -> None:
    """
    Log model checkpoints to MLflow artifacts.
    
    Args:
        save_folder: Path to folder containing checkpoint files
        run_id: MLflow run ID to log artifacts to
    """
    client = mlflow.tracking.MlflowClient()
    pt_files = [f for f in os.listdir(save_folder) if f.endswith('.pt')]
    
    for f in pt_files:
        local_path = os.path.join(save_folder, f)
        client.log_artifact(run_id, local_path, "checkpoints")
        print(f"Logged {f} to MLflow artifacts")
        
    print("Finished logging checkpoints to MLflow")

# COMMAND ----------

test = get_dataloader_with_mosaic(config['output_dir_train'], batch_size=100, label="train")

# COMMAND ----------

for i, batch in enumerate(test):
  print(batch)
  if i > 2:
    break

# COMMAND ----------

# MAGIC %md ### Dataclasses for ModelConfig, DataConfig, and TrainingConfing

# COMMAND ----------

@dataclass
class ModelConfig:
    """Configuration for TwoTower model architecture"""
    num_users: int
    num_items: int
    embedding_dim: int = 128
    hidden_dims: List[int] = (128, 64)
    dropout_rate: float = 0.2
    user_key: str = 'user_id'
    item_key: str = 'product_id'
    label_key: str = 'label'

@dataclass
class DataConfig:
    """Configuration for data loading"""
    train_path: Union[str, Path]
    eval_path: Union[str, Path]
    batch_size: int = 32
    write_to_local: bool = False
    num_workers: int = 4
    shuffle: bool = True

@dataclass
class TrainerConfig:
    """Configuration for Composer Trainer"""
    max_duration: str = '1ep'
    eval_interval: str = '100ba'
    train_subset_num_batches: Optional[int] = 20  # -1 to run max_duration
    eval_subset_num_batches: Optional[int] = 10   # -1 to run eval_interval
    save_folder: str = '/local_disk0/composer-training/checkpoints'
    save_overwrite: bool = True
    device: str = 'gpu'
    
    # Optimizer configs
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Scheduler configs
    scheduler_start_factor: float = 1.0
    scheduler_end_factor: float = 0
    scheduler_total_iters: int = 150

# COMMAND ----------

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
        
#         # Initialize embeddings
#         self._init_embeddings()
        
#         # Build network architecture
#         self.layers = self._build_network()
        
#         # Loss function
#         self.loss_fn = nn.BCEWithLogitsLoss()

#         # Initialize metrics
#         self._init_metrics(metrics)

#     def _init_embeddings(self):
#         """Initialize embedding layers"""
#         self.user_embedding = nn.Embedding(
#             num_embeddings=self.config.num_users,
#             embedding_dim=self.config.embedding_dim,
#             sparse=False
#         )
#         self.item_embedding = nn.Embedding(
#             num_embeddings=self.config.num_items,
#             embedding_dim=self.config.embedding_dim,
#             sparse=False
#         )

#     def _build_network(self) -> nn.ModuleList:
#         """Build dynamic network architecture"""
#         layers = nn.ModuleList()
#         input_dim = 2 * self.config.embedding_dim  # Combined embedding dimension
        
#         # Build hidden layers
#         for hidden_dim in self.config.hidden_dims:
#             block = nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 nn.ReLU(inplace=False),
#                 nn.Dropout(p=self.config.dropout_rate, inplace=False)
#             )
#             layers.append(block)
#             input_dim = hidden_dim
        
#         # Add final output layer
#         layers.append(nn.Linear(input_dim, 1))
        
#         return layers

#     def _init_metrics(self, metrics):
#         """Initialize model metrics"""
#         if metrics is None:
#             metrics = {
#                 'accuracy': torchmetrics.Accuracy(task='binary'),
#                 'auroc': torchmetrics.AUROC(task='binary'),
#                 'f1': torchmetrics.F1Score(task='binary')
#             }
        
#         self.train_metrics = torchmetrics.MetricCollection(
#             {f'train_{k}': v.clone() for k, v in metrics.items()}
#         )
#         self.val_metrics = torchmetrics.MetricCollection(
#             {f'val_{k}': v.clone() for k, v in metrics.items()}
#         )

#     def _get_batch_data(self, batch):
#         """Extract and validate batch data"""
#         try:
#             users = batch[self.config.user_key]
#             items = batch[self.config.item_key]
#             labels = batch[self.config.label_key]
            
#             # Validate indices
#             if torch.any(users >= self.config.num_users) or torch.any(users < 0):
#                 raise ValueError("User indices out of bounds")
#             if torch.any(items >= self.config.num_items) or torch.any(items < 0):
#                 raise ValueError("Item indices out of bounds")
                
#             return users, items, labels
#         except KeyError as e:
#             raise KeyError(f"Missing key in batch: {str(e)}")
#         except Exception as e:
#             raise RuntimeError(f"Error processing batch data: {str(e)}")

#     def forward(self, batch):
#         try:
#             users, items, _ = self._get_batch_data(batch)
            
#             # Get embeddings
#             user_embedded = self.user_embedding(users)
#             item_embedded = self.item_embedding(items)
            
#             # Concatenate embeddings
#             x = torch.cat([user_embedded, item_embedded], dim=1)
            
#             # Forward pass through hidden layers
#             for layer in self.layers[:-1]:
#                 x = layer(x)
            
#             # Final layer
#             logits = self.layers[-1](x).squeeze(-1)
            
#             return logits
            
#         except Exception as e:
#             raise RuntimeError(f"Forward pass error: {str(e)}")

#     def loss(self, outputs, batch):
#         try:
#             _, _, labels = self._get_batch_data(batch)
#             return self.loss_fn(outputs, labels.float())
#         except Exception as e:
#             raise RuntimeError(f"Loss computation error: {str(e)}")
    
#     def update_metric(self, batch, outputs, metric):
#         try:
#             _, _, labels = self._get_batch_data(batch)
            
#             # Ensure outputs are probabilities
#             if outputs.requires_grad:
#                 outputs = torch.sigmoid(outputs.detach())
            
#             # Convert labels to float
#             labels = labels.float()
            
#             # Update metric
#             metric.update(outputs, labels)
            
#         except Exception as e:
#             raise RuntimeError(f"Metric update error: {str(e)}")

#     def get_metrics(self, is_train=False):
#         return self.train_metrics if is_train else self.val_metrics

# COMMAND ----------

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

    def _get_input_features(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and validate input features.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Tuple of (user_ids, item_ids)
        """
        try:
            users = batch[self.config.user_key]
            items = batch[self.config.item_key]
            
            # Move to same device as model
            device = next(self.parameters()).device
            users = users.to(device)
            items = items.to(device)
            
            # # Validate indices
            # if torch.any(users >= self.config.num_users) or torch.any(users < 0):
            #     raise ValueError(f"User indices must be in range [0, {self.config.num_users})")
            # if torch.any(items >= self.config.num_items) or torch.any(items < 0):
            #     raise ValueError(f"Item indices must be in range [0, {self.config.num_items})")
                
            return users, items
            
        except KeyError as e:
            raise KeyError(f"Missing required input key: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error processing input features: {str(e)}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for computing embeddings and predictions.
        
        Args:
            batch: Dictionary containing input tensors
            
        Returns:
            Model output logits
        """
        try:
            # Get input features
            users, items = self._get_input_features(batch)
            
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

    def loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for training"""
        try:
            labels = batch[self.config.label_key].float()
            return self.loss_fn(outputs, labels)
        except Exception as e:
            raise RuntimeError(f"Loss computation error: {str(e)}")
    
    def update_metric(self, batch: Dict[str, torch.Tensor], 
                     outputs: torch.Tensor, 
                     metric: torchmetrics.Metric) -> None:
        """Update metrics during training/validation"""
        try:
            labels = batch[self.config.label_key].float()
            
            # Ensure outputs are probabilities
            if outputs.requires_grad:
                outputs = torch.sigmoid(outputs.detach())
            
            # Update metric
            metric.update(outputs, labels)
            
        except Exception as e:
            raise RuntimeError(f"Metric update error: {str(e)}")

    def get_metrics(self, is_train: bool = False) -> torchmetrics.MetricCollection:
        """Get appropriate metrics collection"""
        return self.train_metrics if is_train else self.val_metrics

# COMMAND ----------

def main_fn(model_config: ModelConfig, 
            data_config: DataConfig, 
            trainer_config: TrainerConfig):
  
    import mlflow
    import torch.distributed as dist
    from composer.utils import get_device
    import streaming
    
    # Set environment variables
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token
    experiment = mlflow.set_experiment(experiment_path)

    # Initialize distributed training
    dist.init_process_group("nccl")
    composer.utils.dist.initialize_dist(get_device(None))
    
    # Get dataloaders
    train_dataloader = get_dataloader_with_mosaic(
        data_config.train_path, 
        batch_size=data_config.batch_size,
        label="train",
        write_to_local=data_config.write_to_local,
        num_workers=data_config.num_workers,
        shuffle=data_config.shuffle
    )
    
    eval_dataloader = get_dataloader_with_mosaic(
        data_config.eval_path,
        batch_size=data_config.batch_size,
        label="validation",
        write_to_local=data_config.write_to_local,
        num_workers=data_config.num_workers,
        shuffle=False
    )
    
    # Create model
    model = TwoTowerComposerModel(config=model_config)

    # Optimizer and scheduler
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        betas=(trainer_config.beta1, trainer_config.beta2)
    )

    scheduler = LinearLR(
        optimizer,
        start_factor=trainer_config.scheduler_start_factor,
        end_factor=trainer_config.scheduler_end_factor,
        total_iters=trainer_config.scheduler_total_iters
    )
    
    # MLflow logger
    mlflow_logger = composer.loggers.MLFlowLogger(
        experiment_name=experiment_path,
        synchronous=True,
        resume=True,
        tracking_uri="databricks"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        schedulers=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration=trainer_config.max_duration,
        eval_interval=trainer_config.eval_interval,
        train_subset_num_batches=trainer_config.train_subset_num_batches,
        eval_subset_num_batches=trainer_config.eval_subset_num_batches,
        save_folder=trainer_config.save_folder,
        save_overwrite=trainer_config.save_overwrite,
        device=trainer_config.device,
        loggers=[mlflow_logger]
    )

    # Training and checkpoint logging
    trainer.fit()

    run_id = get_run_id(experiment.experiment_id, mlflow_logger.run_name)

    rank_of_gpu = os.environ["RANK"]
    if rank_of_gpu == "0":
        log_checkpoints(trainer_config.save_folder, run_id)
    
    return run_id

# COMMAND ----------

# MAGIC %md ### Run single-node/local training

# COMMAND ----------

# # Usage example
# model_config = ModelConfig(
#     num_users=user_ct,
#     num_items=product_ct,
#     embedding_dim=128,
#     hidden_dim=128,
#     dropout_rate=0.2
# )

# model = TwoTowerComposerModel(config=model_config)

# # Optimizer and scheduler
# optimizer = DecoupledAdamW(
#     model.parameters(),
#     lr=0.001,
#     weight_decay=1e-5,
#     betas=(0.9, 0.95)
# )

# linear_lr_decay = LinearLR(
#     optimizer, start_factor=1.0,
#     end_factor=0, total_iters=150
# )

# util.clean_stale_shared_memory()
# composer.utils.dist.initialize_dist(get_device(None))

# # get dataloaders
# train_dataloader = get_dataloader_with_mosaic(config['output_dir_train'], batch_size=32, label="train")
# eval_dataloader = get_dataloader_with_mosaic(config['output_dir_train'], batch_size=32, label="validation")

# mlflow_logger = composer.loggers.MLFlowLogger(
#   experiment_name=experiment_path,
#   log_system_metrics=True,
#   resume=True,
#   tracking_uri="databricks")

# # checkpoint paths
# save_overwrite=True
# save_folder='/local_disk0/composer-training/checkpoints'

# # Initialize trainer
# trainer = Trainer(
#     model=model,
#     optimizers=optimizer,
#     schedulers=linear_lr_decay,
#     train_dataloader=train_dataloader,
#     eval_dataloader=eval_dataloader,
#     max_duration='1ep',
#     eval_interval='100ba',
#     train_subset_num_batches=1,
#     eval_subset_num_batches=1,
#     save_folder=save_folder,
#     save_overwrite=save_overwrite,
#     device="gpu",
#     loggers=[mlflow_logger]
# )

# # Start training
# trainer.fit()

# COMMAND ----------

# MAGIC %md ### Run single-node + multi-GPU

# COMMAND ----------

os.environ["NCCL_SOCKET_IFNAME"] = "eth0"

model_config = ModelConfig(
    num_users=user_ct,
    num_items=product_ct,
    embedding_dim=128,
    hidden_dims=[128, 64],
    dropout_rate=0.2
)

data_config = DataConfig(
    train_path=config['output_dir_train'],
    eval_path=config['output_dir_validation'],
    batch_size=32,
    write_to_local=False,
    num_workers=4,
    shuffle=True
)

trainer_config = TrainerConfig(
    max_duration="1ep",
    eval_interval='1000ba',
    train_subset_num_batches=-1,
    eval_subset_num_batches=-1,
    save_folder='/local_disk0/composer-training/checkpoints',
    save_overwrite=True,
    device="gpu",
    learning_rate=0.001,
    weight_decay=1e-5
)


run_id = TorchDistributor(
    num_processes=4, local_mode=True, use_gpu=True).run(main_fn, model_config, data_config, trainer_config)

# COMMAND ----------

# MAGIC %md ### Run multi-node + multi-GPU

# COMMAND ----------

model_config = ModelConfig(
    num_users=user_ct,
    num_items=product_ct,
    embedding_dim=128,
    hidden_dims=[128],
    dropout_rate=0.2
)

data_config = DataConfig(
    train_path=config['output_dir_train'],
    eval_path=config['output_dir_validation'],
    batch_size=32,
    write_to_local=False,
    num_workers=4,
    shuffle=True
)

trainer_config = TrainerConfig(
    max_duration="1ep",
    eval_interval='100ba',
    train_subset_num_batches=1,
    eval_subset_num_batches=1,
    save_folder='/local_disk0/composer-training/checkpoints',
    save_overwrite=True,
    device="gpu",
    learning_rate=0.001,
    weight_decay=1e-5
)


run_id = TorchDistributor(
    num_processes=4, local_mode=False, use_gpu=True).run(main_fn, model_config, data_config, trainer_config)

# COMMAND ----------

# MAGIC %md ### Download checkpoints from MLflow and run inference on test set

# COMMAND ----------

def load_checkpoint_from_mlflow(run_id: str, model_class: torch.nn.Module, model_config: dict) -> Optional[torch.nn.Module]:
    """
    Download and load latest-rank0 checkpoint from MLflow artifacts.
    
    Args:
        run_id: MLflow run ID
        model_class: PyTorch model class to instantiate
        model_config: Configuration dictionary for model initialization
        
    Returns:
        Loaded PyTorch model or None if loading fails
    """
    try:
        # List artifacts in checkpoints directory
        artifacts = mlflow.artifacts.list_artifacts(
            run_id=run_id,
            artifact_path="checkpoints"
        )
        
        # Find latest-rank0 checkpoint
        checkpoint_files = [
            artifact.path 
            for artifact in artifacts 
            if 'latest-rank0' in artifact.path and artifact.path.endswith('.pt')
        ]
        
        if not checkpoint_files:
            print("No latest-rank0 checkpoint found")
            return None
            
        # Download checkpoint file
        checkpoint_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=checkpoint_files[0]
        )
        
        print(f"Downloaded checkpoint from: {checkpoint_files[0]}")
        
        # Initialize model
        model = model_class(config=model_config)
        
        # Load state dict from checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Handle Composer checkpoint format
        if 'state' in checkpoint and 'model' in checkpoint['state']:
            state_dict = checkpoint['state']['model']
        else:
            state_dict = checkpoint['model_state_dict']
            
        model.load_state_dict(state_dict)
        
        print("Successfully loaded model from checkpoint")
        return model
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None
      
model = load_checkpoint_from_mlflow(
    run_id=run_id,
    model_class=TwoTowerComposerModel,
    model_config=model_config
)

# Move model to GPU
model = model.to("cuda:0")
model.eval()

# Get batch and move to GPU
batch = get_dataloader_with_mosaic(config['output_dir_test'], batch_size=10, label="train")
next_batch = next(iter(batch))

# Move batch tensors to same device as model
device = next(model.parameters()).device
next_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in next_batch.items()}

# Run inference
with torch.no_grad():
    expected_result = next_batch["label"]
    actual_result = model(next_batch)
    actual_result = torch.sigmoid(actual_result)

print(f"Expected Result: {expected_result}; Actual Result: {actual_result.round()}")

# COMMAND ----------

# MAGIC %md ### Register model to MLflow so we can use in batch inference pipeline or serve using Model Serving endpoint

# COMMAND ----------

# DBTITLE 1,Build PyFunc
import pandas as pd
import numpy as np
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.models.signature import infer_signature

class TwoTowerWrapper(PythonModel):
    """
    MLflow PythonModel wrapper for TwoTower model that handles dictionary input and returns list outputs
    """
    def __init__(self, two_tower_model):
        self.two_tower_model = two_tower_model
        
    def predict(self, model_input: Dict[str, List]) -> List[float]:
        batch = {key: torch.tensor(value) for key, value in model_input.items()}
        with torch.no_grad():
            output = self.two_tower_model(batch).cpu()
        output = torch.sigmoid(output)
        return output.tolist()

# COMMAND ----------

# DBTITLE 1,Test PyFunc Wrapper
test_model = TwoTowerWrapper(model)

test_model.predict(next_batch)

# COMMAND ----------

sample_input = {
            "user_id": np.array([1, 2, 3]),
            "product_id": np.array([100, 200, 300])
        }

{"probability": test_model.predict(sample_input)}

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
            "user_id": np.array([1, 2, 3]),
            "product_id": np.array([100, 200, 300])
        }
        # Getting the model signature and logging the model
        pyfunc_two_tower_model = TwoTowerWrapper(model)
        current_output = {"prediction": pyfunc_two_tower_model.predict(sample_input)}
        signature = infer_signature(
          model_input=sample_input, 
          model_output=current_output)
            
        # Set MLflow registry to Unity Catalog
        mlflow.set_registry_uri("databricks-uc")
        
        # Start run in existing context
        with mlflow.start_run(run_id=run_id):
            # Log model as PyTorch flavor with signature
            model_info = mlflow.pyfunc.log_model(
              artifact_path="two_tower_pyfunc", 
              python_model=pyfunc_two_tower_model, 
              signature=signature, 
              registered_model_name=model_name
              )
            
            print(f"Model logged to: {model_info.model_uri}")
            
    except Exception as e:
        print(f"Error logging model: {str(e)}")
        raise

# Example usage:
model_name = "main.alex_m.two_tower_model_pyfunc"

# Log and register model
log_pyfunc_model_to_mlflow(
    model=model,
    run_id=run_id,
    model_name=model_name
)

# COMMAND ----------

def log_pytorch_model_to_mlflow(
    model: torch.nn.Module,
    run_id: str,
    model_name: str,
    input_example: Dict[str, torch.Tensor] = None
) -> None:
    """
    Log PyTorch model to MLflow and register in Unity Catalog with proper signature.
    
    Args:
        model: PyTorch model to log
        run_id: MLflow run ID to log model to
        model_name: Name to register model as in Unity Catalog (format: catalog.schema.model_name)
        input_example: Optional example input for model signature
    """
    try:
        # Create sample input and output for signature
        sample_input = {
            "user_id": np.array([1, 2, 3]),
            "product_id": np.array([100, 200, 300])
        }
        
        # Convert model to CPU and eval mode
        model = model.cpu().eval()
        
        # Create sample tensor inputs
        sample_users = torch.tensor(sample_input["user_id"])
        sample_products = torch.tensor(sample_input["product_id"])
        # sample_label = torch.tensor(sample_input['label'])
        
        # Get sample output
        with torch.no_grad():
            sample_output = model({"user_id": sample_users, "product_id": sample_products})
            
        # Convert to numpy for signature
        sample_output = {"labels": sample_output.numpy()}
        
        # Infer signature from numpy arrays
        signature = mlflow.models.infer_signature(
            model_input=sample_input,
            model_output=sample_output
        )
            
        # Set MLflow registry to Unity Catalog
        mlflow.set_registry_uri("databricks-uc")
        
        # Start run in existing context
        with mlflow.start_run(run_id=run_id):
            # Log model as PyTorch flavor with signature
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name
            )
            
            print(f"Model logged to: {model_info.model_uri}")
            
    except Exception as e:
        print(f"Error logging model: {str(e)}")
        raise

# Example usage:
model_name = "main.alex_m.two_tower_model"

# Move model to CPU and eval mode
model = model.cpu().eval()

# Log and register model
log_pytorch_model_to_mlflow(
    model=model,
    run_id=run_id,
    model_name=model_name
)
