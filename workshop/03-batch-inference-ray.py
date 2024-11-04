# Databricks notebook source
# MAGIC %pip install -q --upgrade --no-deps --force-reinstall torch torchvision fbgemm-gpu torchrec --index-url https://download.pytorch.org/whl/cu118
# MAGIC %pip install -q torchmetrics==1.0.3 iopath pyre_extensions mosaicml-streaming==0.7.5
# MAGIC %pip install --upgrade mlflow ray
# MAGIC dbutils.library.restartPython()

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

# Ray allows you to define custom cluster configurations using setup_ray_cluster function
# This allows you to allocate CPUs and GPUs on Ray cluster
cluster = setup_ray_cluster(
  min_worker_nodes=1,       # minimum number of worker nodes to start
  max_worker_nodes=1,       # maximum number of worker nodes to start
  num_gpus_worker_node=4,   # number of GPUs to allocate per worker node
  num_gpus_head_node=4,     # number of GPUs to allocate on head node (driver)
  num_cpus_worker_node=40,  # number of CPUs to allocate on worker nodes
  num_cpus_head_node=32     # number of CPUs to allocate on head node (driver)
)


# Pass any custom configuration to ray.init
context = ray.init(ignore_reinit_error=True)
print(ray.cluster_resources())

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import ray
import os

os.environ["DATABRICKS_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

ds = ray.data.read_parquet(f"{config['volumes_path']}/parquet")
ds.take(1)

# how to use with databricks tables
# ds = ray.data.read_databricks_tables(
#   warehouse_id="475b94ddc7cd5211",
#   table="learning_from_sets_dataset",
#   catalog="main",
#   schema="alex_m"
# )

# COMMAND ----------

import mlflow
import torch
import numpy as np
from typing import Dict
from mlflow.utils.databricks_utils import get_databricks_env_vars

# Load the model from the Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{schema}.learning_from_sets_two_tower"
model_version = 5

mlflow_db_creds = get_databricks_env_vars("databricks")

# Get the source run_id from the model registered in UC
client = mlflow.tracking.MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=model_version)
source_run_id = model_version_details.run_id
logged_model = f'runs:/{source_run_id}/two_tower_pyfunc'

class PyTorchPredictor:
    def __init__(self):

        # Set Databricks credentials
        os.environ.update(mlflow_db_creds)

        # Load model
        self.model = mlflow.pyfunc.load_model(logged_model)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Make predictions on batch with proper type conversion.
        
        Args:
            batch: Dictionary of numpy arrays
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Convert input types to match model signature
            input_batch = {
                'userId': batch['userId'].astype(np.int64),
                'movieId': batch['movieId'].astype(np.int64),
                'label': batch['label'].astype(np.int64)
            }
            
            # Get predictions
            predictions = self.model.predict(input_batch)
            
            # Convert predictions to probabilities and classes
            probs = np.array(predictions)
            predicted_classes = (probs >= 0.5).astype(np.int64)
            
            return {
                "userId": batch['userId'],
                "movieId": batch["movieId"],
                "label": batch["label"],
                "predicted_classes": predicted_classes,
                "prediction_prob": probs
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise

# COMMAND ----------

predictions = ds.map_batches(
  PyTorchPredictor,
  num_gpus=1,
  batch_size=500,
  concurrency=8
)

df = predictions.to_pandas()
display(df)

# COMMAND ----------

# MAGIC %md ## Write Ray Data to Spark:
# MAGIC
# MAGIC To write Ray data to Spark, you must write the dataset to a location that Spark can access.
# MAGIC
# MAGIC In Databricks Runtime ML below 15.0, you can write directly to an object store location using the Ray Parquet writer, ray_dataset.write_parquet() from the ray.data module. Spark can read this Parquet data with native readers.
# MAGIC
# MAGIC For Unity Catalog enabled workspaces, use the ray.data.Dataset.write_databricks_table function to write to a Unity Catalog table.
# MAGIC
# MAGIC This function temporarily stores the Ray dataset in Unity Catalog Volumes, reads from Unity Catalog volumes with Spark, and finally writes to a Unity Catalog table. Before calling ray.data.Dataset.write_databricks_table function, ensure that the environment variable "_RAY_UC_VOLUMES_FUSE_TEMP_DIR" is set to a valid and accessible Unity Catalog volume path, like "/Volumes/MyCatalog/MySchema/MyVolume/MyRayData".
# MAGIC
# MAGIC `ds = ray.data`
# MAGIC
# MAGIC `ds.write_databricks_table()`
# MAGIC
# MAGIC For workspaces that do not have Unity Catalog enabled, you can manually store a Ray Data dataset as a temporary file, such as a Parquet file in DBFS, and then read the data file with Spark.
# MAGIC
# MAGIC `ds.write_parquet(tmp_path)`
# MAGIC
# MAGIC `df = spark.read.parquet(tmp_path)`
# MAGIC
# MAGIC `df.write.format("delta").saveAsTable(table_name)`
