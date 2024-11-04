# Databricks notebook source
# MAGIC %md ### Cluster configuration:
# MAGIC - Important to set Spark environment variables when using AWS workspace
# MAGIC
# MAGIC `{
# MAGIC     "cluster_name": "alex_m_multinode_gpus",
# MAGIC     "spark_version": "15.0.x-gpu-ml-scala2.12",
# MAGIC     "spark_conf": {
# MAGIC         "spark.databricks.pyspark.dataFrameChunk.enabled:": "true"
# MAGIC     },
# MAGIC     "aws_attributes": {
# MAGIC         "first_on_demand": 2,
# MAGIC         "availability": "SPOT",
# MAGIC         "zone_id": "auto",
# MAGIC         "spot_bid_price_percent": 100,
# MAGIC         "ebs_volume_count": 0
# MAGIC     },
# MAGIC     "node_type_id": "g5.24xlarge",
# MAGIC     "driver_node_type_id": "g5.24xlarge",
# MAGIC     "spark_env_vars": {
# MAGIC         "NCCL_P2P_DISABLE": "1",
# MAGIC         "NCCL_IGNORE_DISABLED_P2P": "1"
# MAGIC     },
# MAGIC     "autotermination_minutes": 120,
# MAGIC     "enable_elastic_disk": false,
# MAGIC     "single_user_name": "alex.miller@databricks.com",
# MAGIC     "enable_local_disk_encryption": false,
# MAGIC     "data_security_mode": "SINGLE_USER",
# MAGIC     "runtime_engine": "STANDARD",
# MAGIC     "num_workers": 1
# MAGIC }`

# COMMAND ----------

# MAGIC %pip install xgboost_ray ray[default]==2.10
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("experiment_name", "spark_vs_ray_xgboost", "experiment_name")
experiment_name = dbutils.widgets.get("experiment_name")
print(f"Logging to MLflow Experiment {experiment_name}")

dbutils.widgets.text("num_training_rows", "100", "num_training_rows")
num_training_rows = int(dbutils.widgets.get("num_training_rows"))
print(f"Generating {num_training_rows} synthetic rows")

dbutils.widgets.text("num_training_columns", "1000", "num_training_columns")
num_training_columns = int(dbutils.widgets.get("num_training_columns"))
print(f"Generating {num_training_columns} synthetic columns")

dbutils.widgets.text("num_labels", "2", "num_labels")
num_labels = int(dbutils.widgets.get("num_labels"))
print(f"Generating {num_labels} synthetic labels")

dbutils.widgets.text("max_depth", "5", "max_depth")
max_depth = int(dbutils.widgets.get("max_depth"))
print(f"XGBoost max_depth: {max_depth}")

dbutils.widgets.text("n_estimators", "100", "n_estimators")
n_estimators = int(dbutils.widgets.get("n_estimators"))
print(f"XGBoost n_estimators: {n_estimators}")

concurrency = sc.defaultParallelism
print(f"Setting Spark.XGBoost num_workers to {concurrency} = num cores on workers in cluster")

# COMMAND ----------

# MAGIC %md ### Setup Ray Cluster:

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
setup_ray_cluster(
  min_worker_nodes=1,       # minimum number of worker nodes to start
  max_worker_nodes=1,       # maximum number of worker nodes to start
  num_gpus_worker_node=4,   # number of GPUs to allocate per worker node
  num_gpus_head_node=4,     # number of GPUs to allocate on head node (driver)
  num_cpus_worker_node=80,  # number of CPUs to allocate on worker nodes
  num_cpus_head_node=80     # number of CPUs to allocate on head node (driver)
)


# Pass any custom configuration to ray.init
ray.init(ignore_reinit_error=True)
print(ray.cluster_resources())

# COMMAND ----------

from typing import Tuple

import ray
from ray.data import Dataset, Preprocessor
from ray.data.preprocessors import StandardScaler
from ray.train.xgboost import XGBoostTrainer
from ray.train import Result, ScalingConfig, RunConfig, Checkpoint
import xgboost

# COMMAND ----------

# MAGIC %md ### Setup catalog, schema, and table paths

# COMMAND ----------

catalog = "main"
schema = "alex_m"
if num_labels > 2:
  table_path = f"synthetic_data_{num_training_rows}_rows_{num_training_columns}_columns_{num_labels}_labels"
else:
  table_path = f"synthetic_data_{num_training_rows}_rows_{num_training_columns}_columns"
  
parquet_path = f"/Volumes/{catalog}/{schema}/my_volume/{table_path}"
print(f"Parquet path: {parquet_path}")

# COMMAND ----------

# MAGIC %md ### Prepare data function using Ray Data:
# MAGIC - Ray data is reading parquet from UC Volumes

# COMMAND ----------

def prepare_data() -> Tuple[Dataset, Dataset, Dataset]:
    
    dataset = ray.data.read_parquet(parquet_path)
    train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)
    test_dataset = valid_dataset.drop_columns(["target"])
    return train_dataset, valid_dataset, test_dataset
  
train_dataset, valid_dataset, _ = prepare_data()

# COMMAND ----------

# MAGIC %md ### For multi-node clusters you need to pass dbfs or s3 path to log XGBoost results

# COMMAND ----------

import os

# for multi-node you need to pass shared storage path (dbfs or s3)
dbfs_path = '/dbfs/tmp/ray_xgboost_trainer/'  # specify the DBFS path here

# check if the folder already exists
if not os.path.exists(dbfs_path):
    # create the folder if it doesn't exist
    os.makedirs(dbfs_path)

print(os.path.exists(dbfs_path))

# COMMAND ----------

# MAGIC %md ### XGBoost train function using XGBoostTrainer

# COMMAND ----------

def train_xgboost(train_dataset, valid_dataset, num_workers: int, use_gpu: bool = False) -> Result:

    # XGBoost specific params
    params = {
        "tree_method": "gpu_hist",
        "max_depth": max_depth
    }

    # dynamically update params based on the number of labels/classes
    if num_labels > 2:
        # using 'one_output_per_tree' but you can also use 'multi_output_tree' too
        params['multi_strategy'] = 'one_output_per_tree'
        params["objective"] = "multi:softprob"
        params["num_class"] = num_labels
        params["eval_metric"] = ["mlogloss", "merror"]
    else:
        params['objective'] = "binary:logistic"
        params["eval_metric"] = ["logloss", "error"]

    trainer = XGBoostTrainer(
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config=RunConfig(storage_path=dbfs_path),
        label_column="target",
        params=params,
        datasets={"train": train_dataset, "valid": valid_dataset},
        num_boost_round=n_estimators
    )
            # metadata = {"preprocessor_pkl": preprocessor.serialize()}
    result = trainer.fit()
    print(result.metrics)

    return result

# COMMAND ----------

# MAGIC %md ### Create cluster and ray params to log into MLflow:
# MAGIC - Ray/XGBoost documentation: https://docs.ray.io/en/latest/train/distributed-xgboost-lightgbm.html

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import os

databricks_host = os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
databricks_token = os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

w = WorkspaceClient()
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

ray_cluster_params = ray.cluster_resources()

# COMMAND ----------

# MAGIC %md ### Run training function and log to MLflow

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from time import time
import numpy as np

mlflow.set_registry_uri("databricks-uc")
experiment_path = f"/Users/alex.miller@databricks.com/deep-learning/shutterfly/xgboost_scaling/{experiment_name}"
mlflow.set_experiment(experiment_name=experiment_path)

with mlflow.start_run() as run:

  params = {
    "num_training_rows": num_training_rows,
    "num_training_columns": num_training_columns,
    "framework": "ray-xgboost",
    "max_depth": max_depth,
    "n_estimators": n_estimators
  }
  # log parameters
  mlflow.log_params(params)
  mlflow.log_params(cluster_params)
  mlflow.log_params(ray_cluster_params)

  # Start the timer
  start_time = time()

  # run the Ray XGBoost Trainer
  result = train_xgboost(train_dataset, valid_dataset, num_workers=8, use_gpu=True)

  # Calculate and log the elapsed time
  elapsed_time = time() - start_time
  mlflow.log_metric("elapsed_time", elapsed_time)

  # log the pandas dataset that contains training metrics
  mlflow_dataset = mlflow.data.from_pandas(result.metrics_dataframe)
  mlflow.log_input(mlflow_dataset, context="training_metrics")

  results_metrics = result.metrics_dataframe.iloc[-1].to_dict()
  mlflow.log_params(results_metrics)

  # log the model using the best checkpoint
  checkpoint = Checkpoint.from_directory(result.checkpoint.path)
  model = XGBoostTrainer.get_model(checkpoint)

  # infer signature, using random array since data is randomly generated
  data = np.random.rand(1, num_training_columns + 1)
  dtrain = xgboost.DMatrix(data, feature_names=model.feature_names)
  output = model.predict(dtrain)
  signature = infer_signature(model_input=data, model_output=output)

  # log model to mlflow as xgboost model
  model_info = mlflow.xgboost.log_model(
    model, 
    artifact_path="model",
    signature=signature,
    input_example=data,
    registered_model_name=f"{catalog}.{schema}.ray_xgboost_scaling"
    )

mlflow.end_run()

# COMMAND ----------

# load in mlflow registered model and predict mock batch
mlflow_model = mlflow.xgboost.load_model(model_info.model_uri)

mlflow_model.predict(dtrain)
