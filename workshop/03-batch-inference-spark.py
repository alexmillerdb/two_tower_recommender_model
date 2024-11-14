# Databricks notebook source
# MAGIC %pip install -q --upgrade --no-deps --force-reinstall torch torchvision fbgemm-gpu torchrec --index-url https://download.pytorch.org/whl/cu118
# MAGIC %pip install -q torchmetrics==1.0.3 iopath pyre_extensions mosaicml-streaming==0.7.5
# MAGIC %pip install --upgrade mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

df = spark.table(f"{catalog}.{schema}.learning_from_sets_training_set")
display(df)

# COMMAND ----------

import mlflow
import torch
import numpy as np
from typing import Dict

# Load the model from the Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{schema}.learning_from_sets_two_tower"
model_version = 6

model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/{model_version}")

# COMMAND ----------

from pyspark.sql.functions import struct, col

predictions = df.withColumn('predictions', model(struct(*map(col, df.columns)))).toPandas()
display(predictions)

# COMMAND ----------

# predictions.write.mode("overwrite").saveAsTable("learning_from_sets_two_tower_predictions")
