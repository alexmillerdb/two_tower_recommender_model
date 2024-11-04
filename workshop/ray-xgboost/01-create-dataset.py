# Databricks notebook source
# MAGIC %pip install dbldatagen
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

concurrency = sc.defaultParallelism
print(f"Setting Spark.XGBoost num_workers to {concurrency} = num cores on workers in cluster")

# COMMAND ----------

# # Print out cluster configurations in Python
# configs = spark.sparkContext.getConf().getAll()
# for key, value in configs:
#     print(f"{key} = {value}")

# COMMAND ----------

import dbldatagen as dg
from pyspark.sql.types import FloatType, IntegerType, StringType

# partitions_requested = concurrency

testDataSpec = (
    dg.DataGenerator(spark, name="synthetic_data", rows=num_training_rows, partitions=concurrency)
    .withIdOutput()
    .withColumn(
        "r",
        FloatType(),
        expr="rand()",
        numColumns=num_training_columns,
    )
    .withColumn(
      "target",
      IntegerType(),
      expr=f"floor(rand()*{num_labels})",
      numColumns=1
      )
)

df = testDataSpec.build()
df = df.repartition(50)
df.write.format("delta").mode("overwrite").saveAsTable(f"main.alex_m.synthetic_data_{num_training_rows}_rows_{num_training_columns}_columns_{num_labels}_labels")
df.write.mode("overwrite").format("parquet").save(f"/Volumes/main/alex_m/my_volume/synthetic_data_{num_training_rows}_rows_{num_training_columns}_columns_{num_labels}_labels")
