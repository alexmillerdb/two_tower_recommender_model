# Databricks notebook source
# MAGIC %pip install -q mosaicml-streaming
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd
import random

# COMMAND ----------

# load prior order details to create training set
order_detail = spark.table("order_detail")
prior_order_detail = order_detail.filter(F.col('eval_set') == 'prior')

# create training set with positive labels
positive_samples = (
  prior_order_detail
  .select("user_id", "product_id").distinct()
  .withColumn("label", F.lit(1))
)

display(positive_samples)

# COMMAND ----------

# MAGIC %md ### Apply random negative sampling:
# MAGIC
# MAGIC The two-tower model aims to learn embeddings for users and items such that the dot product between user and item is high for relevant (positive interactions) and low for irrelevant (negative) interactions. This requires distinguishing between positive and negative samples during training. 
# MAGIC
# MAGIC Positive labels are derived from user interactions with items such as clicks, purchases, likes, or other engagement metrics. Negative labels can be generated using several methods but the most common is "Random Negative Sampling" which we will use in this accelerator. 
# MAGIC
# MAGIC We will use a "Pandas UDF" to create the negative samples since we are doubling the dataset size making this is an expensive operation.

# COMMAND ----------

# Create a list of all product_ids
all_product_ids = prior_order_detail.select("product_id").distinct().toPandas()['product_id'].to_list()

# Python function to dynamically return negative samples 
def generate_negatives(user_products):
    user, products = user_products['user_id'][0], list(user_products['products'][0])
    num_samples = len(products)
    negative_samples = random.sample(list(set(all_product_ids) - set(products)), num_samples)
    return pd.DataFrame([({"user_id": user, "product_id": product, "label": 0}) for product in negative_samples])

# Get the positive samples per user
user_products_list = positive_samples \
    .groupBy("user_id") \
    .agg(F.collect_set("product_id").alias("products"))

# Generate negative samples
negative_samples = user_products_list \
    .groupby("user_id") \
    .applyInPandas(generate_negatives, schema="user_id integer, product_id integer, label integer")

# display(negative_samples)

# Union negative samples with positive samples to create training set
training_set = positive_samples.union(negative_samples).cache()

print(f"Training set with positive and negative samples: {training_set.count()}")
display(training_set.groupby("label").count())

# COMMAND ----------

(training_set
 .write
 .format("delta")
 .mode("overwrite")
 .saveAsTable("training_set"))

# COMMAND ----------

# from pyspark.ml.feature import StringIndexer
# from pyspark.sql.types import LongType

# training_set = spark.table("training_set")
# string_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_index")
# indexed_df = string_indexer.fit(training_set).transform(training_set)
# indexed_df = indexed_df.withColumn("user_id_index", indexed_df["user_id_index"].cast(LongType()))
# # indexed_df = indexed_df.withColumn("user_id", indexed_df["user_id_index"]).drop("user_id_index")

# display(indexed_df)

# COMMAND ----------

# Split the dataframe into train, test, and validation sets
training_set = spark.table("training_set")
train_df, validation_df, test_df = training_set.randomSplit([0.7, 0.2, 0.1], seed=42)

# Show the count of each split to verify the distribution
print(f"Training Dataset Count: {train_df.count()}")
print(f"Validation Dataset Count: {validation_df.count()}")
print(f"Test Dataset Count: {test_df.count()}")

# COMMAND ----------

sample_train_df, sample_validation_df, sample_test_df = training_set.sample(fraction=0.05).randomSplit([0.7, 0.2, 0.1], seed=42)

# Show the count of each split to verify the distribution
print(f"Sample Training Dataset Count: {sample_train_df.count()}")
print(f"Sample Validation Dataset Count: {sample_validation_df.count()}")
print(f"Sample Test Dataset Count: {sample_test_df.count()}")

# COMMAND ----------

# import shutil

# # Specify the path of the folder to delete
# output_dir_train = config['output_dir_test']

# # Delete the folder and its contents
# shutil.rmtree(output_dir_train)

# COMMAND ----------

from streaming import StreamingDataset
from streaming.base.converters import dataframe_to_mds
from streaming.base import MDSWriter
from shutil import rmtree
import os
from tqdm import tqdm

# Parameters required for saving data in MDS format
cols = ["user_id", "product_id", "label"]
columns = { key: 'int32' for key in cols }
# label_dict = { 'label' : 'int' }
# columns = {**label_dict, **cat_dict}

compression = 'zstd:7'
hashes = ['sha1']
limit = 8192

# Specify where the data will be stored
output_dir_train = config['output_dir_train']
output_dir_validation = config['output_dir_validation']
output_dir_test = config['output_dir_test']
output_dir_train_sample = config['output_dir_train_sample']
output_dir_validation_sample = config['output_dir_validation_sample']
output_dir_test_sample = config['output_dir_test_sample']

# Save the training data using the `dataframe_to_mds` function, which divides the dataframe into `num_workers` parts and merges the `index.json` from each part into one in a parent directory.
def save_data(df, output_path, label, num_workers=4):
    print(f"Saving {label} data to: {output_path}")
    mds_kwargs = {'out': output_path, 'columns': columns, 'compression': compression, 'hashes': hashes, 'size_limit': limit}
    dataframe_to_mds(df.repartition(num_workers), merge_index=True, mds_kwargs=mds_kwargs)

# save full dataset
save_data(train_df, output_dir_train, 'train')
save_data(validation_df, output_dir_validation, 'validation')
save_data(test_df, output_dir_test, 'test')

# save sample dataset
save_data(sample_train_df, output_dir_train_sample, 'sample_train')
save_data(sample_validation_df, output_dir_validation_sample, 'sample_validation')
save_data(sample_test_df, output_dir_test_sample, 'sample_test')
