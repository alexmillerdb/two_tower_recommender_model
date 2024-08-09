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

display(positive_samples.groupby("user_id").count().orderBy(F.col("count").desc()))

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
    # num_samples = 50 if len(products) > 50 else (5 if len(products) < 2 else len(products))
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

# MAGIC %md ### Split the data into train, validation, and test sets
# MAGIC
# MAGIC In order to create embeddings for all products and users in the training set during the model training process, we need to ensure that every user is included in each split (train/val/test). The reason for this is to ensure every customer has an embedding created during inference.
# MAGIC
# MAGIC We will again use a Pandas UDF to efficiently split each 'user_id' into train/val/test for the reasons mentioned above. This is a simple approach that randomly assigns the splits but other methods could include a time dimension to split the data based on order dates.

# COMMAND ----------

# training_set = positive_samples

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Add a random number column
df_with_random = training_set.withColumn("random", F.rand())

# Create a window partitioned by user_id and ordered by the random number
window = Window.partitionBy("user_id").orderBy("random")

# Add a row number within each user's partition
df_with_rownum = df_with_random.withColumn("row_num", F.row_number().over(window))

# Calculate the total number of rows for each user
df_with_total = df_with_rownum.withColumn(
    "total_rows", 
    F.count("*").over(Window.partitionBy("user_id"))
) \
  .withColumn("row_percent", F.col("row_num") / F.col("total_rows"))

# Assign groups based on the row number and total rows
df_with_groups = df_with_total.withColumn(
    "group",
    F.when(F.col("row_num") == 1, "train")
     .when(F.col("row_percent") < 0.8, "train")
     .when(F.col("row_percent") < 0.9, "val").otherwise("test")
     )

# # Remove the temporary columns
final_df = df_with_groups.drop("random", "row_num", "total_rows", "row_percent")
display(final_df)

# COMMAND ----------

display(final_df.groupby("group").count().orderBy("count"))

# COMMAND ----------

display(final_df.select("user_id", "group").distinct().groupBy("user_id").count().filter(F.col("count") < 3))

# COMMAND ----------

# Assuming final_df is the DataFrame with user_id and group columns

# Aggregate to check if each user_id has at least one 'train' group
user_group_check = final_df.groupBy("user_id").agg(
    F.sum(F.when(F.col("group") == "train", 1).otherwise(0)).alias("train_count")
)

# Filter out users who do not have any 'train' group
users_without_train = user_group_check.filter(F.col("train_count") == 0).select("user_id")

# Show the users who do not have the 'train' group
display(users_without_train)

# COMMAND ----------

# spark.sql("drop table if exists training_set")

# COMMAND ----------

(final_df
 .write
 .format("delta")
 .mode("overwrite")
 .saveAsTable("training_set"))

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import LongType

training_set = spark.table("training_set")
string_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_index")
indexed_df = string_indexer.fit(training_set).transform(training_set)
indexed_df = indexed_df.withColumn("user_id_index", indexed_df["user_id_index"].cast(LongType()))
# indexed_df = indexed_df.withColumn("user_id", indexed_df["user_id_index"]).drop("user_id_index")

display(indexed_df)

# COMMAND ----------

# Split the dataframe into train, test, and validation sets
training_set = indexed_df

# train_df, validation_df, test_df = training_set.randomSplit([0.7, 0.2, 0.1], seed=42)
train_df = training_set.filter(F.col("group") == "train").drop("group")
validation_df = training_set.filter(F.col("group") == "val").drop("group")
test_df = training_set.filter(F.col("group") == "test").drop("group")

# Show the count of each split to verify the distribution
print(f"Training Dataset Count: {train_df.count()}")
print(f"Validation Dataset Count: {validation_df.count()}")
print(f"Test Dataset Count: {test_df.count()}")

# COMMAND ----------

from streaming import StreamingDataset
from streaming.base.converters import dataframe_to_mds
from streaming.base import MDSWriter
from shutil import rmtree
import os
from tqdm import tqdm

# Parameters required for saving data in MDS format
cols = ["user_id", "product_id", "label", "user_id_index"]
cols = ["user_id", "product_id", "user_id_index"]
cat_dict = {key: ('int32' if key != "user_id_index" else 'int64') for key in cols}
label_dict = { 'label' : 'int32' }
columns = {**label_dict, **cat_dict}

compression = 'zstd:7'
hashes = ['sha1']
limit = 8192

# Specify where the data will be stored
output_dir_train = config['output_dir_train']
output_dir_validation = config['output_dir_validation']
output_dir_test = config['output_dir_test']

# Save the training data using the `dataframe_to_mds` function, which divides the dataframe into `num_workers` parts and merges the `index.json` from each part into one in a parent directory.
def save_data(df, output_path, label, num_workers=4):
    if os.path.exists(output_path):
        print(f"Deleting {label} data: {output_path}")
        rmtree(output_path)
    print(f"Saving {label} data to: {output_path}")
    mds_kwargs = {'out': output_path, 'columns': columns, 'compression': compression, 'hashes': hashes, 'size_limit': limit}
    dataframe_to_mds(df.repartition(num_workers), merge_index=True, mds_kwargs=mds_kwargs)

# save full dataset
save_data(train_df, output_dir_train, 'train', 10)
save_data(validation_df, output_dir_validation, 'validation', 10)
save_data(test_df, output_dir_test, 'test', 10)
