# Databricks notebook source
# MAGIC %md # Two Tower Data Curation with Mosaic Streaming
# MAGIC
# MAGIC This notebook illustrates how to preprocess and persist the data required for a Two Tower recommendation model using Mosaic's `streaming` library. This notebook was tested on `g4dn.12xlarge` instances (one instance as the driver, one instance as the worker) using the Databricks Runtime for ML 14.3 LTS.
# MAGIC
# MAGIC This notebook uses the small sample of 100k ratings from the [Learning From Sets of Items](https://grouplens.org/datasets/learning-from-sets-of-items-2019/) dataset. In this section you preprocess it and save it to a Volume in Unity Catalog.

# COMMAND ----------

# MAGIC %pip install -q mosaicml-streaming==0.7.5
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Step 0. Defining Variables
# MAGIC
# MAGIC The following cell contains variables that are relevant to this notebook. If this notebook was imported from the marketplace, make sure to update the variables and paths as needed to point to the correct catalogs, volumes, etc.

# COMMAND ----------

# MAGIC %sh wget https://files.grouplens.org/datasets/learning-from-sets-2019/learning-from-sets-2019.zip -O /databricks/driver/learning-from-sets-2019.zip && unzip /databricks/driver/learning-from-sets-2019.zip -d /databricks/driver/

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import pandas as pd

# Load the CSV file into a pandas DataFrame (since the data is stored on local machine)
df = pd.read_csv("/databricks/driver/learning-from-sets-2019/item_ratings.csv")

# Create a Spark DataFrame from the pandas DataFrame and save it to UC
spark_df = spark.createDataFrame(df)

# TODO: Update this with a path in UC for where this data should be saved
spark_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.learning_from_sets_dataset")
spark_df.write.format("parquet").mode("overwrite").save(f"{config['volumes_path']}/parquet")

# COMMAND ----------

# TODO: This path may need to be updated to point to the location where the Criteo dataset is located.
learning_from_sets_table = f"{catalog}.{schema}.learning_from_sets_dataset"

# TODO: The marketplace example provides you with the MDS-formatted data, but if the data needs to be saved to an alternate UC Volumes path, set this variable to `True` and update the following paths.
save_data_to_uc_volumes = False

# TODO: These paths should be updated if the MDS-formatted data need to be stored at a different UC Volumes location.
output_dir_train = config['output_dir_train']
output_dir_validation = config['output_dir_validation']
output_dir_test = config['output_dir_test']

# COMMAND ----------

# MAGIC %md ## Step 1. Reading the Dataset from UC
# MAGIC
# MAGIC The dataset's description, taken from their [`README.md`](https://files.grouplens.org/datasets/learning-from-sets-2019/README):
# MAGIC
# MAGIC *This dataset contains results from a survey about the users' ratings on sets of movies...*
# MAGIC
# MAGIC *General notes:*
# MAGIC
# MAGIC - *Movie identifiers are consistent with those used in the MovieLens datasets: <https://grouplens.org/datasets/movielens/>*
# MAGIC - *User identifiers have been obfuscated to protect users' privacy.*
# MAGIC - *Both movie and user identifiers are consistent across this dataset.*
# MAGIC
# MAGIC The original dataset contains roughly 500k data points. This example notebook uses a sample of 100k data points from the dataset.

# COMMAND ----------

spark_df = spark.table(learning_from_sets_table)
print(f"Dataset size: {spark_df.count()}")
display(spark_df)

# COMMAND ----------

# Order by userId and movieId (this allows you to get a better representation of movieIds and userIds for the dataset)
ordered_df = spark_df.orderBy("userId", "movieId").limit(100_000)

print(f"Updated Dataset Size: {ordered_df.count()}")
# Show the result
display(ordered_df)

# COMMAND ----------

from pyspark.sql.functions import countDistinct

# Get the total number of data points
print("Total # of data points:", ordered_df.count())

# Get the total number of users
total_users = ordered_df.select(countDistinct("userId")).collect()[0][0]
print(f"Total # of users: {total_users}")

# Get the total number of movies
total_movies = ordered_df.select(countDistinct("movieId")).collect()[0][0]
print(f"Total # of movies: {total_movies}")

# COMMAND ----------

# MAGIC %md ## Step 2. Preprocessing and Cleaning the Data
# MAGIC
# MAGIC The first step is to convert the hashes (in string format) of each user to an integer using the StringIndexer.
# MAGIC
# MAGIC The Two Tower Model provided by TorchRec [here](https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L4) requires a binary label. The code in this section converts all ratings less than the mean to `0` and all values greater than the mean to `1`. For your own use case, you can modify the training task described [here](https://github.com/pytorch/torchrec/blob/main/examples/retrieval/modules/two_tower.py#L117) to use MSELoss instead (which can scale to ratings from 0 -> 5).

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import LongType

string_indexer = StringIndexer(inputCol="userId", outputCol="userId_index")
indexed_df = string_indexer.fit(ordered_df).transform(ordered_df)
indexed_df = indexed_df.withColumn("userId_index", indexed_df["userId_index"].cast(LongType()))
indexed_df = indexed_df.withColumn("userId", indexed_df["userId_index"]).drop("userId_index")

display(indexed_df)

# COMMAND ----------

from pyspark.sql import functions as F

# Select only the userId, movieId, and ratings columns
relevant_df = indexed_df.select('userId', 'movieId', 'rating')

# Calculate the mean of the ratings column
ratings_mean = relevant_df.groupBy().avg('rating').collect()[0][0]
print(f"Mean rating: {ratings_mean}")

# Modify all ratings less than the mean to 0 and greater than the mean to 1 and using a UDF to apply the transformation
modify_rating_udf = F.udf(lambda x: 0 if x < ratings_mean else 1, 'int')
relevant_df = relevant_df.withColumn('rating', modify_rating_udf('rating'))

# Rename rating to label
relevant_df = relevant_df.withColumnRenamed('rating', 'label')

# Displaying the dataframe
display(relevant_df)

# COMMAND ----------

# MAGIC %md ## Step 3. Saving to MDS Format within UC Volumes
# MAGIC
# MAGIC In this step, you convert the data to MDS to allow for efficient train/validation/test splitting and then save it to a UC Volume.
# MAGIC
# MAGIC View the Mosaic Streaming guide here for more details:
# MAGIC 1. General details: https://docs.mosaicml.com/projects/streaming/en/stable/
# MAGIC 2. Main concepts: https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/main_concepts.html#dataset-conversion
# MAGIC 2. `dataframeToMDS` details: https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/spark_dataframe_to_mds.html

# COMMAND ----------

# Split the dataframe into train, test, and validation sets
train_df, validation_df, test_df = relevant_df.randomSplit([0.7, 0.2, 0.1], seed=42)

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
cols = ["userId", "movieId"]
cat_dict = { key: 'int64' for key in cols }
label_dict = { 'label' : 'int' }
columns = {**label_dict, **cat_dict}

compression = 'zstd:7'

# Save the train/validation/test data using the `dataframe_to_mds` function, which divides the dataframe into `num_workers` parts and merges the `index.json` from each part into one in a parent directory.
def save_data(df, output_path, label, num_workers=40):
    print(f"Saving {label} data to: {output_path}")
    mds_kwargs = {'out': output_path, 'columns': columns, 'compression': compression}
    dataframe_to_mds(df.repartition(num_workers), merge_index=True, mds_kwargs=mds_kwargs)

if save_data_to_uc_volumes:
    save_data(train_df, output_dir_train, 'train')
    save_data(validation_df, output_dir_validation, 'validation')
    save_data(test_df, output_dir_test, 'test')
