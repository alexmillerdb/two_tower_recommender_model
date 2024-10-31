# Databricks notebook source
# %pip install -q mosaicml-streaming
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd
import random

# COMMAND ----------

products = spark.table("products")
display(products)

aisles = spark.table("aisles")
display(aisles)

departments = spark.table("departments")
display(departments)

# COMMAND ----------

# DBTITLE 1,Join product features together and concat text fields
combined_df = (products
               .join(aisles, on=['aisle_id'])
               .join(departments, on=['department_id'])
               .withColumn("product_desc", F.concat(
                 F.lit("Department: "), F.col("department"), 
                 F.lit("\nAisle: "), F.col("aisle"), 
                 F.lit("\nProduct: "), F.col("product_name"))
               )
               .drop("department_id", "aisle_id", "department", "aisle", "product_name")
)
display(combined_df)
combined_df.createOrReplaceTempView("combined_product_view")

# COMMAND ----------

# DBTITLE 1,Calculate Embeddings of the text fields using AI_QUERY
embed_df = spark.sql("""
SELECT *,
  ai_query(
    "databricks-gte-large-en",
    product_desc
  ) AS embedding
FROM combined_product_view
""").cache()

print(embed_df.count())
display(embed_df)

# COMMAND ----------

# DBTITLE 1,Analyze similar products based on embeddings
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
from scipy.spatial.distance import cosine
import pandas as pd

# Define the pandas UDF for cosine similarity
@pandas_udf(DoubleType())
def cosine_similarity(embeddings1: pd.Series, embeddings2: pd.Series) -> pd.Series:
    return pd.Series([
        1 - cosine(emb1, emb2) 
        for emb1, emb2 in zip(embeddings1, embeddings2)
    ])

# Create self cross join and calculate similarities
sample_df = embed_df.limit(100)
similar_products = (sample_df
    .alias('a')
    .crossJoin(sample_df.alias('b'))
    .filter('a.product_id != b.product_id')  # Avoid self-comparisons and duplicates
    .select(
        'a.product_id',
        'a.product_desc',
        'b.product_id',
        'b.product_desc',
        cosine_similarity('a.embedding', 'b.embedding').alias('similarity_score')
    )
    .orderBy('a.product_id', 'similarity_score', ascending=False)
)

# Show top similar pairs
display(similar_products)

# COMMAND ----------

# DBTITLE 1,Write results to Feature Store
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

def write_feature_table(name, df, primary_keys, description, timestamp_keys=None, mode="merge"):
  if not spark.catalog.tableExists(name):
    print(f"Feature table {name} does not exist. Creating feature table")
    fe.create_table(name=name,
                    primary_keys=primary_keys, 
                    timestamp_keys=timestamp_keys, 
                    df=df, 
                    description=description)
  else:
    print(f"Feature table {name} exists, writing updated results with mode {mode}")
    fe.write_table(
      df=df,
      name=name,
      mode=mode
    )

write_feature_table(
  name="product_embeddings",
  primary_keys=["product_id"],
  timestamp_keys=None,
  df=embed_df,
  description="Product text embeddings")
