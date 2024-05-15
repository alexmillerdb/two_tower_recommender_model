# Databricks notebook source
# MAGIC %md Dataset: https://www.kaggle.com/c/instacart-market-basket-analysis

# COMMAND ----------

# MAGIC %md ### Move to utils folder

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

import os
# os.environ['kaggle_username'] = 'YOUR KAGGLE USERNAME HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_username'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_username_am")
os.environ['kaggle_key'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_key_am")

# COMMAND ----------

catalog = "main"
schema = "alex_m"
volume = "instacart_data"

config = {}
config['catalog'] = catalog
config['schema'] = schema
config['volume'] = volume
config['volumes_path'] = f"/Volumes/{catalog}/{schema}/{volume}"
config['volumes_table_path'] = f"{catalog}.{schema}.{volume}"

# COMMAND ----------

# MAGIC %md ### Download data from Kaggle and unzip csv files to Databricks driver

# COMMAND ----------

# MAGIC %sh -e
# MAGIC cd /databricks/driver
# MAGIC export KAGGLE_USERNAME=$kaggle_username
# MAGIC export KAGGLE_KEY=$kaggle_key
# MAGIC
# MAGIC # Download the dataset
# MAGIC kaggle competitions download -c instacart-market-basket-analysis
# MAGIC
# MAGIC # Unzip the downloaded files
# MAGIC unzip -o instacart-market-basket-analysis.zip
# MAGIC unzip -o aisles.csv.zip          
# MAGIC unzip -o departments.csv.zip     
# MAGIC unzip -o order_products__prior.csv.zip  
# MAGIC unzip -o order_products__train.csv.zip  
# MAGIC unzip -o orders.csv.zip          
# MAGIC unzip -o products.csv.zip        
# MAGIC unzip -o sample_submission.csv.zip
# MAGIC

# COMMAND ----------

# MAGIC %md ### Move the downloaded data to UC Volumes which will be used throughout the accelerator

# COMMAND ----------

import os

spark.sql(f"CREATE VOLUME IF NOT EXISTS {config['volumes_table_path']}")
dbutils.fs.mv("file:/databricks/driver/aisles.csv", 
              f"{config['volumes_path']}/aisles/aisles.csv")
print(f"Moving aisles.csv to {config['volumes_path']}/aisles/aisles.csv")

dbutils.fs.mv("file:/databricks/driver/departments.csv", 
              f"{config['volumes_path']}/departments/departments.csv")
print(f"Moving departments.csv to {config['volumes_path']}/departments/departments")

dbutils.fs.mv("file:/databricks/driver/order_products__prior.csv", 
              f"{config['volumes_path']}/order_products/order_products__prior.csv")
print(f"Moving order_products__prior.csv to {config['volumes_path']}/order_products/order_products__prior.csv")

dbutils.fs.mv("file:/databricks/driver/order_products__train.csv", 
              f"{config['volumes_path']}/order_products/order_products__train.csv")
print(f"Moving order_products__train.csv to {config['volumes_path']}/order_products/order_products__train.csv")

dbutils.fs.mv("file:/databricks/driver/orders.csv", 
              f"{config['volumes_path']}/orders/orders.csv")
print(f"Moving orders.csv to {config['volumes_path']}/orders/orders.csv")

dbutils.fs.mv("file:/databricks/driver/products.csv", 
              f"{config['volumes_path']}/products/products.csv")
print(f"Moving products.csv to {config['volumes_path']}/products/products.csv")
