# Databricks notebook source
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

config['products_path'] = config['volumes_path'] + '/products/products.csv'
config['orders_path'] = config['volumes_path'] + '/orders/orders.csv'
config['order_products_path'] = config['volumes_path'] + '/order_products'
config['order_products_path_prior'] = config['volumes_path'] + '/order_products/order_products__prior.csv'
config['order_products_path_train'] = config['volumes_path'] + '/order_products/order_products__train.csv'
config['aisles_path'] = config['volumes_path'] + '/aisles/aisles.csv'
config['departments_path'] = config['volumes_path'] + '/departments/departments.csv'

# COMMAND ----------

spark.sql(f"USE CATALOG {config['catalog']}")
spark.sql(f"USE SCHEMA {config['schema']}")
