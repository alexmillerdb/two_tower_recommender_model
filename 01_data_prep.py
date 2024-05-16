# Databricks notebook source
# MAGIC %run ./config/notebook_config

# COMMAND ----------

# MAGIC %run ./config/data_extract

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import window as w
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType

# COMMAND ----------

# DBTITLE 1,Helper functions
def read_data(file_path, schema):
  df = (
    spark
      .read
      .csv(
        file_path,
        header=True,
        schema=schema
        )
    )
  return df
 
def write_data(df, table_name):
   _ = (
       df
        .write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema','true')
        .saveAsTable(table_name)
       )  

# COMMAND ----------

# orders data
# ---------------------------------------------------------
orders_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('user_id', IntegerType()),
  StructField('eval_set', StringType()),
  StructField('order_number', IntegerType()),
  StructField('order_dow', IntegerType()),
  StructField('order_hour_of_day', IntegerType()),
  StructField('days_since_prior_order', FloatType())
  ])
 
orders = read_data(config['orders_path'], orders_schema)
write_data(df=orders, table_name=f"{config['catalog']}.{config['schema']}.orders")
# ---------------------------------------------------------
 
# products
# ---------------------------------------------------------
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('aisle_id', IntegerType()),
  StructField('department_id', IntegerType())
  ])
 
products = read_data(config['products_path'], products_schema)
write_data(df=products, table_name=f"{config['catalog']}.{config['schema']}.products")
# ---------------------------------------------------------
 
# order products
# ---------------------------------------------------------
order_products_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('add_to_cart_order', IntegerType()),
  StructField('reordered', IntegerType())
  ])
 
order_products = read_data( config['order_products_path'], order_products_schema)
write_data( order_products, table_name=f"{config['catalog']}.{config['schema']}.order_products")
# write_data( order_products, table_name=f"{config['catalog']}.{config['schema']}.order_products_prior")
 
# order_products_train = read_data( config['order_products_path_train'], order_products_schema)
# write_data( order_products_train, table_name=f"{config['catalog']}.{config['schema']}.order_products_train")
# ---------------------------------------------------------
 
# departments
# ---------------------------------------------------------
departments_schema = StructType([
  StructField('department_id', IntegerType()),
  StructField('department', StringType())  
  ])
 
departments = read_data( config['departments_path'], departments_schema)
write_data(df=products, table_name=f"{config['catalog']}.{config['schema']}.departments")
# ---------------------------------------------------------
 
# aisles
# ---------------------------------------------------------
aisles_schema = StructType([
  StructField('aisle_id', IntegerType()),
  StructField('aisle', StringType())  
  ])
 
aisles = read_data( config['aisles_path'], aisles_schema)
write_data(df=products, table_name=f"{config['catalog']}.{config['schema']}.aisles")
# ---------------------------------------------------------

# COMMAND ----------

orders = spark.table('orders')
order_products = spark.table("order_products")
order_detail = (
  order_products
    .join(orders, on=['order_id'])
)
display(order_detail)

# COMMAND ----------

write_data(df=order_detail, table_name=f"{config['catalog']}.{config['schema']}.order_detail")
