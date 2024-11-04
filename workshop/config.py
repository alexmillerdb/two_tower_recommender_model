# Databricks notebook source
catalog = "main"
schema = "alex_m"
volume = "learning_from_sets_data"

config = {}
config['catalog'] = catalog
config['schema'] = schema
config['volume'] = volume
config['volumes_path'] = f"/Volumes/{catalog}/{schema}/{volume}"
config['volumes_table_path'] = f"{catalog}.{schema}.{volume}"

config['output_dir_train'] = f"{config['volumes_path']}/mds_train"
config['output_dir_validation'] = f"{config['volumes_path']}/mds_validation"
config['output_dir_test'] = f"{config['volumes_path']}/mds_test"

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Volumes path: {config['volumes_path']}")

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")
