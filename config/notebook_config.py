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
spark.sql(f'CREATE SCHEMA IF NOT EXISTS {schema}')
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f'USE SCHEMA {schema}')
