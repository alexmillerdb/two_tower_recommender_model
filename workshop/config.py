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
