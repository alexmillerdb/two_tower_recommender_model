# Two Tower Recommender Model Workshop

This workshop demonstrates how to build and train a Two Tower recommendation model using **TorchRec**, **Ray**, and **Mosaic ML's streaming capabilities** on **Databricks**. The model is designed for large-scale recommendation systems with efficient distributed training.

Additional documentation:
- https://docs.databricks.com/en/machine-learning/train-recommender-models.html
- https://www.databricks.com/blog/training-deep-recommender-systems-1

## Workshop Structure

The workshop consists of four main scripts, which should be executed in the following order:

### 1. Configuration (`config.py`)
- Configures the environment and dependencies needed for the entire workflow.
- Sets up paths, model parameters, and data configurations for use across the notebooks.

**File:** [config.py](https://github.com/alexmillerdb/two_tower_recommender_model/blob/main/workshop/config.py)

### 2. Data Preparation (`01-mosaic-streaming.py`)
- Downloads and processes the **Learning From Sets** dataset
- Converts data into Mosaic's streaming format
- Saves processed data to **Unity Catalog** volumes
- Implements efficient data loading using **Mosaic StreamingDataset**

**File:** [01-mosaic-streaming.py](https://github.com/alexmillerdb/two_tower_recommender_model/blob/main/workshop/01-mosaic-streaming.py)

**Key Features:**
- Efficient data preprocessing pipeline
- Data conversion to **MDS format**
- Train/validation/test split functionality
- Unity Catalog integration

### 3. Model Training (`02-mosaic-model-training.py`)
- Implements **Two Tower architecture** using TorchRec
- Supports multiple training configurations:
  - Single Node Single GPU (**SNSG**)
  - Single Node Multi GPU (**SNMG**)
  - Multi Node Multi GPU (**MNMG**)
- **MLflow** integration for experiment tracking
- Model inference capabilities

**File:** [02-mosaic-model-training.py](https://github.com/alexmillerdb/two_tower_recommender_model/blob/main/workshop/02-mosaic-model-training.py)

**Key Features:**
- Distributed training support
- GPU memory optimization
- Embedding table sharding
- MLflow experiment tracking
- Inference pipeline

### 4. Batch Inference (`03-batch-inference.py`)
- Performs batch inference on the trained model
- Uses **Ray** for distributed inference, optimized for large-scale data
- Capable of handling batch predictions with optimized latency

**File:** [03-batch-inference.py](https://github.com/alexmillerdb/two_tower_recommender_model/blob/main/workshop/03-batch-inference-ray.py)

**Key Features:**
- Batch inference for high throughput
- Distributed inference with Ray
- Suitable for real-time recommendation needs

## Prerequisites

- **Databricks Runtime ML 14.3 LTS** or later
- GPU-enabled cluster (tested on `g4dn.12xlarge` instances)
- **Unity Catalog** enabled workspace
- Python packages:
  ```text
  mosaicml-streaming==0.7.5
  torch 
  torchvision
  fbgemm-gpu
  torchrec
  torchmetrics==1.0.3
  iopath
  pyre_extensions

## Setup Instructions

### Cluster Configuration
```python
# Recommended cluster config
- Node Type: g5.12xlarge
- Workers: 1-2 nodes
- GPUs: 4 A10 GPUs per node
- Runtime: Databricks Runtime ML 14.3 LTS
