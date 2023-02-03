#!/usr/bin/env sh

# Set environment variable for the tracking URL where the Model Registry resides
export MLFLOW_TRACKING_URI=http://localhost:6007

# Serve the production model from the model registry
mlflow models serve -m "/home/airflow/dags/Clinical_data_pipeline/mlruns/1/81c6cb214a53453ca3cc181f5144ab64/artifacts/models"
