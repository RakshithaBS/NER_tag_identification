#!/bin/sh

mlflow server \
  --backend-store-uri $BACKEND_STORE \
  --default-artifact-root $ARTIFACT_STORE \
  --host $SERVER_HOST\
  --port $SERVER_PORT