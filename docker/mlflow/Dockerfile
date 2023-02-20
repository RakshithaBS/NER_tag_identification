FROM python:3.7-slim-buster

ENV MLFLOW_HOME /opt/mlflow 
ENV SERVER_PORT 5000
ENV SERVER_HOST 0.0.0.0
ENV BACKEND_STORE sqlite:///mlflow.db 
ENV ARTIFACT_STORE mlruns/ 
ENV MLFLOW_VERSION 0.7.0

# Set the user and group
ARG AIRFLOW_UID=50000
ARG AIRFLOW_GID=0
RUN useradd -u $AIRFLOW_UID -g $AIRFLOW_GID -s /bin/bash -m airflow

RUN  pip install mlflow && \
mkdir -p ${MLFLOW_HOME}/scripts && \
mkdir -p ${ARTIFACT_STORE}


COPY run.sh ${MLFLOW_HOME}/scripts/run.sh
RUN chmod +x ${MLFLOW_HOME}/scripts/run.sh
WORKDIR ${MLFLOW_HOME}

ENTRYPOINT ["./scripts/run.sh"]
