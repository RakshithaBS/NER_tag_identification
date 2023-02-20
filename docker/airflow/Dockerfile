FROM apache/airflow:slim-latest
ENV AIRFLOW_HOME /opt/airflow
ENV AIRFLOW_USERNAME rakshitha
ENV AIRFLOW_FIRSTNAME rakshitha \
ENV AIRFLOW_LASTNAME bs
ENV AIRFLOW_EMAIL abc@gmail.com
ENV AIRFLOW_PASSWORD airflow
ENV MLFLOW_TRACKING_URI http://mlflow:5000
COPY requirements.txt /

RUN pip install --no-cache-dir -r /requirements.txt
RUN python -m spacy download en_core_web_sm

COPY run.sh /

WORKDIR ${AIRFLOW_HOME}
ENTRYPOINT ["/run.sh"]