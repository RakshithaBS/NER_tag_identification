from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime,timedelta
import importlib.util

default_args ={'owner':'airflow',
              'start_date':datetime(2022,12,14),
              'retries':1,
              'retry_delay':timedelta(seconds=5)}

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = module_from_file("utils", "/opt/airflow/dags/scripts/utils.py")
constants = module_from_file("constants", "/opt/airflow/dags/scripts/constants.py")

ML_data_cleaning_dag = DAG("clinical_training_pipeline",default_args=default_args,
                          description="DAG to run model training pipeline for processed clinical data",
                          schedule_interval="@daily")


## task for reading file

read_train_sentence = PythonOperator(task_id='read_train_sentence',python_callable= utils.process_file,dag = ML_data_cleaning_dag,op_kwargs={'filename':constants.TRAIN_SENT_PATH,'dump_file_path':constants.TRAIN_SENT_PATH_PREPROCESSED,'config_file_path':constants.TRAINING_CONFIG_FILE_PATH})

read_train_labels = PythonOperator(task_id='read_train_labels',python_callable= utils.process_file,dag = ML_data_cleaning_dag,op_kwargs={'filename':constants.TRAIN_LABEL_PATH,'dump_file_path':constants.TRAIN_LABEL_PATH_PREPROCESSED,'config_file_path':constants.TRAINING_CONFIG_FILE_PATH})

preprocess_data = PythonOperator(task_id="pre_process_data",python_callable=utils.getPreProcessedOutput,dag=ML_data_cleaning_dag,op_kwargs=
                                 {'X_file_path':constants.TRAIN_SENT_PATH_PREPROCESSED,'Y_file_path':constants.TRAIN_LABEL_PATH_PREPROCESSED,'dump_file_path':constants.DATA_PATH,'training':True,'config_file_path':constants.TRAINING_CONFIG_FILE_PATH})

train_model = PythonOperator(task_id='train_model',python_callable=utils.trainModel,dag=ML_data_cleaning_dag,op_kwargs={'tracking_uri':constants.TRACKING_URI,'experiment_name':constants.EXPERIMENT_NAME,'training_file_path':constants.DATA_PATH,'config_file_path':constants.TRAINING_CONFIG_FILE_PATH})


preprocess_data.set_upstream([read_train_sentence,read_train_labels])
preprocess_data.set_downstream([train_model])






