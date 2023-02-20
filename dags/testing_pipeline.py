from airflow import DAG
from airflow.operators.python import PythonOperator,BranchPythonOperator
from airflow.operators.bash import BashOperator
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

ML_inference_cleaning_dag = DAG("clinical_testing_pipeline",default_args=default_args,
                          description="DAG to run model training pipeline for processed clinical data",
                          schedule_interval="@daily")


## task for reading file

read_test_sentence = PythonOperator(task_id='read_test_sentence',python_callable= utils.process_file,dag = ML_inference_cleaning_dag,op_kwargs={'filename':constants.TEST_SENT_PATH,'dump_file_path':constants.TEST_SENT_PATH_PREPROCESSED,'config_file_path':constants.INFERENCE_CONFIG_FILE_PATH})

read_test_labels = PythonOperator(task_id='read_test_labels',python_callable= utils.process_file,dag = ML_inference_cleaning_dag,op_kwargs={'filename':constants.TEST_LABEL_PATH,'dump_file_path':constants.TEST_LABEL_PATH_PREPROCESSED,'config_file_path':constants.INFERENCE_CONFIG_FILE_PATH})

preprocess_data = PythonOperator(task_id="pre_process_data",python_callable=utils.getPreProcessedOutput,dag=ML_inference_cleaning_dag,op_kwargs=
                                 {'X_file_path':constants.TEST_SENT_PATH_PREPROCESSED,'Y_file_path':constants.TEST_LABEL_PATH_PREPROCESSED,'dump_file_path':constants.DATA_PATH,'training':False,'config_file_path':constants.INFERENCE_CONFIG_FILE_PATH})

evaluate_model = PythonOperator(task_id='evaluate_model',python_callable=utils.evaluateModel,dag=ML_inference_cleaning_dag,op_kwargs={'model_path': '{{ dag_run.conf["model_path"] }}' ,'test_file_path':constants.DATA_PATH,'config_file_path':constants.INFERENCE_CONFIG_FILE_PATH})

check_model=BranchPythonOperator(task_id='check_model_performance',python_callable=utils.checkIfDeployModel,dag=ML_inference_cleaning_dag,op_kwargs={'file_path':constants.DATA_PATH})

# serve model using airflow pipeline. Just for experimentation. 
deploy_model = BashOperator(
    task_id='deploy_model',
    bash_command='mlflow serve --model-uri {{dag_run.conf["model_path"]}} -p 9080',
    depends_on_past=True,
    dag=ML_inference_cleaning_dag
)

failure_model = BashOperator(
    task_id='task_failure',
    bash_command ='echo task_failure',
    dag=ML_inference_cleaning_dag
)
preprocess_data.set_upstream([read_test_sentence,read_test_labels])
preprocess_data.set_downstream(evaluate_model)
evaluate_model.set_downstream(check_model)
check_model.set_downstream([deploy_model,failure_model])


