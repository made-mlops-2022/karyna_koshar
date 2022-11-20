import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.email import send_email_smtp
from docker.types import Mount


def _failure_function(context):
    dag_run = context.get("dag_run")
    msg = "DAG run failed"
    subject = f"DAG {dag_run} has failed"
    send_email_smtp(to=default_args["email"], subject=subject, html_content=msg)


def _wait_for_file(file_name):
    return os.path.exists(file_name)


default_args = {
    "owner": "karyna_koshar",
    "email": ["k.kshr42@gmail.com"],
    "retries": 1,
    "on_failure_callback": _failure_function,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2022, 11, 15),
) as dag:
    wait_data = PythonSensor(
        task_id="wait_for_data",
        python_callable=_wait_for_file,
        op_args=["/opt/airflow/data/raw/{{ ds }}/data.csv"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    preprocess_data = DockerOperator(
        image="airflow-preprocess-data",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="airflow_preprocess_data",
        do_xcom_push=False,
        auto_remove=True,
        network_mode="bridge",
        mounts=[Mount(source=Variable.get("data_path"), target="/data", type="bind")],
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/predictions/{{ ds }}",
        task_id="airflow_predict",
        do_xcom_push=False,
        auto_remove=True,
        network_mode="host",
        mounts=[
            Mount(source=Variable.get("data_path"), target="/data", type="bind"),
            Mount(source=Variable.get("mlruns_path"), target="/mlruns", type="bind"),
        ],
    )

    wait_data >> preprocess_data >> predict
