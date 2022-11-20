from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.email import send_email_smtp
from docker.types import Mount


def _failure_function(context):
    dag_run = context.get("dag_run")
    msg = "DAG run failed"
    subject = f"DAG {dag_run} has failed"
    send_email_smtp(to=default_args["email"], subject=subject, html_content=msg)


default_args = {
    "owner": "karyna_koshar",
    "email": ["k.kshr42@gmail.com"],
    "retries": 1,
    "on_failure_callback": _failure_function,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="generate_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2022, 11, 15),
) as dag:
    get_data = DockerOperator(
        image="airflow-generate-data",
        command="--output-dir /data/raw/{{ ds }}",
        task_id="airflow_generate_data",
        do_xcom_push=False,
        auto_remove=True,
        network_mode="bridge",
        mounts=[Mount(source=Variable.get("data_path"), target="/data", type="bind")],
    )

    get_data
