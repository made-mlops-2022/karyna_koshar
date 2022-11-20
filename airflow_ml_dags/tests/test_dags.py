import pytest
from airflow.models import DagBag


@pytest.fixture()
def dagbag():
    return DagBag()


def assert_dag_dict_equal(source, dag):
    assert dag.task_dict.keys() == source.keys()
    for task_id, downstream_list in source.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)


def test_generate_data_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="generate_data")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 1


def test_train_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="train")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 6


def test_predict_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="predict")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 3


def test_generate_data_structure(dagbag):
    dag = dagbag.get_dag(dag_id="generate_data")
    assert_dag_dict_equal(
        {"airflow_generate_data": []},
        dag,
    )


def test_train_structure(dagbag):
    dag = dagbag.get_dag(dag_id="train")
    assert_dag_dict_equal(
        {
            "wait_for_data": ["airflow_preprocess_data"],
            "wait_for_target": ["airflow_preprocess_data"],
            "airflow_preprocess_data": ["airflow_split_data"],
            "airflow_split_data": ["airflow_train"],
            "airflow_train": ["airflow_validate"],
            "airflow_validate": [],
        },
        dag,
    )


def test_predict_structure(dagbag):
    dag = dagbag.get_dag(dag_id="predict")
    assert_dag_dict_equal(
        {
            "wait_for_data": ["airflow_preprocess_data"],
            "airflow_preprocess_data": ["airflow_predict"],
            "airflow_predict": [],
        },
        dag,
    )
