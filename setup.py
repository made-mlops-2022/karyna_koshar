from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ml_project",
    packages=['ml_project', 'tests', 'ml_project.enities', 'ml_project.models', 'ml_project.features'],
    entry_points={
        "console_scripts": [
            "ml_project_train = ml_project.train_pipeline:train_pipeline",
            "ml_project_predict = ml_project.predict:predict_model",
            "ml_project_eda = ml_project.eda:eda_pandas_profiling",

        ]
    },
    install_requires=required,
)
