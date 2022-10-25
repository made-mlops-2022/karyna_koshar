from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ml_project",
    packages=['ml_project', 'tests', 'ml_project.enities', 'ml_project.models', 'ml_project.features'],
    install_requires=required,
)
