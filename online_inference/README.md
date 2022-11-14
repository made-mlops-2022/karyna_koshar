# MADE_MLOps_Homework_2

This project uses [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) dataset.

## Usage
This package allows you to use an online model to determine heart disease.
1. If you want to build image 
- From *online_inference/*, the command should look like this:
```sh
docker build -t karynakoshar/mlops_online_inference:v1 .
```
- From [DockerHub](https://hub.docker.com/repository/docker/karynakoshar/mlops_online_inference), the command should look like this:
```sh
docker pull karynakoshar/mlops_online_inference:v1
```
2. Run container with the following command:
```sh
docker run --name online_model -p 5000:5000 karynakoshar/mlops_online_inference:v1
```
3. Run tests in container:
```sh
docker exec -it online_model bash
pytest test_app.py
```
4. Run a script with requests to the service:
```sh
python request.py
```

**Note: optimizing docker image size**

- online_inference:v1  
The first version of dockerfile - size 674.9 MB;
- online_inference:v2  
--no-cache-dir flag has been added to pip - size 563.11 MB;
![docker images](https://user-images.githubusercontent.com/98235486/201635464-00838ea2-98ad-49cc-b1ac-8028181f1e3e.jpg)

- online_inference:v3   
Removed unnecessary packages, requirements.txt contains only necessary packages - size 563.11 MB;
- online_inference:v4   
Excluded all files that don't need to be copied, added .dockerignore - size 563.11 MB;
- online_inference:v5   
The instructions in dockerfile are combined - size 561.04 MB. 
