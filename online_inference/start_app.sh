#!/bin/bash

mkdir -p 'data'
mkdir -p 'model'

if [ -z $PATH_DATA ]
then
    export PATH_DATA='data/heart_cleveland_upload.csv'
fi

if [ -z $PATH_TO_MODEL ]
then
    export PATH_TO_MODEL='model/model.pkl'
fi

if [[ ! -f $PATH_DATA ]]
then
    wget -q -O $PATH_DATA 'https://drive.google.com/uc?export=download&id=1gOtgHlL-pm8wqQq8aYzYyoPoAZhGPMuo' 
    echo 'data downloaded'
else 
    echo 'data is already loaded'
fi

if [[ ! -f $PATH_TO_MODEL ]]
then
    wget -q -O $PATH_TO_MODEL 'https://drive.google.com/uc?export=download&id=1RxOo6Bdfkw4uSP1KI4X4-xAvNH1CMMkB'  
    echo 'model downloaded'
else
    echo 'model is already loaded'
fi

uvicorn app:app --reload --host 0.0.0.0 --port 5000