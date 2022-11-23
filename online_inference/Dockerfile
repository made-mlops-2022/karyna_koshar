FROM python:3.8-slim-buster
COPY app.py validator_app.py test_app.py start_app.sh requirements.txt ./online_inference/
WORKDIR /online_inference

RUN python -m pip install --upgrade pip \
    pip install --no-cache-dir -r requirements.txt

ENV PATH_DATA='data/heart_cleveland_upload.csv' \
    PATH_TO_MODEL='model/model.pkl'

CMD ["bash", "start_app.sh"]