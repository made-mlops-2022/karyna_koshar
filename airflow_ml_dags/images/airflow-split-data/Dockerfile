FROM python:3.8-slim-buster
RUN python -m pip install --upgrade pip

COPY split_data.py requirements.txt /

RUN pip install --default-timeout=9000 --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "split_data.py"]