FROM python:3.8-slim-buster
RUN python -m pip install --upgrade pip

COPY validate.py requirements.txt /

RUN pip install --default-timeout=9000 --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "validate.py"]