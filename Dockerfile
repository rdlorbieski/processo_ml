FROM python:3.7

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN [ "python3"]

COPY . /app

CMD ["gunicorn", "--workers=2","--threads=2","--timeout=360", "--worker-class=gthread", "--bind", "0.0.0.0:5000", "app:app"]