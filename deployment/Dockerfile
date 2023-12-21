FROM python:latest

WORKDIR /app

#RUN pip install --upgrade pip

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

VOLUME /app/models/

COPY ["predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
