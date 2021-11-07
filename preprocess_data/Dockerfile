FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy pandas

COPY preprocess.py ./preprocess.py
COPY Emissions_data.csv ./Emissions_data.csv

ENTRYPOINT [ "python", "preprocess.py" ]
