FROM python:3.7-slim

WORKDIR /app

RUN pip install scikit-learn numpy streamlit joblib

COPY emission_api.py ./emission_api.py

# expose port
EXPOSE 8501

# run the application:
ENTRYPOINT ["streamlit", "run", "emission_api.py", "--"]
