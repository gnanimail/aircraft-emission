FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy, streamlit==0.83.0

COPY aircraft_emission_api.py ./aircraft_emission_api.py

# expose port
EXPOSE 8501

# run the application:
CMD streamlit run aircraft_emission_api.py