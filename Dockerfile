
# Use the official Python 3.10.6 image as the base image
FROM python:3.10.6-buster

# Create a directory for your project within the container
WORKDIR /app

# Copy your project files and requirements.txt into the container
COPY eficient_frontier /app/eficient_frontier
COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
COPY finalized_model.sav /app/finalized_model.sav
COPY finalized_preprocessor.sav /app/finalized_preprocessor.sav
COPY sp500_all.pkl /app/sp500_all.pkl

# Upgrade pip and install project dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
RUN pip install -e .
# Specify the command to run your application
CMD uvicorn eficient_frontier.api.app:app --host 0.0.0.0 --port $PORT
