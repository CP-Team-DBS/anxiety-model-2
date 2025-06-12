FROM python:3.11-slim

COPY ./api /app
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
