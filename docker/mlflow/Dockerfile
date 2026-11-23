FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.17.2 psycopg2-binary "boto3>=1.35,<2"

RUN useradd --create-home mlflow && mkdir /mlartifacts && chown mlflow /mlartifacts
USER mlflow

EXPOSE 5000
# Shell form so $MLFLOW_BACKEND_STORE_URI is expanded at start
CMD mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root /mlartifacts \
    --allowed-hosts "*"
