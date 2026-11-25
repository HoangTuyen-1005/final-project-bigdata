FROM python:3.12-slim-bookworm

ENV PYTHONPATH=/app:/app/services/data_service \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /app

ARG LOCAL_UID=1000
ARG LOCAL_GID=1000

COPY services/data_service/requirements.txt /tmp/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless passwd procps \
    && rm -rf /var/lib/apt/lists/* \
    && (getent group "$LOCAL_GID" >/dev/null || groupadd --gid "$LOCAL_GID" spark) \
    && (getent passwd "$LOCAL_UID" >/dev/null || useradd --uid "$LOCAL_UID" --gid "$LOCAL_GID" \
        --no-create-home --home-dir /tmp --no-log-init --shell /usr/sbin/nologin spark) \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import os, urllib.request; jars = '/usr/local/lib/python3.12/site-packages/pyspark/jars'; urllib.request.urlretrieve('https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar', os.path.join(jars, 'hadoop-aws-3.3.4.jar')); urllib.request.urlretrieve('https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar', os.path.join(jars, 'aws-java-sdk-bundle-1.12.262.jar'))" \
    && python -c "from pyspark.ml import Pipeline, PipelineModel; from pyspark.ml.functions import vector_to_array"

ENV SPARK_HOME=/usr/local/lib/python3.12/site-packages/pyspark
ENV PATH=${SPARK_HOME}/bin:${PATH} \
    PYSPARK_PYTHON=/usr/local/bin/python \
    PYSPARK_DRIVER_PYTHON=/usr/local/bin/python \
    HADOOP_USER_NAME=spark

COPY config.py project_state.py ./
COPY contracts/ ./contracts/
COPY services/__init__.py ./services/
COPY services/data_service/ ./services/data_service/
