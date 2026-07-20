FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "predict.py", "--server.port=8080", "--server.address=0.0.0.0"]