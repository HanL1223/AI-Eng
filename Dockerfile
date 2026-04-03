#Base Image
FROM python:3.12-slim

#Enviornment varialbe
ENV PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1

#System deencencies
RUN apt-get update && \
    apt-get install-y --no-install-recommends \
        build-essential \
        curl
    && rm -rf /var/lib/apt/lists/*


#working dir

WORKDIR /app

#install depencencies
COPY pyproject.toml .

#install python depencencies
RUN pip install --no-chache-dir .

#Copy sourec code
COPY . .


#Crate directories
RUN mkdir -p logs docs

#expose port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8000/api/health || exit 1


CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]