# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and DVC
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip and uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY ml_services/ ./ml_services/
COPY app.py ./
COPY .env ./.env

# Install Python dependencies using uv
RUN uv pip install --system .

# Install DVC
RUN pip install --no-cache-dir "dvc[s3]"

# Run the two inference scripts once at build time
RUN python src/sandhya_aqua_erp/anomaly_detection/supply_chain/pipeline/infer/infer_feature_wise_stats.py \
 && python src/sandhya_aqua_erp/anomaly_detection/supply_chain/pipeline/infer/infer_cross_stage_yield_stats.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
