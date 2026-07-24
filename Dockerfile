FROM python:3.11-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5050

WORKDIR /app

# Install only required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Render port
EXPOSE 5050

# Start Gunicorn with a single worker to reduce memory usage
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]