FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies including gevent
RUN pip install --no-cache-dir -r requirements.txt gunicorn gevent

# Copy application code
COPY . .

# Expose Cloud Run port
EXPOSE 8080

# Run using gunicorn with gevent worker (required for WebSocket support)
CMD ["gunicorn", "-b", ":8080", "app:app", "--worker-class", "gevent"]
