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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn eventlet gevent gevent-websocket

# Create directory structure
RUN mkdir -p /app/static /app/audio /app/exercises

# Copy application code
COPY . .

# Copy index.html to static folder for Flask to serve
RUN cp templates/index.html static/

# Make sure the audio directory exists
RUN mkdir -p audio

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV EVENTLET_NO_GREENDNS=yes
ENV PYTHONPATH=/app

# Expose Cloud Run port
EXPOSE 8080

# Run using gunicorn with eventlet worker (required for WebSocket support)
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:8080", "wsgi:app"]