FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    python3-dev \
    gcc \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p /app/static /app/audio /app/exercises

# Copy application code
COPY . .

# Copy index.html to static folder for Flask to serve
RUN cp templates/index.html static/ 2>/dev/null || echo "index.html already in static"

# Make sure the audio directory exists and has proper permissions
RUN mkdir -p audio && chmod 755 audio

# Set environment variables for better stability
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV EVENTLET_NO_GREENDNS=yes
ENV PYTHONPATH=/app
ENV SDL_AUDIODRIVER=dummy
ENV MPLBACKEND=Agg

# Expose Cloud Run port
EXPOSE 8080

# Use a more stable gunicorn configuration
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:8080", "--timeout", "300", "--keep-alive", "2", "--max-requests", "1000", "--max-requests-jitter", "100", "wsgi:app"]