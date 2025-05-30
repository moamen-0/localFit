#!/usr/bin/env python

# This module provides the WSGI entry point for gunicorn when deployed to Cloud Run

import os
import logging

# Setup logging before importing the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app import app, socketio
    logger.info("Successfully imported app and socketio")
except Exception as e:
    logger.error(f"Failed to import app: {e}")
    raise

# Ensure we have a proper application object
application = socketio

if __name__ == "__main__":
    # This allows the app to be run using Python directly (without gunicorn)
    # This will be used by the CMD in Dockerfile
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting application on port {port}")
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=True)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise