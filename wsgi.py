#!/usr/bin/env python

# This module provides the WSGI entry point for gunicorn when deployed to Cloud Run

from app import app, socketio

if __name__ == "__main__":
    # This allows the app to be run using Python directly (without gunicorn)
    # This will be used by the CMD in Dockerfile
    import os
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port)