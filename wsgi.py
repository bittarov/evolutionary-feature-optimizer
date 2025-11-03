"""
WSGI Entry Point for DataOptima Platform
Production deployment configuration
"""
from server import app

# This is the application entry point for WSGI servers (e.g., Gunicorn, uWSGI)
if __name__ == "__main__":
    app.run()

