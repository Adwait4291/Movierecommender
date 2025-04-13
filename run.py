# run.py
from app import create_app
import os

# Create the app instance using the factory
app = create_app()

if __name__ == '__main__':
    # Use Flask's built-in server for development
    # For production, use a production-ready WSGI server like Gunicorn or uWSGI
    host = os.environ.get('FLASK_RUN_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_RUN_PORT', 5000))
    app.run(host=host, port=port) # Debug is controlled by FLASK_DEBUG in .flaskenv