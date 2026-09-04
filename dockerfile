# Dockerfile
# Containerizes the Flask ML application for deployment to Google Cloud Run

# ==========================================================
# 1. Base Image
# ==========================================================
# FROM defines the starting point for your image.
# python:3.10-slim = Official Python 3.10 image with minimal Debian OS.
# 'slim' saves ~700MB compared to the full python image.
# Always use a specific version instead of 'latest'.

FROM python:3.10-slim


# ==========================================================
# 2. Working Directory
# ==========================================================
# Sets /app as the working directory inside the container.
# Docker creates it automatically if it doesn't exist.

WORKDIR /app


# ==========================================================
# 3. Copy Requirements First (Layer Caching)
# ==========================================================
# Copy requirements.txt before application code.
# This allows Docker to cache installed packages and
# avoid reinstalling dependencies if only the source code changes.

COPY requirements.txt .


# ==========================================================
# 4. Install Dependencies
# ==========================================================
# Installs all Python packages listed in requirements.txt.
# --no-cache-dir reduces image size by removing pip cache.

RUN pip install --no-cache-dir -r requirements.txt


# ==========================================================
# 5. Copy Application Code
# ==========================================================
# Copy Flask application into the container.

COPY app.py .


# ==========================================================
# 6. Copy Trained ML Model
# ==========================================================
# Copies model.pkl and scaler.pkl into the image.

COPY model/ model/


# ==========================================================
# 7. Security - Run as Non-Root User
# ==========================================================
# Create a normal user instead of running the container as root.
# Recommended for Cloud Run and production deployments.

RUN useradd -m appuser && \
    chown -R appuser:appuser /app

USER appuser


# ==========================================================
# 8. Expose Port
# ==========================================================
# Documents that the application listens on port 5000.

EXPOSE 5000


# ==========================================================
# 9. Environment Variable
# ==========================================================
# Cloud Run overrides PORT automatically.
# Locally, the default value is 5000.

ENV PORT=5000


# ==========================================================
# 10. Start Application
# ==========================================================
# Run the Flask application using Gunicorn.
#
# --bind 0.0.0.0:5000
#     Listen on all network interfaces.
#
# --workers 1
#     One worker process (sufficient for Cloud Run).
#
# --timeout 120
#     Kill requests taking longer than 120 seconds.
#
# app:app
#     First "app" = app.py
#     Second "app" = Flask application object

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]