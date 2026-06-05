FROM python:3.11-slim

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code, model weights, and student metadata
COPY app          /app/app
COPY models       /app/models
COPY STUDENT.json /app/STUDENT.json

# Make sure Python can find the app package from /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "/app/app/cli.py"]
