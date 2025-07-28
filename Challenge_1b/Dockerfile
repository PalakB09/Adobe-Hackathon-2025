# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY processor.py .

# Entry point: You can override these at run-time
ENTRYPOINT ["python", "processor.py"]