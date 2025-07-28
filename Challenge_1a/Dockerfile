FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y libpoppler-cpp-dev tesseract-ocr && \
    pip install --no-cache-dir -r requirements.txt

# Copy your whole app
COPY . .

# Download required NLTK and spaCy models
RUN python -m nltk.downloader punkt && \
    python -m spacy download en_core_web_sm

# Run your script
CMD ["python", "process_pdfs.py"]
