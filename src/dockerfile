FROM python:3.11-slim

# 1. Environment variables to help Python find 'src'
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# 2. Set the working directory
WORKDIR /app

# 3. Install build tools for certain Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Download NLP data
RUN python -m nltk.downloader punkt wordnet

# 6. Copy the entire project folder into the Docker image
# This copies everything into /app, so your 'src' folder becomes /app/src
COPY . .

# 7. Expose the port Northflank expects
EXPOSE 8000

# 8. THE CRITICAL FIX:
# We force the PYTHONPATH to be the current directory (.) inside the command.
CMD ["sh", "-c", "PYTHONPATH=. uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]