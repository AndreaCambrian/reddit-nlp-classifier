# Reddit NLP Classifier Docker Image
# Andrea Oquendo Araujo - AIE1007

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 nlp_user && chown -R nlp_user:nlp_user /app
USER nlp_user

# Expose port (if you add a web interface later)
EXPOSE 8000

# Default command
CMD ["python", "interactive_demo.py"]