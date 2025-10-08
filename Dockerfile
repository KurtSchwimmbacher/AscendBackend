# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies for ML libraries and OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    libfontconfig1 \
    libxss1 \
    libxtst6 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create static directories
RUN mkdir -p static/images

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
