# Use Python 3.12 slim (Debian 12 Bookworm for stability)
FROM python:3.12-slim-bookworm

# Set working directory in container
WORKDIR /app

# Install system dependencies for ML libraries and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Copy and install dependencies first for caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure static directory exists
RUN mkdir -p static/images

# Expose FastAPI port
EXPOSE 8000

# Environment setup
ENV PYTHONPATH=/app

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
