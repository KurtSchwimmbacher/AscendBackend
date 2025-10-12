# Use Python 3.12 slim (Debian 12 Bookworm)
FROM python:3.12-slim-bookworm

# Make logs appear in real time & avoid pip cache
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install only the system libraries needed for CV/ML and runtime.
# build-essential is included because some pip packages may need to compile C extensions;
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
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

# Copy and install Python deps first for caching
COPY requirements.txt .

# Upgrade pip/tools and install Python requirements
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Ensure static folder exists and set permissions for a non-root user
RUN mkdir -p /app/static/images \
 && chown -R 1000:1000 /app

# Switch to a non-root user (safer). Render runs containers as root by default,
# but running app code as non-root is a best practice.
USER 1000

# Expose default port (can be overridden by PORT env var on Render)
EXPOSE ${PORT}

# Start the app. Use sh -c so ${PORT} expansion works and exec so process receives signals.
# --proxy-headers is helpful when behind proxies/load-balancers.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --lifespan auto"]
