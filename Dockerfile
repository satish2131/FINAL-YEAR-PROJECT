FROM python:3.10-slim

# Install necessary C++ build tools and OpenCV system dependencies
RUN apt-get update -y && apt-get install -y \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# CRITICAL FIX for Render OOM (Out of Memory) Issue:
# By default, pip tries to compile dlib using all available CPU cores at once. 
# Each core takes ~1.5GB of RAM, easily maxing out Render's 8GB limit.
# Forcing it to 1 job (-j1) makes it slower but uses only ~1GB of RAM!
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV MAKEFLAGS="-j1"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

CMD ["python", "deploy.py"]
