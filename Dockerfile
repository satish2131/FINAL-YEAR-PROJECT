FROM python:3.10-slim

# Install necessary C++ build tools and OpenCV system dependencies
RUN apt-get update -y && apt-get install -y \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory directly in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (this might take a few minutes for dlib)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
# This includes the /dataset and /database folders so the app has its data
COPY . .

# Ensure Python output is logged immediately in Render
ENV PYTHONUNBUFFERED=1

# Expose the default Waitress port
EXPOSE 5000

# Command to automatically start the production server
CMD ["python", "deploy.py"]
