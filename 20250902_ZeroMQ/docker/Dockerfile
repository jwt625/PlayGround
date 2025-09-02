# ZeroMQ Cluster Learning Environment
# Based on cluster_runner's container setup

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libzmq3-dev \
    pkg-config \
    curl \
    net-tools \
    iproute2 \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 zmquser && \
    chown -R zmquser:zmquser /app
USER zmquser

# Expose common ZeroMQ ports
EXPOSE 5555-5580

# Default command
CMD ["python", "-c", "print('ZeroMQ cluster container ready. Use docker-compose to start services.')"]
