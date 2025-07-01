# Use a slim Python image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install necessary packages including CMake
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ cmake python3-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Command to run your application
CMD ["gunicorn", "-w 4", "-k uvicorn.workers.UvicornWorker", "src.api.main:app", "--bind", "0.0.0.0:8000"]