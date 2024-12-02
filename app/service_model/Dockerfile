# Use a base image of Python
FROM python:3.11

# Install ffmpeg and necessary dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and models into the container
COPY model_server.py .
COPY parking_spot_detection.py .
COPY models ./models

# Expose the port that the server will run on
EXPOSE 8000

# Command to run the server
CMD ["python", "model_server.py"]