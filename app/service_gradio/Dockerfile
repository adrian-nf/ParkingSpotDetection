# Use a base image of Python
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY app.py .

# Expose the port that Gradio will run on
EXPOSE 7860

# Command to run the Gradio application
CMD ["python", "app.py"]