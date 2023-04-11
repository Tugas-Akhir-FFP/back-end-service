# Base image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port that the application will be running on
EXPOSE 8000

# Start the Gunicorn server to serve the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]