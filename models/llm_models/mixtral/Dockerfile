
# Use the official PyTorch image as a base image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /opt/project

# Copy the application code
COPY . /opt/project

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a new user and group
RUN groupadd -r dsi && useradd -r -g dsi buznahy

# Set the user and group
USER buznahy:dsi

# Copy the rest of the application code into the container
COPY . .

# Set the entry point to your script
ENTRYPOINT ["python", "modeling_mixtral.py"]