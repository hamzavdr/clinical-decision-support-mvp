# Use a specific Python version for reproducibility
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# --- Layer Caching for Dependencies ---
# 1. Copy only the requirements.txt file first.
# Docker will cache this layer. It will only re-run the pip install
# if the requirements.txt file itself changes.
COPY requirements.txt .

# 2. Install the dependencies.
# This is the slow step that will now be cached.
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code ---
# 3. Copy the rest of your application code.
# Since your app code changes more frequently than your dependencies,
# this layer will be rebuilt often, but the slow pip install step above will not.
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]