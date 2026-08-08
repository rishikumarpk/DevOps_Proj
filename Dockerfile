# Self-contained image: dependencies + source + the trained model, serving the
# FastAPI app with uvicorn. `docker run -p 8000:8000 <image>` and you're live.
FROM python:3.12-slim

WORKDIR /app

# Install deps first so this layer is cached until requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code and the trained model artifact into the image.
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# 0.0.0.0 so the server is reachable from outside the container.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
