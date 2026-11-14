FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# The pinned torch build is the CPU wheel. A CUDA host can replace the index.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Bind every interface. Render and other hosts route to $PORT.
CMD ["sh", "-c", "python -m uvicorn serve.app:app --host 0.0.0.0 --port ${PORT}"]
