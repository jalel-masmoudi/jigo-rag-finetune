# python:3.12-slim. numpy 2.5.4 publishes a cp312 wheel and not a cp311 wheel.
# Digest checked against the registry for the tag python:3.12-slim.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

WORKDIR /app

COPY requirements-cpu.txt requirements.txt .
# CPU torch first (requirements-cpu.txt), then the other packages.
# This image is for a machine without a GPU. Kaggle and Colab should not use it;
# they already have a CUDA build of torch.
RUN pip install --no-cache-dir -r requirements-cpu.txt \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Bind every interface. Hosts route to $PORT.
CMD ["sh", "-c", "python -m uvicorn serve.app:app --host 0.0.0.0 --port ${PORT}"]
