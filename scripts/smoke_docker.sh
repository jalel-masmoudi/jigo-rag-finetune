#!/bin/sh
# Build the service image and hit /health, /query, and a blank question.
# JIGO_GENERATOR=extractive skips Qwen so this can run on a CPU.
set -eu
cd "$(dirname "$0")/.."

docker build -t jigo-rag-finetune .
docker rm -f jigo-smoke >/dev/null 2>&1 || true
docker run -d --name jigo-smoke -p 8000:8000 \
  -e PORT=8000 \
  -e JIGO_GENERATOR=extractive \
  jigo-rag-finetune

cleanup() {
  docker rm -f jigo-smoke >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
i=0
while [ "$i" -lt 30 ]
do
  if curl -sf http://127.0.0.1:8000/health >/dev/null
  then
    ready=1
    break
  fi
  i=$((i + 1))
  sleep 2
done
if [ "$ready" -ne 1 ]
then
  echo "health check did not succeed" >&2
  docker logs jigo-smoke >&2 || true
  exit 1
fi

hit() {
  label=$1
  payload=$2
  expect=$3
  body=$(mktemp)
  code=$(curl -sS -o "$body" -w '%{http_code}' -H 'content-type: application/json' -d "$payload" http://127.0.0.1:8000/query)
  echo "$label"
  echo "HTTP $code"
  cat "$body"
  echo
  rm -f "$body"
  if [ "$code" != "$expect" ]
  then
    echo "expected HTTP $expect" >&2
    exit 1
  fi
}

echo "GET /health"
curl -sS -D - http://127.0.0.1:8000/health
echo
hit "POST /query" '{"question":"What is the TTS Handbook?"}' 200
hit "POST /query blank" '{"question":"   "}' 422
