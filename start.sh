#!/bin/bash
# =============================================================================
# Knowledge Assistant — Startup Script
# Runs ingestion (idempotent), then starts FastAPI + Streamlit
# =============================================================================

set -e

echo "=============================================="
echo "  Intelligent Knowledge Assistant"
echo "=============================================="

# Run document ingestion (idempotent — skips unchanged docs)
echo ""
echo "📚 Running document ingestion..."
python -I -m scripts.ingest

# Start FastAPI in the background
echo ""
echo "🚀 Starting FastAPI server on port ${API_PORT:-8000}..."
uvicorn src.api.app:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --log-level info &

FASTAPI_PID=$!

# Wait for FastAPI to be ready
echo "⏳ Waiting for API to be ready..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:${API_PORT:-8000}/api/v1/health" > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    sleep 1
done

# Start Streamlit
echo ""
echo "🎨 Starting Streamlit UI on port ${STREAMLIT_PORT:-8501}..."
streamlit run ui/app.py \
    --server.port "${STREAMLIT_PORT:-8501}" \
    --server.address "0.0.0.0" \
    --server.headless true \
    --browser.gatherUsageStats false &

STREAMLIT_PID=$!

echo ""
echo "=============================================="
echo "  ✅ All services running!"
echo "  API:       http://localhost:${API_PORT:-8000}/docs"
echo "  UI:        http://localhost:${STREAMLIT_PORT:-8501}"
echo "=============================================="

# Wait for either process to exit
wait -n $FASTAPI_PID $STREAMLIT_PID
exit $?
