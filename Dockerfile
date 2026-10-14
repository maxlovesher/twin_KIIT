FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY glucotwin ./glucotwin
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY tests ./tests
COPY .streamlit ./.streamlit
COPY data ./data
COPY models ./models
COPY reports ./reports
COPY docs ./docs

# If the trained artefacts were not copied in (fresh clone without data), build them once.
RUN test -f models/glucotwin_bundle.joblib || python scripts/build_all.py

EXPOSE 8501 8000
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
