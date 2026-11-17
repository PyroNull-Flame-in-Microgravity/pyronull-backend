FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8001

WORKDIR /app

# Copy only the release-allowlisted model and CSV files. Unresolved PSI-107/PSI-69
# artifacts are excluded from the build context and are not copied into the image.
COPY models/PSI_159_burntime_regressor_Predictive_Model.joblib ./models/
COPY models/psi10/inverse_flame_tad_predictor.joblib ./models/psi10/
COPY models/psi10/inverse_flame_zst_predictor.joblib ./models/psi10/
COPY models/psi10/inverse_flame_pressure_rise_predictor.joblib ./models/psi10/
COPY models/psi159/PSI_159_ignition_classifier_Predictive_Model.joblib ./models/psi159/
COPY data/raw/psi_full_dataset.csv ./data/raw/
COPY data/raw/psi_investigations_dataset.csv ./data/raw/

# Validate the complete service asset set before spending time installing large ML dependencies.
RUN python -c 'from pathlib import Path; required = ["models/PSI_159_burntime_regressor_Predictive_Model.joblib", "models/psi10/inverse_flame_tad_predictor.joblib", "models/psi10/inverse_flame_zst_predictor.joblib", "models/psi10/inverse_flame_pressure_rise_predictor.joblib", "models/psi159/PSI_159_ignition_classifier_Predictive_Model.joblib", "models/psi107/fuel_classifier_pipeline.joblib", "models/psi69/dnn_cls_model.keras", "models/psi69/dnn_cls_preprocessor.joblib", "data/raw/psi_full_dataset.csv", "data/raw/psi_investigations_dataset.csv"]; missing = [path for path in required if not Path(path).is_file()]; assert not missing, "Missing runtime artifacts (supply approved model files and runtime CSVs): " + ", ".join(missing)'

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# PyroNull runs CPU inference; avoid pulling the CUDA-enabled PyTorch wheel.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir --no-deps -r requirements.lock \
    && python -m pip check

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8001') + '/health', timeout=3).read()"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
