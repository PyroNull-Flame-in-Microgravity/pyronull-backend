from pathlib import Path
import re
import subprocess
import sys

from fastapi.testclient import TestClient

from app.main import app


ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def test_health_and_swagger_endpoints_are_available():
    assert client.get("/health").json() == {"status": "ok", "service": "pyronull-backend"}
    assert client.get("/docs").status_code == 200
    assert "NASA provenance and scientific validity are unverified" in app.description


def test_dockerfile_is_production_oriented_and_uses_dynamic_port():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "python:3.11-slim" in dockerfile
    assert "--host 0.0.0.0" in dockerfile
    assert "${PORT:-8001}" in dockerfile
    assert "--reload" not in dockerfile
    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "COPY models/PSI_159_burntime_regressor_Predictive_Model.joblib ./models/" in dockerfile
    assert "COPY models/psi10/inverse_flame_tad_predictor.joblib ./models/psi10/" in dockerfile
    assert "COPY models/psi10/inverse_flame_zst_predictor.joblib ./models/psi10/" in dockerfile
    assert "COPY models/psi10/inverse_flame_pressure_rise_predictor.joblib ./models/psi10/" in dockerfile
    assert "COPY models/psi159/PSI_159_ignition_classifier_Predictive_Model.joblib ./models/psi159/" in dockerfile
    assert "COPY data/raw/psi_full_dataset.csv ./data/raw/" in dockerfile
    assert "COPY data/raw/psi_investigations_dataset.csv ./data/raw/" in dockerfile
    assert "COPY models ./models" not in dockerfile
    assert "Missing runtime artifacts" in dockerfile
    assert "COPY requirements.lock ./requirements.lock" in dockerfile
    assert "python -m pip check" in dockerfile
    assert dockerfile.index("Missing runtime artifacts") < dockerfile.index("RUN apt-get update")


def test_dockerignore_excludes_secrets_but_keeps_runtime_assets():
    entries = (ROOT / ".dockerignore").read_text().splitlines()
    assert ".env" in entries
    assert ".env.*" in entries
    assert ".git" in entries
    assert ".venv" in entries
    assert "models" not in entries
    assert "data/raw" not in entries
    assert "data/external" in entries
    assert "/models/psi107/fuel_classifier_pipeline.joblib" in entries
    assert "/models/psi69/dnn_cls_model.keras" in entries
    assert "/models/psi69/dnn_cls_preprocessor.joblib" in entries
    assert "/PYRONULL_COMPLETE_BACKEND_AUDIT.md" in entries
    assert "/PYRONULL_DOCKER_INTEGRATION_REPORT.md" in entries
    assert "/PYRONULL_FINAL_RELEASE_REPORT.md" in entries
    assert "/PYRONULL_STAGING_REVIEW_REPORT.md" in entries
    assert "**/credentials.json" in entries
    assert "**/*.pem" in entries
    assert "credentials" in entries
    assert "secrets" in entries


def test_uncleared_model_artifacts_are_git_ignored():
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    assert "/models/psi107/fuel_classifier_pipeline.joblib" in gitignore
    assert "/models/psi69/dnn_cls_model.keras" in gitignore
    assert "/models/psi69/dnn_cls_preprocessor.joblib" in gitignore
    assert "/PYRONULL_COMPLETE_BACKEND_AUDIT.md" in gitignore
    assert "/PYRONULL_DOCKER_INTEGRATION_REPORT.md" in gitignore
    assert "/PYRONULL_FINAL_RELEASE_REPORT.md" in gitignore
    assert "/PYRONULL_STAGING_REVIEW_REPORT.md" in gitignore
    assert "credentials/" in gitignore
    assert "secrets/" in gitignore


def test_docker_artifact_guard_passes_complete_context_and_lists_excluded_models(tmp_path):
    dockerfile = (ROOT / "Dockerfile").read_text()
    match = re.search(r"^RUN python -c '([^']+)'$", dockerfile, flags=re.MULTILINE)
    assert match is not None
    guard = match.group(1)
    required = [
        "models/PSI_159_burntime_regressor_Predictive_Model.joblib",
        "models/psi10/inverse_flame_tad_predictor.joblib",
        "models/psi10/inverse_flame_zst_predictor.joblib",
        "models/psi10/inverse_flame_pressure_rise_predictor.joblib",
        "models/psi159/PSI_159_ignition_classifier_Predictive_Model.joblib",
        "models/psi107/fuel_classifier_pipeline.joblib",
        "models/psi69/dnn_cls_model.keras",
        "models/psi69/dnn_cls_preprocessor.joblib",
        "data/raw/psi_full_dataset.csv",
        "data/raw/psi_investigations_dataset.csv",
    ]

    complete_context = tmp_path / "complete"
    complete_context.mkdir()
    for relative_path in required:
        target = complete_context / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    complete_result = subprocess.run(
        [sys.executable, "-c", guard], cwd=complete_context, capture_output=True, text=True
    )
    assert complete_result.returncode == 0

    public_context = tmp_path / "public"
    public_context.mkdir()
    excluded = {
        "models/psi107/fuel_classifier_pipeline.joblib",
        "models/psi69/dnn_cls_model.keras",
        "models/psi69/dnn_cls_preprocessor.joblib",
    }
    for relative_path in set(required) - excluded:
        target = public_context / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    public_result = subprocess.run(
        [sys.executable, "-c", guard], cwd=public_context, capture_output=True, text=True
    )
    assert public_result.returncode != 0
    assert "Missing runtime artifacts" in public_result.stderr
    for relative_path in excluded:
        assert relative_path in public_result.stderr


def test_pytest_is_kept_out_of_the_runtime_dependency_set():
    runtime_requirements = (ROOT / "requirements.txt").read_text()
    dev_requirements = (ROOT / "requirements-dev.txt").read_text()
    assert "pytest" not in runtime_requirements
    assert "pytest" in dev_requirements
    assert "-r requirements.txt" in dev_requirements


def test_environment_file_is_ignored_by_git_and_debug_is_disabled():
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in gitignore
    assert ".env.*" in gitignore
    assert "!.env.example" in gitignore
    assert app.debug is False
