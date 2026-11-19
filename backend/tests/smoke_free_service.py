"""Smoke-check the REAL free-tier service with its lightweight text estimator.

Runs only locally or in CI. Uses invented data and an isolated temporary SQLite DB.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
PORT = 8837
BASE = f"http://127.0.0.1:{PORT}"


def main():
    with tempfile.TemporaryDirectory(prefix="disastersignal-free-smoke-") as folder:
        env = os.environ.copy()
        env.update({
            "LITE_MODEL_ENABLED": "true",
            "DATABASE_URL": f"sqlite:///{Path(folder, 'smoke.db')}",
            "UPLOAD_DIR": str(Path(folder, "uploads")),
            "JWT_SECRET": "ci-smoke-only-do-not-use-in-production-f7b97cb123c",
            "DEMO_ACCOUNTS_ENABLED": "false",
            "PYTHONPATH": str(ROOT / "backend"),
            "CORS_ORIGINS": '["https://disastersignal-bhuvan.vercel.app"]',
        })
        logs = Path(folder, "server.log")
        with logs.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app",
                 "--host", "127.0.0.1", "--port", str(PORT)],
                env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            )
            try:
                response = None
                for _ in range(30):
                    if proc.poll() is not None:
                        raise RuntimeError("API exited during startup")
                    try:
                        response = httpx.get(BASE + "/health", timeout=2)
                        if response.status_code == 200:
                            break
                    except (httpx.TransportError, httpx.TimeoutException):
                        pass
                    time.sleep(1)
                assert response is not None and response.status_code == 200
                health = response.json()
                assert health["database"] == "connected"
                assert health["text_model_loaded"] is True, health
                assert health["vision_model_loaded"] is False
                assert health["model_variant"] == "free-tier-supervised-retrieval"
                assert health["storage_persistence"] == "ephemeral"
                assert health["available_inference_modes"] == ["text"]

                report = "Flood waters have entered nearby homes; neighbours need evacuation help."
                pred = httpx.post(BASE + "/predict", data={"text_report": report}, timeout=20)
                assert pred.status_code == 200, pred.text
                assert 0 <= pred.json()["priority_score"] <= 100
                assert pred.json()["class_probabilities"] == {}

                incident = httpx.post(BASE + "/incidents", data={
                    "description": report,
                    "latitude": "28.6139",
                    "longitude": "77.209",
                    "address_text": "Fictional test location",
                }, timeout=20)
                assert incident.status_code == 201, incident.text
                incident_id = incident.json()["id"]
                summary = httpx.get(BASE + "/dashboard/stats", timeout=10)
                assert summary.status_code == 200
                assert summary.json()["total_incidents"] >= 1
                assert httpx.patch(BASE + f"/incidents/{incident_id}",
                                   json={"status": "Completed"}).status_code == 401
                print("FREE SERVICE SMOKE PASS: health + real lite text + incident + dashboard + auth")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                if sys.exc_info()[0] is not None:
                    print("Server logs:\n" + logs.read_text(encoding="utf-8")[-6000:])


if __name__ == "__main__":
    main()
