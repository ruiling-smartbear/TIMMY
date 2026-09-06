from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from music_eval.listening_server import make_study_server


def test_server_validates_submissions_and_hides_responses(tmp_path):
    study = {
        "study_id": "study-1",
        "criteria": ["overall"],
        "trials": [{"id": "trial-1"}],
    }
    (tmp_path / "study.json").write_text(json.dumps(study))
    (tmp_path / "index.html").write_text("study")
    server = make_study_server(tmp_path, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        incomplete = urllib.request.Request(
            f"{base}/submit",
            data=json.dumps({"study_id": "study-1", "rater_id": "r"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(incomplete)
        assert rejected.value.code == 400

        valid = urllib.request.Request(
            f"{base}/submit",
            data=json.dumps(
                {
                    "study_id": "study-1",
                    "rater_id": "r",
                    "answers": [
                        {"trial_id": "trial-1", "ratings": {"overall": "tie"}}
                    ],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(valid) as response:
            assert response.status == 201
        saved = next((tmp_path / "responses").glob("*.json"))
        assert json.loads(saved.read_text())["rater_id"] == "r"

        with pytest.raises(urllib.error.HTTPError) as hidden:
            urllib.request.urlopen(f"{base}/responses/{saved.name}")
        assert hidden.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
