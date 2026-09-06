from __future__ import annotations

import json
import secrets
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from music_eval.listening import validate_response


def make_study_server(directory: Path, host: str, port: int) -> ThreadingHTTPServer:
    directory = directory.resolve()
    study = json.loads((directory / "study.json").read_text())
    responses = directory / "responses"
    responses.mkdir(exist_ok=True)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_POST(self) -> None:
            if self.path != "/submit":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if not 0 < length <= 1_000_000:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.send_error(HTTPStatus.BAD_REQUEST, "invalid JSON")
                return
            try:
                payload = validate_response(payload, study)
            except ValueError as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                return
            name = f"response-{secrets.token_hex(12)}.json"
            (responses / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            body = json.dumps({"saved": name}).encode()
            self.send_response(HTTPStatus.CREATED)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0].startswith("/responses"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            super().do_GET()

        def do_HEAD(self) -> None:
            if self.path.split("?", 1)[0].startswith("/responses"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            super().do_HEAD()

        def list_directory(self, path: str) -> None:  # type: ignore[override]
            self.send_error(HTTPStatus.NOT_FOUND)

    return ThreadingHTTPServer((host, port), Handler)


def serve_study(directory: Path, host: str, port: int) -> None:
    server = make_study_server(directory, host, port)
    print(f"listening study: http://{host}:{server.server_port}/")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
