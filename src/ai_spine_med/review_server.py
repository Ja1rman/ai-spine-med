"""Private review server with bounded DICOM upload and isolated model inference.

REVIEW_TOKEN is mandatory. REVIEW_ROOT defaults to this script's directory.
Serve /r/<token>/ on 127.0.0.1:8768; never print the capability URL or access logs.
Original cohort DICOM routes read allowlisted ZIP members. Uploaded studies
are processed separately under the private upload root.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from .review_uploads import UploadRejected, UploadStore

MIMES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
    ".zip": "application/zip",
}
SECURITY_HEADERS = {
    "Cache-Control": "private, no-store, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-Robots-Tag": "noindex, nofollow, noarchive, nosnippet",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
}
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{24,128}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FILENAME = re.compile(r"^[A-Za-z0-9_.-]{1,180}$")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate source-index key")
        result[key] = value
    return result


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(
        self,
        package_root: Path,
        token: str,
        port: int = 8768,
        *,
        uploads: UploadStore | None = None,
        public_origin: str = "",
        host: str = "127.0.0.1",
        max_http_connections: int = 24,
    ):
        if not _TOKEN.fullmatch(token):
            raise ValueError("REVIEW_TOKEN must be an opaque URL-safe capability")
        if host not in {"127.0.0.1", "0.0.0.0"}:
            raise ValueError("Unsupported bind host")
        if type(max_http_connections) is not int or not 1 <= max_http_connections <= 128:
            raise ValueError("Invalid HTTP concurrency limit")
        self.http_slots = threading.BoundedSemaphore(max_http_connections)
        self.package_root = Path(package_root).resolve()
        self.site_root = (self.package_root / "site").resolve()
        if not self.site_root.is_dir():
            raise ValueError("Missing review site")
        self.token_bytes = token.encode("ascii")
        self.prefix = "/r/" + token + "/"
        if uploads is not None and uploads.root.is_relative_to(self.site_root):
            raise ValueError("Uploads must be outside the static site")
        if public_origin:
            parsed_origin = urlsplit(public_origin)
            if (
                parsed_origin.scheme not in {"http", "https"}
                or not parsed_origin.netloc
                or parsed_origin.path not in {"", "/"}
                or parsed_origin.query
                or parsed_origin.fragment
            ):
                raise ValueError("REVIEW_ORIGIN must be one exact origin")
            public_origin = f"{parsed_origin.scheme}://{parsed_origin.netloc}"
        if uploads is not None and not public_origin:
            raise ValueError("An exact public origin is required for uploads")
        self.uploads, self.public_origin = uploads, public_origin
        source = json.loads(
            (self.package_root / "private-source-index.json").read_text(),
            object_pairs_hook=_unique_object,
        )
        if source.get("schema_version") != 1 or not isinstance(source.get("routes"), dict):
            raise ValueError("Invalid private source index")
        self.dicom_routes, self.allowed_zips = {}, set()
        for route, entry in source["routes"].items():
            parts = PurePosixPath(route).parts
            member = entry.get("member", "")
            zip_relative = Path(entry.get("zip_path", ""))
            archive = (self.package_root / zip_relative).resolve()
            if (
                not route.startswith("dicom/")
                or not route.endswith(".dcm")
                or route.startswith("/")
                or "\\" in route
                or any(p in {".", ".."} for p in parts)
                or zip_relative.is_absolute()
                or not archive.is_relative_to(self.site_root / "originals")
                or archive.suffix != ".zip"
                or not archive.is_file()
                or not _FILENAME.fullmatch(archive.name)
                or not isinstance(member, str)
                or not member
                or "\\" in member
                or PurePosixPath(member).is_absolute()
                or ".." in PurePosixPath(member).parts
                or not _FILENAME.fullmatch(PurePosixPath(member).name)
                or not isinstance(entry.get("sha256"), str)
                or not _SHA.fullmatch(entry["sha256"])
                or type(entry.get("bytes")) is not int
                or not 0 < entry["bytes"] <= 256 * 1024 * 1024
            ):
                raise ValueError("Invalid DICOM allowlist entry")
            self.dicom_routes[route] = {**entry, "archive": archive}
            self.allowed_zips.add(archive)
        super().__init__((host, port), ReviewHandler)

    def handle_error(self, request, client_address):
        # Never emit a traceback containing a capability URL or local file path.
        pass

    def process_request(self, request, client_address):
        # Bound accepted sockets before parsing headers or creating a thread.
        # A listener backlog does not limit simultaneous slow HTTP clients.
        if not self.http_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.http_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.http_slots.release()

    def server_close(self):
        if self.uploads is not None:
            self.uploads.close()
        super().server_close()


class ReviewHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Review"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(120)

    def log_message(self, format, *args):
        pass

    def _headers(self, code, length, mime="text/plain; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "none")
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _error(self, code):
        text = {
            400: "Bad request\n",
            404: "Not found\n",
            405: "Method not allowed\n",
            500: "Resource unavailable\n",
        }.get(code, "Request rejected\n")
        content = text.encode()
        self._headers(code, len(content), extra={"Allow": "GET, HEAD"} if code == 405 else None)
        if self.command != "HEAD":
            self.wfile.write(content)

    def send_error(self, code, message=None, explain=None):
        if code == 501 and hasattr(self, "path"):
            self._method_not_allowed()
        else:
            self._error(code if code in {400, 404, 405, 500} else 400)

    def _relative_path(self):
        try:
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or len(parsed.path) > 4096:
                raise ValueError
            decoded = unquote(parsed.path, encoding="utf-8", errors="strict")
            parts = decoded.split("/")
            if len(parts) < 3 or parts[:2] != ["", "r"]:
                raise ValueError
            candidate = parts[2].encode("utf-8")
            if not hmac.compare_digest(candidate, self.server.token_bytes):
                raise ValueError
            remainder = parts[3:]
            if (
                "\\" in decoded
                or "\x00" in decoded
                or any(p in {".", ".."} or p.startswith(".") for p in remainder if p)
            ):
                raise ValueError
            if any(not part for part in remainder[:-1]):
                raise ValueError
            return "/".join(remainder), decoded == self.server.prefix[:-1]
        except (ValueError, UnicodeError):
            self._error(404)
            return None

    def _method_not_allowed(self):
        if self._relative_path() is not None:
            self._error(405)

    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_OPTIONS = _method_not_allowed
    do_TRACE = _method_not_allowed
    do_CONNECT = _method_not_allowed

    def _json(self, code, value):
        content = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()
        self._headers(code, len(content), "application/json; charset=utf-8")
        if self.command != "HEAD":
            self.wfile.write(content)

    def do_POST(self):
        # Never leave an unread POST body on a persistent HTTP connection.
        self.close_connection = True
        parsed = self._relative_path()
        if parsed is None:
            return
        relative, _ = parsed
        if relative != "api/uploads" or self.server.uploads is None:
            self._error(405)
            return
        if (
            self.headers.get("Origin") != self.server.public_origin
            or self.headers.get("X-Upload-Intent") != "dicom-series"
            or self.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin"
        ):
            self._json(
                403,
                {
                    "error": "origin_rejected",
                    "message": "Загрузка разрешена только со страницы этого сайта.",
                },
            )
            return
        if (
            self.headers.get("Transfer-Encoding")
            or self.headers.get_content_type() != "application/zip"
        ):
            self._json(
                415,
                {"error": "unsupported_type", "message": "Загрузите ZIP-архив одной DICOM-серии."},
            )
            return
        if len(self.headers.get_all("Content-Length", [])) != 1:
            self._json(
                411, {"error": "length_required", "message": "Не удалось определить размер файла."}
            )
            return
        try:
            length = int(self.headers["Content-Length"])
            record = self.server.uploads.receive(self.rfile, length)
            self._json(202, record)
        except UploadRejected as error:
            self._json(error.status, {"error": error.code, "message": error.message})
        except (ValueError, TimeoutError, OSError):
            self._json(
                400, {"error": "upload_failed", "message": "Не удалось завершить загрузку файла."}
            )

    def do_GET(self):
        self._serve()

    def do_HEAD(self):
        self._serve()

    def _serve(self):
        try:
            parsed = self._relative_path()
            if parsed is None:
                return
            relative, redirect = parsed
            if redirect:
                self._headers(308, 0, extra={"Location": self.server.prefix})
                return
            if relative.startswith("api/"):
                self._api(relative)
                return
            if relative in self.server.dicom_routes:
                self._dicom(self.server.dicom_routes[relative])
                return
            if relative.startswith("dicom/"):
                self._error(404)
                return
            relative = relative or "index.html"
            path = (self.server.site_root / relative).resolve()
            suffix = path.suffix.lower()
            mime = MIMES.get(suffix)
            if (
                not path.is_relative_to(self.server.site_root)
                or not path.is_file()
                or not mime
                or (suffix == ".json" and not relative.startswith("data/"))
                or (suffix == ".zip" and path not in self.server.allowed_zips)
            ):
                self._error(404)
                return
            extra = {}
            if suffix == ".zip":
                extra["Content-Disposition"] = f'attachment; filename="{path.name}"'
            with path.open("rb") as stream:
                self._headers(200, os.fstat(stream.fileno()).st_size, mime, extra)
                if self.command != "HEAD":
                    shutil.copyfileobj(stream, self.wfile, 128 * 1024)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except Exception:
            self._error(500)

    def _api(self, relative):
        store = self.server.uploads
        if store is None:
            self._error(404)
            return
        if relative == "api/uploads":
            self._json(200, store.list())
            return
        pieces = relative.split("/", 3)
        if len(pieces) < 3 or pieces[:2] != ["api", "uploads"]:
            self._error(404)
            return
        identifier = pieces[2]
        if len(pieces) == 3:
            record = store.get(identifier)
            if record is None:
                self._error(404)
            else:
                self._json(200, record)
            return
        path = store.asset(identifier, pieces[3])
        if path is None:
            self._error(404)
            return
        mime = "application/dicom" if path.suffix == ".dcm" else MIMES.get(path.suffix)
        if mime is None:
            self._error(404)
            return
        extra = {}
        if path.suffix in {".zip", ".dcm", ".json"}:
            extra["Content-Disposition"] = f'attachment; filename="{path.name}"'
        with path.open("rb") as stream:
            self._headers(200, os.fstat(stream.fileno()).st_size, mime, extra)
            if self.command != "HEAD":
                shutil.copyfileobj(stream, self.wfile, 128 * 1024)

    def _dicom(self, entry):
        with zipfile.ZipFile(entry["archive"]) as zipped:
            member = zipped.getinfo(entry["member"])
            if member.is_dir() or member.file_size != entry["bytes"]:
                raise ValueError("DICOM member size mismatch")
            content = None if self.command == "HEAD" else zipped.read(member)
        if content is not None and (
            len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]
        ):
            raise ValueError("DICOM member integrity mismatch")
        self._headers(
            200,
            entry["bytes"],
            "application/dicom",
            {
                "Content-Disposition": f'attachment; filename="{PurePosixPath(entry["member"]).name}"',
                "ETag": '"' + entry["sha256"] + '"',
            },
        )
        if content is not None:
            self.wfile.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        root = Path(os.environ.get("REVIEW_ROOT", Path(__file__).resolve().parent))
        upload_root = os.environ.get("REVIEW_UPLOAD_ROOT")
        uploads = None
        if upload_root:
            heads = json.loads(os.environ.get("REVIEW_INFERENCE_HEADS", "[]"))
            if not isinstance(heads, list) or any(not isinstance(p, str) for p in heads):
                raise ValueError("Invalid head list")
            uploads = UploadStore(
                Path(upload_root),
                heads=heads,
                cache=Path(os.environ.get("REVIEW_MODEL_CACHE", ".cache")),
                device=os.environ.get("REVIEW_INFERENCE_DEVICE", "cpu"),
                threads=int(os.environ.get("REVIEW_INFERENCE_THREADS", "1")),
            )
        server = ReviewServer(
            root,
            os.environ.get("REVIEW_TOKEN", ""),
            args.port,
            uploads=uploads,
            public_origin=os.environ.get("REVIEW_ORIGIN", ""),
            host=args.host,
        )
    except Exception:
        print("Review server configuration is incomplete or invalid.", file=sys.stderr)
        return 1
    print(
        f"Private review server ready on {args.host}:{args.port}; capability URL omitted.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
