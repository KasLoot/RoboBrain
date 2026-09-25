"""Per-run audit files with bounded, credential-free JSONL and image references.

This module has no simulator dependency. Run directories are chosen by the caller;
all asset paths are constrained to that directory. The caller must keep disk writes
outside the physics owner and pass actual configured secrets for exact redaction.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import inspect
from io import BytesIO
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

_SENSITIVE = {"api_key", "apikey", "authorization", "proxy_authorization", "password", "passwd",
              "secret", "client_secret", "access_token", "refresh_token", "auth_token", "control_token", "view_token"}
_INLINE = re.compile(r"data:([\w.+-]+/[\w.+-]+);base64,([A-Za-z0-9+/=\r\n]+)")
_CREDENTIAL = re.compile(r"(?i)((?:api[_-]?key|authorization|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*)(?:Bearer\s+)?[^\s,;&\"'}]+")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_API_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_URL = re.compile(r"https?://[^\s<>\"']+")


def is_cancellation(exc: BaseException) -> bool:
    return isinstance(exc, (KeyboardInterrupt, SystemExit)) or "cancel" in type(exc).__name__.lower()


class RunArtifacts:
    def __init__(self, run_dir, *, secrets=(), metadata=None, max_record_bytes=4 * 1024 * 1024,
                 max_asset_bytes=64 * 1024 * 1024):
        self.run_dir = Path(run_dir).resolve()
        self.directory = self.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._secrets = {str(value) for value in secrets if value and str(value) not in ("EMPTY", "None")}
        self.max_record_bytes = int(max_record_bytes)
        self.max_asset_bytes = int(max_asset_bytes)
        self._seq = 0
        if metadata is not None:
            self.write_json("metadata.json", metadata)

    def add_secrets(self, *secrets):
        with self._lock:
            self._secrets.update(str(value) for value in secrets if value and str(value) not in ("EMPTY", "None"))

    def _path(self, relative):
        path = (self.run_dir / str(relative)).resolve()
        if path == self.run_dir or self.run_dir not in path.parents:
            raise ValueError("Audit asset path must remain inside the run directory")
        return path

    def _text(self, value):
        for secret in sorted(self._secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        value = _BEARER.sub("Bearer [REDACTED]", value)
        value = _CREDENTIAL.sub(lambda m: m.group(1) + "[REDACTED]", value)
        value = _API_KEY.sub("[REDACTED]", value)
        def clean_url(match):
            try:
                url = urlsplit(match.group())
                host = url.hostname or ""
                if ":" in host:
                    host = "[" + host + "]"
                if url.port:
                    host += ":" + str(url.port)
                query = [(k, "[REDACTED]" if k.lower().replace("-", "_") in _SENSITIVE | {"key", "token"} else v)
                         for k, v in parse_qsl(url.query, keep_blank_values=True)]
                return urlunsplit((url.scheme, host, url.path, urlencode(query), url.fragment))
            except ValueError:
                return "[REDACTED_URL]"
        return _URL.sub(clean_url, value)

    def _clean(self, value, *, inline_assets=False):
        if is_dataclass(value) and not isinstance(value, type):
            value = asdict(value)
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                label = str(key)
                normalized = label.lower().replace("-", "_")
                if normalized in _SENSITIVE or normalized.endswith("_api_key"):
                    result[label] = "[REDACTED]"
                elif normalized in ("base64", "b64_json", "image_base64"):
                    result[label] = "[INLINE_BINARY_REDACTED]"
                else:
                    result[label] = self._clean(item, inline_assets=inline_assets)
            return result
        if isinstance(value, (list, tuple)):
            return [self._clean(item, inline_assets=inline_assets) for item in value]
        if isinstance(value, (bytes, bytearray, memoryview)):
            data = bytes(value)
            return {"binary_bytes": len(data), "sha256": sha256(data).hexdigest()}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, str):
            match = _INLINE.fullmatch(value)
            if match and inline_assets and match.group(1).startswith("image/"):
                encoded = match.group(2)
                if len(encoded) > self.max_asset_bytes * 4 // 3 + 8:
                    return {"asset_error": "inline image exceeds maximum size"}
                try:
                    data = base64.b64decode(encoded, validate=True)
                    extension = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(match.group(1), ".bin")
                    path = "assets/" + sha256(data).hexdigest() + extension
                    return self.save_bytes(path, data) | {"mime_type": match.group(1)}
                except ValueError:
                    return {"asset_error": "invalid inline image"}
            return self._text(_INLINE.sub("[INLINE_BINARY_REDACTED]", value))
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            import math
            return value if math.isfinite(value) else str(value)
        if hasattr(value, "tolist"):
            return self._clean(value.tolist(), inline_assets=inline_assets)
        return self._text(str(value))

    def redact(self, value):
        with self._lock:
            return self._clean(value)

    def save_bytes(self, relative_path, data):
        data = bytes(data)
        if len(data) > self.max_asset_bytes:
            raise ValueError("Audit asset exceeds maximum size")
        with self._lock:
            path = self._path(relative_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)
            return {"asset": path.relative_to(self.run_dir).as_posix(), "bytes": len(data), "sha256": sha256(data).hexdigest()}

    def save_image(self, relative_path, image):
        from PIL import Image
        import numpy as np
        rgb = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
        buffer = BytesIO()
        rgb.save(buffer, format="PNG")
        return self.save_bytes(relative_path, buffer.getvalue()) | {"mime_type": "image/png", "width": rgb.width, "height": rgb.height}

    def write_json(self, relative_path, value):
        with self._lock:
            encoded = json.dumps(self._clean(value, inline_assets=True), ensure_ascii=False, allow_nan=False, indent=2).encode()
            if len(encoded) > self.max_record_bytes:
                raise ValueError("Audit JSON exceeds maximum record size")
            return self.save_bytes(relative_path, encoded)

    def append(self, stream, event):
        """Append one complete record; independently readable after interruption."""
        with self._lock:
            self._seq += 1
            record = {"audit_seq": self._seq, "server_monotonic_ns": time.monotonic_ns(), **event}
            record = self._clean(record, inline_assets=True)
            encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
            if len(encoded) > self.max_record_bytes:
                # Keep start/end/identity visible even when a provider emits an oversized payload.
                asset = self.save_bytes("oversized/" + uuid4().hex + ".json", encoded)
                record = {"audit_seq": self._seq, "server_monotonic_ns": record["server_monotonic_ns"],
                          "event": record.get("event"), "call_id": record.get("call_id"), "payload": asset}
                encoded = json.dumps(record, separators=(",", ":")).encode()
            path = self._path(str(stream) if str(stream).endswith(".jsonl") else str(stream) + ".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(encoded + b"\n")
            return record


class AuditedClient:
    """Audit both public model calls and their raw completion transports.

    ``check`` validates the execution generation before/after a provider call. A
    cancelled late response is retained for audit but can never reach the runner.
    The client is owned exclusively by this wrapper; close() restores its transport.
    """
    def __init__(self, client, artifacts: RunArtifacts, *, context=None, check=None, role="vlm"):
        self._client, self._artifacts = client, artifacts
        self._context = context or (lambda: {})
        self._check = check
        self._role = role
        self._local = threading.local()
        artifacts.add_secrets(getattr(client, "api_key", None), getattr(client, "_api_key", None))
        self._transport = getattr(client, "_post_chat", None)
        if self._transport is not None:
            client._post_chat = self._post_chat

    def close(self):
        if self._transport is not None:
            self._client._post_chat = self._transport

    def __getattr__(self, name):
        value = getattr(self._client, name)
        if not callable(value) or not name.startswith("complete"):
            return value
        def wrapped(*args, **kwargs):
            call_id = uuid4().hex
            context = dict(self._context() or {})
            context.setdefault("role", self._role)
            metadata = {**context, "call_id": call_id, "method": name, "model": getattr(self._client, "model", None)}
            try:
                parameters = dict(inspect.signature(value).bind(*args, **kwargs).arguments)
            except (TypeError, ValueError):
                parameters = {"prompt": args[0] if args else kwargs.get("prompt", "")}
            image_refs = {}
            for key in list(parameters):
                if key.endswith("image") or key.endswith("images"):
                    item = parameters.pop(key)
                    if item is not None:
                        sequence = item if isinstance(item, (list, tuple)) else [item]
                        image_refs[key] = [self._artifacts.save_image(f"vlm/{call_id}/{key}_{i}.png", image)
                                           for i, image in enumerate(sequence)]
            self._artifacts.append("vlm", {"event": "vlm_call_start", **metadata, "parameters": parameters, "images": image_refs})
            previous = getattr(self._local, "call", None)
            self._local.call = metadata
            started = time.monotonic()
            try:
                if self._check:
                    self._check()
                result = value(*args, **kwargs)
                if self._check:
                    self._check()
                self._artifacts.append("vlm", {"event": "vlm_call_end", **metadata, "elapsed_s": time.monotonic()-started,
                                                "raw_text": getattr(result, "raw_text", None), "token": getattr(result, "token", None),
                                                "response": getattr(result, "payload", result)})
                return result
            except BaseException as exc:
                self._artifacts.append("vlm", {"event": "vlm_call_cancel" if is_cancellation(exc) else "vlm_call_error",
                                                **metadata, "elapsed_s": time.monotonic()-started,
                                                "error_type": type(exc).__name__, "error": str(exc)})
                raise
            finally:
                self._local.call = previous
        return wrapped

    def _post_chat(self, payload):
        self._artifacts.add_secrets(getattr(self._client, "api_key", None), getattr(self._client, "_api_key", None))
        metadata = dict(getattr(self._local, "call", None) or {**self._context(), "role": self._role, "call_id": uuid4().hex})
        metadata.update(transport_id=uuid4().hex, model=payload.get("model", getattr(self._client, "model", None)))
        self._artifacts.append("vlm", {"event": "vlm_transport_start", **metadata, "request": payload})
        started = time.monotonic()
        try:
            if self._check:
                self._check()
            response = self._transport(payload)
            self._artifacts.add_secrets(getattr(self._client, "api_key", None), getattr(self._client, "_api_key", None))
            # Persist the raw response even when stop superseded it during the request.
            self._artifacts.append("vlm", {"event": "vlm_transport_end", **metadata,
                                            "elapsed_s": time.monotonic()-started, "response": response})
            if self._check:
                self._check()
            return response
        except BaseException as exc:
            self._artifacts.add_secrets(getattr(self._client, "api_key", None), getattr(self._client, "_api_key", None))
            self._artifacts.append("vlm", {"event": "vlm_transport_cancel" if is_cancellation(exc) else "vlm_transport_error",
                                            **metadata, "elapsed_s": time.monotonic()-started,
                                            "error_type": type(exc).__name__, "error": str(exc)})
            raise
