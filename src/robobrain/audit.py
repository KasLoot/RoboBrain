"""Provider-call audit trail with content-addressed image assets, without credentials."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from uuid import uuid4


class ModelAudit:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.secrets = tuple(value for name, value in os.environ.items()
                             if len(value) >= 8 and any(word in name.upper() for word in ('API_KEY', 'TOKEN', 'SECRET')))

    def clean(self, value, key=''):
        if re.search(r'(api[_-]?key|authorization|(?:access|control|view)[_-]?token|secret|password)', key, re.I):
            return '[redacted]'
        if isinstance(value, dict):
            return {str(k): self.clean(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            cleaned = [self.clean(v, 'image_base64' if key == 'images' else key) for v in value]
            return tuple(cleaned) if isinstance(value, tuple) else cleaned
        if isinstance(value, str):
            if value.startswith('data:image/') and ';base64,' in value:
                prefix, encoded = value.split(',', 1)
                return self._image(encoded, '.png' if 'png' in prefix else '.jpg')
            if key == 'image_base64':
                return self._image(value, '.jpg')
            for secret in self.secrets:
                value = value.replace(secret, '[redacted]')
            return value
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)

    def _image(self, encoded, suffix):
        if len(encoded) > 64 * 1024 * 1024:
            raise ValueError('Audit image exceeds 48 MiB decoded limit')
        data = base64.b64decode(encoded, validate=True)
        digest = hashlib.sha256(data).hexdigest()
        path = self.directory / 'assets' / (digest + suffix)
        path.parent.mkdir(exist_ok=True)
        try:
            with path.open('xb') as stream:
                stream.write(data)
        except FileExistsError:
            pass
        return {'asset': str(path.relative_to(self.directory)), 'sha256': digest, 'bytes': len(data)}

    def event(self, event):
        record = self.clean({'utc': datetime.now(timezone.utc).isoformat(), 'monotonic_ns': time.monotonic_ns(), **event})
        with self.lock:
            with (self.directory / 'calls.jsonl').open('a', encoding='utf-8') as stream:
                encoded = json.dumps(record, ensure_ascii=False)
                if len(encoded.encode('utf-8')) > 4 * 1024 * 1024:
                    asset = self.directory / 'large-events' / (uuid4().hex + '.json')
                    asset.parent.mkdir(exist_ok=True)
                    asset.write_text(encoded, encoding='utf-8')
                    encoded = json.dumps({'event': record.get('event'), 'call_id': record.get('call_id'),
                        'payload': str(asset.relative_to(self.directory)), 'utc': record['utc']})
                stream.write(encoded + '\n')

    def begin(self, payload, *, specialist, evidence, call_id=None):
        call_id = call_id or uuid4().hex
        self.event({'event': 'request_started', 'call_id': call_id, 'system': 'robobrain',
                    'role': specialist, 'model': payload.get('model'), 'evidence': evidence, 'request': payload})
        return call_id
