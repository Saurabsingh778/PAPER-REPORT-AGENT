"""Secret scrubbing. Every line written by the logger passes through a Redactor."""
from __future__ import annotations

import os
import re

SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
MIN_SECRET_LEN = 8
REDACTED = "[REDACTED]"

_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),          # Google API keys
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),       # GitHub tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),     # GitHub fine-grained tokens
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),           # generic sk- style keys
]
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}")


class Redactor:
    """Replaces known secrets with [REDACTED].

    Secrets come from two places: the values of environment variables whose names look
    secret-ish (API_KEY, GITHUB_TOKEN, KAGGLE_KEY...) captured when the Redactor is
    created, plus well-known token shapes matched by regex.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._secrets: list[str] = []
        if enabled:
            for name, value in os.environ.items():
                if any(h in name.upper() for h in SECRET_ENV_HINTS):
                    self.add_secret(value)

    def add_secret(self, value: str | None) -> None:
        if value and len(value) >= MIN_SECRET_LEN and value not in self._secrets:
            self._secrets.append(value)
            self._secrets.sort(key=len, reverse=True)  # longest first

    def __call__(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, REDACTED)
        for pat in _PATTERNS:
            text = pat.sub(REDACTED, text)
        return _BEARER.sub(lambda m: m.group(1) + REDACTED, text)
