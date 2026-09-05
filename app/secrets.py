"""Keeping provider API keys out of the database in plain text.

The settings row is JSON, and until now the Gemini or OpenAI key sat in it as
written. Anyone with the database file - a backup, a copied volume, a support
dump - had the key. Encrypting it does not make the file safe to share; it
makes the key one secret further away, held in `APP_SECRET_KEY` rather than in
the same file as the data.

Three modes, in decreasing order of how much the deployment trusts the browser:

- `APP_ENV=production`: keys are read from the environment only. The API
  refuses to store one, so there is nothing in the database to steal.
- `APP_SECRET_KEY` set: a key saved from the interface is encrypted at rest with
  Fernet (AES-CBC with an HMAC), and decrypted when settings are loaded.
- neither: keys are stored as before, and the application says so once at start
  rather than pretending otherwise.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os

logger = logging.getLogger("adaptive_metric_rag.secrets")

PREFIX = "enc:v1:"


def production_mode() -> bool:
    return os.getenv("APP_ENV", "").strip().lower() in {"production", "prod"}


def _fernet():
    """The cipher for this deployment, or None when no secret is configured."""
    secret = os.getenv("APP_SECRET_KEY", "").strip()
    if not secret:
        return None
    try:
        from cryptography.fernet import Fernet
    except ImportError:  # pragma: no cover - only without the optional wheel
        logger.warning("APP_SECRET_KEY is set but `cryptography` is not installed; "
                       "API keys will be stored in clear text")
        return None
    # Fernet wants 32 url-safe base64 bytes; any passphrase is hashed into that
    # shape so an operator can set APP_SECRET_KEY to whatever their secret store
    # gives them.
    material = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(material)


def available() -> bool:
    """Whether saved keys can be encrypted at all."""
    return _fernet() is not None


def encrypt(value: str) -> str:
    """Encrypt a secret for storage; returns it unchanged when no key is set."""
    if not value or value.startswith(PREFIX):
        return value
    cipher = _fernet()
    if cipher is None:
        return value
    return PREFIX + cipher.encrypt(value.encode("utf-8")).decode("ascii")


def decrypt(value: str) -> str:
    """Read a stored secret, whether or not it was encrypted.

    A value that cannot be decrypted - the secret was rotated, or the database
    was moved to a deployment with a different one - returns empty rather than
    raising, so a lost key costs the provider connection and not the whole
    application.
    """
    if not value or not value.startswith(PREFIX):
        return value
    cipher = _fernet()
    if cipher is None:
        logger.warning("a stored API key is encrypted but APP_SECRET_KEY is not set")
        return ""
    from cryptography.fernet import InvalidToken

    try:
        return cipher.decrypt(value[len(PREFIX):].encode("ascii")).decode("utf-8")
    except InvalidToken:
        logger.warning("a stored API key could not be decrypted with the current APP_SECRET_KEY")
        return ""


def describe() -> dict[str, bool]:
    """What the interface needs to know about how keys are handled here."""
    return {"production": production_mode(), "encryption": available()}
