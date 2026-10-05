from __future__ import annotations

import hashlib
import html
import ipaddress
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from artifex.config.models import ResearchConfig

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
_SPACE = re.compile(r"\s+")


class ResearchSecurityError(ValueError):
    pass


def sanitize_text(value: object, *, max_chars: int) -> str:
    text = html.unescape(str(value or ""))
    text = _CONTROL.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    return text[:max_chars]


def normalize_query(value: str) -> str:
    return _SPACE.sub(" ", value.casefold()).strip()


def canonicalize_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ResearchSecurityError("research URLs must use http/https with a hostname")
    hostname = parsed.hostname.casefold()
    port = parsed.port
    netloc = hostname
    if port is not None and not (
        (parsed.scheme.casefold() == "http" and port == 80)
        or (parsed.scheme.casefold() == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.casefold(), netloc, path, query, ""))


def validate_external_url(value: str, config: ResearchConfig) -> str:
    canonical = canonicalize_url(value)
    parsed = urlsplit(canonical)
    host = parsed.hostname or ""
    host_lower = host.casefold()

    blocked = tuple(domain.casefold().lstrip(".") for domain in config.blocked_domains)
    if any(
        host_lower == domain or host_lower.endswith(f".{domain}")
        for domain in blocked
    ):
        raise ResearchSecurityError(f"research domain is blocked: {host}")

    allowed = tuple(domain.casefold().lstrip(".") for domain in config.allowed_domains)
    if allowed and not any(
        host_lower == domain or host_lower.endswith(f".{domain}")
        for domain in allowed
    ):
        raise ResearchSecurityError(f"research domain is not allowlisted: {host}")

    if not config.allow_private_network:
        if host_lower in {"localhost", "localhost.localdomain"} or host_lower.endswith(".local"):
            raise ResearchSecurityError("private/local research URLs are not allowed")
        try:
            address = ipaddress.ip_address(host_lower.strip("[]"))
        except ValueError:
            address = None
        if address is not None and (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise ResearchSecurityError("private/reserved research URLs are not allowed")

    return canonical


def content_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\x00")
    return digest.hexdigest()
