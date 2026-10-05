from __future__ import annotations

from artifex.config.models import DiscordConfig
from artifex.discord.models import NotificationKind


class NotificationPolicy:
    def __init__(self, config: DiscordConfig) -> None:
        self._config = config

    def enabled(self, kind: NotificationKind) -> bool:
        match kind:
            case NotificationKind.COMPLETION:
                return self._config.notify_completion
            case NotificationKind.REVIEW:
                return self._config.notify_review
            case NotificationKind.ERROR:
                return self._config.notify_error
            case NotificationKind.BACKEND:
                return self._config.notify_backend
            case NotificationKind.DISK:
                return self._config.notify_disk
            case NotificationKind.DAILY_SUMMARY:
                return True
