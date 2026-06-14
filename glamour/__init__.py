"""Glamour — web-design critique arena + synthetic-data engine."""
from .critique import enrich_file, enrich_record, is_grounded
from .openrouter import (
    ChatResponse,
    OpenRouter,
    OpenRouterError,
    image_part,
    system_message,
    text_part,
    user_message,
)

__all__ = [
    "OpenRouter",
    "OpenRouterError",
    "ChatResponse",
    "user_message",
    "system_message",
    "text_part",
    "image_part",
    "enrich_record",
    "enrich_file",
    "is_grounded",
]
