"""The OpenRouter client now lives in anatta.backends.openrouter (clean-lab step 3); scout and the lab's other callers
keep importing it from here. Claude (Opus 5.5) wrote this file, 3 Nov 2026.
"""

from anatta.backends.openrouter import (  # noqa: F401
    ENDPOINTS_URL,
    KEY_ENV,
    MODELS_URL,
    OPENROUTER,
    REFERER,
    RETRY_CODES,
    TITLE,
    ApiError,
    OpenRouterClient,
    _allowed,
    _content_text,
    _get_json,
    _slug,
    model_meta,
    parse_completion,
)
