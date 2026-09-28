"""Corpus readers: turn session logs and text files into chat-message documents.

See ``corpus/formats.py``.  Every input location is an explicit argument; no
reader looks anywhere by default.
"""

from .formats import (  # noqa: F401
    iter_arrow_docs,
    iter_file_docs,
    load_claude_sessions,
    load_codex_sessions,
    load_lmstudio_conversations,
    mk,
    normalize,
    parse_claude_session,
)
