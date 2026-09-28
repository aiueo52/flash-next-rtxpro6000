from __future__ import annotations

from pathlib import Path


class LocalTokenizer:
    def __init__(self, path: str | Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "The CPU-only 'tokenizers' package is required. Install requirements.txt without using transformers or torch."
            ) from exc
        source = Path(path).expanduser()
        tokenizer_file = source / "tokenizer.json" if source.is_dir() else source
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_file}")
        self.path = tokenizer_file
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids
