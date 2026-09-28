"""Standalone trainer for the Qwen3.8-Flash-Next MTP (NEXTN) draft head."""

from .config import MTPConfig
from .model import MTPHead

__all__ = ["MTPConfig", "MTPHead"]
