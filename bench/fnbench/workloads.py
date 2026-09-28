from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import Sampling, Workload

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "workloads" / "manifest.json"


class WorkloadError(ValueError):
    """Raised when a workload manifest is invalid."""


def load_workloads(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Workload]:
    try:
        raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadError(f"cannot load workload manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("workloads"), list):
        raise WorkloadError("manifest must contain a 'workloads' list")

    result: dict[str, Workload] = {}
    root = manifest_path.parent
    for item in raw["workloads"]:
        if not isinstance(item, dict):
            raise WorkloadError("each workload must be an object")
        try:
            name = str(item["name"])
            prompt_path = root / str(item["prompt_file"])
            max_tokens = int(item["max_tokens"])
            description = str(item["description"])
            sampling_raw = item["sampling"]
            temperature = float(sampling_raw["temperature"])
            top_p_raw = sampling_raw.get("top_p")
            top_p = float(top_p_raw) if top_p_raw is not None else None
            repeat_prompt = int(item.get("repeat_prompt", 1))
            postamble = str(item.get("postamble", ""))
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkloadError(f"invalid workload entry: {item!r}") from exc
        if name in result:
            raise WorkloadError(f"duplicate workload name: {name}")
        if max_tokens <= 0 or repeat_prompt <= 0:
            raise WorkloadError(f"{name}: max_tokens and repeat_prompt must be positive")
        if not prompt_path.is_file():
            raise WorkloadError(f"{name}: prompt file not found: {prompt_path}")
        result[name] = Workload(
            name=name,
            prompt_path=prompt_path,
            max_tokens=max_tokens,
            sampling=Sampling(temperature=temperature, top_p=top_p),
            description=description,
            repeat_prompt=repeat_prompt,
            postamble=postamble,
        )
    return result


def select_workloads(names: str, manifest_path: Path = DEFAULT_MANIFEST) -> list[Workload]:
    available = load_workloads(manifest_path)
    requested = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in requested if name not in available]
    if unknown:
        choices = ", ".join(sorted(available))
        raise WorkloadError(f"unknown workload(s): {', '.join(unknown)}; choices: {choices}")
    if not requested:
        raise WorkloadError("at least one workload is required")
    return [available[name] for name in requested]


def select_prompt_sets(names: str, root: Path, version: str = "v1", limit: int | None = None) -> list[Workload]:
    """Expand domain manifests in stable round-robin order (same in every arm)."""
    domains = select_workloads(names)
    if limit is not None and limit <= 0:
        raise WorkloadError("prompt limit must be positive")
    groups = []
    for domain in domains:
        path = root / f"{domain.name}-{version}" / "manifest.json"
        try:
            raw = json.loads(path.read_text())
            items = raw["prompts"]
            if raw["domain"] != domain.name or raw["version"] != version or not items:
                raise ValueError("domain/version mismatch or empty set")
            if limit is not None and len(items) < limit:
                raise ValueError("fewer prompts than requested")
            group, seen = [], set()
            for item in items[:limit]:
                prompt = (path.parent / item["file"]).resolve()
                if not prompt.is_relative_to(path.parent.resolve()):
                    raise ValueError("prompt path escapes set")
                text = prompt.read_text().strip()
                if hashlib.sha256(text.encode()).hexdigest() != item["sha256"]:
                    raise ValueError("prompt hash mismatch")
                if item["id"] in seen or not text or int(item["max_tokens"]) <= 0:
                    raise ValueError("duplicate id, empty prompt or invalid budget")
                seen.add(item["id"])
                group.append(Workload(domain.name, prompt, int(item["max_tokens"]),
                                      domain.sampling, domain.description,
                                      prompt_id=item["id"], prompt_set=f"{domain.name}-{version}",
                                      input_tokens=int(item["input_tokens"])))
            groups.append(group)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise WorkloadError(f"invalid prompt set {path}: {exc}") from exc
    return [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]
