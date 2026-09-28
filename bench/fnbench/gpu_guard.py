from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

NVIDIA_SMI_COMMAND = (
    "nvidia-smi",
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader",
)
DEFAULT_ALLOWED_PROCESS_SUBSTRINGS = (
    "sglang-rtxpro6000",
    "llama-server",
)
VRAM_LIMIT_MIB = 2 * 1024


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    process_name: str
    used_memory_mib: int


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    warning: str | None
    blocked: tuple[GpuProcess, ...]
    forced: bool = False


type Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def parse_compute_apps(text: str) -> list[GpuProcess]:
    processes: list[GpuProcess] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.rsplit(",", maxsplit=2)]
        if len(parts) != 3:
            continue
        pid_text, process_name, memory_text = parts
        try:
            pid = int(pid_text)
            memory = int(memory_text.removesuffix("MiB").strip())
        except ValueError:
            continue
        processes.append(GpuProcess(pid, process_name, memory))
    return processes


def is_inference_process(
    process_name: str,
    allowed_substrings: Sequence[str] = DEFAULT_ALLOWED_PROCESS_SUBSTRINGS,
) -> bool:
    normalized_name = process_name.casefold()
    return any(
        substring and substring.casefold() in normalized_name
        for substring in allowed_substrings
    )


def check_gpu_guard(
    *,
    force: bool = False,
    allow_proc: Sequence[str] = (),
    runner: Runner = default_runner,
) -> GuardResult:
    try:
        completed = runner(NVIDIA_SMI_COMMAND)
    except FileNotFoundError:
        return GuardResult(
            allowed=True,
            warning="nvidia-smi not found; GPU process guard was skipped",
            blocked=(),
        )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        return GuardResult(
            allowed=True,
            warning=f"nvidia-smi unavailable ({detail}); GPU process guard was skipped",
            blocked=(),
        )
    allowed_substrings = (*DEFAULT_ALLOWED_PROCESS_SUBSTRINGS, *allow_proc)
    blocked = tuple(
        process
        for process in parse_compute_apps(completed.stdout)
        if process.used_memory_mib > VRAM_LIMIT_MIB
        and not is_inference_process(process.process_name, allowed_substrings)
    )
    return GuardResult(
        allowed=not blocked or force,
        warning=None,
        blocked=blocked,
        forced=bool(blocked and force),
    )


def format_block_reason(result: GuardResult) -> str:
    details = ", ".join(
        f"PID {item.pid} {item.process_name!r} ({item.used_memory_mib} MiB)"
        for item in result.blocked
    )
    return (
        "GPU guard refused the benchmark: unknown process(es) above 2 GiB: "
        f"{details}. Stop them or rerun with --force."
    )
