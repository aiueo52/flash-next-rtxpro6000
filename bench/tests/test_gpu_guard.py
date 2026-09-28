from __future__ import annotations

import subprocess
from collections.abc import Sequence

from fnbench.gpu_guard import check_gpu_guard


def training_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command,
        0,
        (
            "4123, /home/user/other-gpu-job/"
            "venv/bin/python, 66966 MiB\n"
        ),
        "",
    )


def test_unknown_large_gpu_process_is_rejected() -> None:
    result = check_gpu_guard(runner=training_runner)
    assert not result.allowed
    assert result.blocked[0].pid == 4123
    assert result.blocked[0].used_memory_mib == 66966


def test_force_allows_unknown_large_gpu_process() -> None:
    result = check_gpu_guard(force=True, runner=training_runner)
    assert result.allowed
    assert result.forced


def test_realistic_sglang_venv_process_is_allowed() -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            (
                "55, /home/user/tools/sglang-rtxpro6000/"
                ".venv/bin/python3.12, 90000 MiB\n"
            ),
            "",
        )

    assert check_gpu_guard(runner=runner).allowed


def test_custom_allowed_process_substring() -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            "88, /srv/custom-engine/bin/python, 8192 MiB\n",
            "",
        )

    assert not check_gpu_guard(runner=runner).allowed
    assert check_gpu_guard(runner=runner, allow_proc=("CUSTOM-engine",)).allowed
