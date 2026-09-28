"""Test-session setup: the digest cache lives in a temporary directory, never
in the user's ``~/.cache``."""
import os
import shutil
import tempfile

_CACHE_DIR = tempfile.mkdtemp(prefix="mtp-digest-cache-")
os.environ["MTP_DIGEST_CACHE"] = os.path.join(_CACHE_DIR, "digests.json")


def pytest_unconfigure(config):
    shutil.rmtree(_CACHE_DIR, ignore_errors=True)


try:  # keep the suite on a quiet CPU (it is run under taskset/nice as well)
    import torch

    torch.set_num_threads(2)
except ImportError:
    pass
