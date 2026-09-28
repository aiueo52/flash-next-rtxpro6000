"""Command-line parsing shared by every tool: an explicit empty value is an error.

``--buckets`` with no value, ``--eval-set ""`` or ``--exclude-buckets a,,b``
used to read as "not given" (and so as "all", "none" or a default).  With
``StrictParser`` an option given without a value, or with an empty or blank
one, stops the tool; options that are not given keep their defaults.  Comma
lists go through ``split_list``, which refuses empty items.
"""
from __future__ import annotations

import argparse
from typing import List


class StrictParser(argparse.ArgumentParser):
    """``argparse.ArgumentParser`` that refuses explicit empty values."""

    def _get_values(self, action, arg_strings):
        if action.nargs != 0 and action.option_strings is not None:
            name = "/".join(action.option_strings) or action.dest
            if not arg_strings and action.nargs in ("*", "?"):
                self.error(f"{name} was given without a value; leave it out to use "
                           "its default")
            if any(not s.strip() for s in arg_strings):
                self.error(f"{name} was given an empty value")
        return super()._get_values(action, arg_strings)


def split_list(value: str, opt: str) -> List[str]:
    """``"a,b"`` -> ["a", "b"]; "" -> [] (the option's "not given" default);
    an empty item (``a,,b``, ``a,``) is an error."""
    if value == "":
        return []
    items = [x.strip() for x in value.split(",")]
    if any(not x for x in items):
        raise SystemExit(f"{opt} {value!r}: empty item in the comma-separated list")
    return items


__all__ = ["StrictParser", "split_list"]
