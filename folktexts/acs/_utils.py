"""Helper functions for ACS data processing."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

CACHE_PUMS_CODES = {}


def parse_pums_code(
    value: int,
    file: str | Path,
    postprocess: Callable[[str], str] | None = None,
    # cache={},
) -> str:
    # Check if file already loaded into cache
    global CACHE_PUMS_CODES
    if file not in CACHE_PUMS_CODES:
        line_re = re.compile(r"(?P<code>\d+)\s+[.](?P<description>.+)$")

        file_cache = {}
        with open(file) as f:
            for line in f:
                m = line_re.match(line)
                if m is None:
                    logging.error(f"Could not parse line: {line}")
                    continue

                code, description = m.group("code"), m.group("description")
                file_cache[int(code)] = description  # raw; postprocess applied at lookup

        CACHE_PUMS_CODES[file] = file_cache

    # Get file caches
    file_cache = CACHE_PUMS_CODES[file]

    # Return the value from cache, or "N/A" if not found
    if value not in file_cache:
        logging.warning(f"Could not find code '{value}' in file '{file}'")
        return "N/A"

    raw = file_cache[value]
    return postprocess(raw) if postprocess else raw


def reset_cache():
    global CACHE_PUMS_CODES
    CACHE_PUMS_CODES.clear()  # Reset cache
