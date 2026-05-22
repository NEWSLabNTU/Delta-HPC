"""
src/deploy/metrics.py

vLLM Prometheus metrics scraping for the deploy package.

Provides lightweight helpers to fetch and parse the Prometheus text exposition
format exposed by ``GET /metrics`` on each vLLM server.

Public API
----------
* :class:`VLLMMetricsClient` — client to fetch and parse metrics for a specific port.
"""

from __future__ import annotations

import re
from typing import Dict

import requests

# ---------------------------------------------------------------------------
# Prometheus metric patterns
# ---------------------------------------------------------------------------

_TTFT_SUM = re.compile(
    r"^vllm:(?:request_)?time_to_first_token_seconds_sum\{[^}]*\}\s+([\d.eE+\-]+)", re.M
)
_TTFT_COUNT = re.compile(
    r"^vllm:(?:request_)?time_to_first_token_seconds_count\{[^}]*\}\s+([\d.eE+\-]+)",
    re.M,
)
_TPOT_SUM = re.compile(
    r"^vllm:(?:request_)?time_per_output_token_seconds_sum\{[^}]*\}\s+([\d.eE+\-]+)",
    re.M,
)
_TPOT_COUNT = re.compile(
    r"^vllm:(?:request_)?time_per_output_token_seconds_count\{[^}]*\}\s+([\d.eE+\-]+)",
    re.M,
)
_WAITING = re.compile(r"^vllm:num_requests_waiting\{[^}]*\}\s+([\d.eE+\-]+)", re.M)
_RUNNING = re.compile(r"^vllm:num_requests_running\{[^}]*\}\s+([\d.eE+\-]+)", re.M)
_KV_UTIL = re.compile(
    r"^vllm:(?:gpu|kv)_cache_usage_perc\{[^}]*\}\s+([\d.eE+\-]+)", re.M
)


def _first(pattern: re.Pattern[str], text: str, default: float = 0.0) -> float:
    """Return the first numeric capture group matched by *pattern*, or *default*."""
    m = pattern.search(text)
    return float(m.group(1)) if m else default


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


class VLLMMetricsClient:
    """Client for scraping Prometheus metrics from a vLLM server."""

    def __init__(self, port: int, timeout: float = 5.0) -> None:
        """
        Parameters
        ----------
        port:
            TCP port the vLLM server is listening on.
        timeout:
            HTTP request timeout in seconds.
        """
        self.port = port
        self.timeout = timeout

    def fetch_raw(self) -> str:
        """Fetch the raw Prometheus exposition text from ``http://localhost:<port>/metrics``.

        Returns
        -------
        str
            Raw Prometheus text body.

        Raises
        ------
        requests.HTTPError
            If the server returns a non-2xx status.
        requests.exceptions.RequestException
            On any connection / timeout error.
        """
        r = requests.get(f"http://localhost:{self.port}/metrics", timeout=self.timeout)
        r.raise_for_status()
        return r.text

    def running_requests(self) -> float:
        """Return the number of requests currently being processed."""
        return _first(_RUNNING, self.fetch_raw())

    def waiting_requests(self) -> float:
        """Return the number of requests currently waiting in the queue."""
        return _first(_WAITING, self.fetch_raw())

    def collect(self) -> Dict[str, float]:
        """Return a summary metrics dict for the vLLM server.

        Fetches ``GET /metrics`` once and parses:

        * ``ttft_mean_s`` — mean time-to-first-token in seconds (``0`` if no
          requests have been served yet).
        * ``queue_length`` — number of requests currently waiting.
        * ``running_requests`` — number of requests currently being processed.

        Returns
        -------
        dict[str, float]
            Keys: ``"ttft_mean_s"``, ``"queue_length"``, ``"running_requests"``.
        """
        text = self.fetch_raw()

        ttft_sum = _first(_TTFT_SUM, text)
        ttft_count = _first(_TTFT_COUNT, text)
        ttft_mean = (ttft_sum / ttft_count) if ttft_count > 0 else 0.0

        tpot_sum = _first(_TPOT_SUM, text)
        tpot_count = _first(_TPOT_COUNT, text)
        tpot_mean = (tpot_sum / tpot_count) if tpot_count > 0 else 0.0

        return {
            "ttft_mean_s": ttft_mean,
            "tpot_mean_s": tpot_mean,
            "queue_length": _first(_WAITING, text),
            "running_requests": _first(_RUNNING, text),
            "kv_cache_util": _first(_KV_UTIL, text),
        }
