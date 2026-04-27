from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import os
import threading
import time
from typing import Callable, TypeVar

from quotemux.infra.provider_runtime.core import call_provider_api


T = TypeVar("T")
DEFAULT_MAX_CALLS_PER_MINUTE = 700
RATE_LIMIT_PERIOD_SECONDS = 60.0


def _int_env(name: str, default_value: int) -> int:
    text = os.getenv(name, "")
    if text == "":
        return default_value
    try:
        return int(text)
    except ValueError:
        return default_value


@dataclass(frozen=True)
class RateLimitStats:
    total_calls: int
    throttle_count: int
    total_wait_seconds: float


class TushareRateLimiter:
    def __init__(self, max_calls_per_minute: int) -> None:
        self._max_calls_per_minute = max_calls_per_minute
        self._lock = threading.Lock()
        self._call_times: deque[float] = deque()
        self._total_calls = 0
        self._throttle_count = 0
        self._total_wait_seconds = 0.0

    def call(self, func: Callable[..., T], *args: object, **kwargs: object) -> T:
        self._wait_for_slot()
        return func(*args, **kwargs)

    def stats(self) -> RateLimitStats:
        with self._lock:
            return RateLimitStats(
                total_calls=self._total_calls,
                throttle_count=self._throttle_count,
                total_wait_seconds=self._total_wait_seconds,
            )

    def _wait_for_slot(self) -> None:
        if self._max_calls_per_minute <= 0:
            return
        while True:
            wait_seconds = 0.0
            with self._lock:
                now = time.monotonic()
                self._clean_old_calls(now)
                if len(self._call_times) < self._max_calls_per_minute:
                    self._call_times.append(now)
                    self._total_calls += 1
                    return
                wait_seconds = self._calculate_wait_seconds(now)
                self._throttle_count += 1
                self._total_wait_seconds += wait_seconds
            if wait_seconds > 0:
                time.sleep(wait_seconds)

    def _clean_old_calls(self, now: float) -> None:
        cutoff = now - RATE_LIMIT_PERIOD_SECONDS
        while self._call_times and self._call_times[0] <= cutoff:
            self._call_times.popleft()

    def _calculate_wait_seconds(self, now: float) -> float:
        oldest_call = self._call_times[0]
        wait_seconds = oldest_call + RATE_LIMIT_PERIOD_SECONDS - now
        return max(0.0, wait_seconds)


@lru_cache(maxsize=1)
def get_tushare_rate_limiter() -> TushareRateLimiter:
    max_calls = _int_env("MHK_TUSHARE_MAX_CALLS_PER_MINUTE", DEFAULT_MAX_CALLS_PER_MINUTE)
    return TushareRateLimiter(max_calls)


def get_tushare_rate_limit_stats() -> RateLimitStats:
    return get_tushare_rate_limiter().stats()


def call_tushare_api(api_name: str, func: Callable[..., T], *args: object, **kwargs: object) -> T:
    limiter = get_tushare_rate_limiter()
    return call_provider_api("tushare", api_name, limiter.call, func, *args, **kwargs)
