from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable, TypeVar

from quotemux.infra.config import DATALAKE_ROOT
from quotemux.infra.provider_runtime.core import call_provider_api


T = TypeVar("T")
DEFAULT_MAX_CALLS_PER_MINUTE = 700
DEFAULT_API_RATE_LIMITS = {
    "hm_detail": (1, 3600.0),
}
RATE_LIMIT_PERIOD_SECONDS = 60.0
RATE_LIMIT_STATE_PATH = Path(os.getenv("MHK_TUSHARE_RATE_LIMIT_STATE_PATH", "")) if os.getenv("MHK_TUSHARE_RATE_LIMIT_STATE_PATH", "") else DATALAKE_ROOT / "type=cache" / "service=integration_api" / "provider=tushare" / "rate_limit" / "state.json"


def _int_env(name: str, default_value: int) -> int:
    text = os.getenv(name, "")
    if text == "":
        return default_value
    try:
        return int(text)
    except ValueError:
        return default_value


def _float_env(name: str, default_value: float) -> float:
    text = os.getenv(name, "")
    if text == "":
        return default_value
    try:
        return float(text)
    except ValueError:
        return default_value


@dataclass(frozen=True)
class RateLimitStats:
    total_calls: int
    throttle_count: int
    total_wait_seconds: float


class TushareRateLimiter:
    def __init__(self, max_calls_per_minute: int, period_seconds: float = RATE_LIMIT_PERIOD_SECONDS, state_key: str = "") -> None:
        self._max_calls_per_minute = max_calls_per_minute
        self._period_seconds = period_seconds
        self._state_key = state_key
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
                self._load_state()
                now = time.time()
                self._clean_old_calls(now)
                if len(self._call_times) < self._max_calls_per_minute:
                    self._call_times.append(now)
                    self._save_state()
                    self._total_calls += 1
                    return
                wait_seconds = self._calculate_wait_seconds(now)
                self._throttle_count += 1
                self._total_wait_seconds += wait_seconds
            if wait_seconds > 0:
                time.sleep(wait_seconds)

    def _clean_old_calls(self, now: float) -> None:
        cutoff = now - self._period_seconds
        while self._call_times and self._call_times[0] <= cutoff:
            self._call_times.popleft()

    def _calculate_wait_seconds(self, now: float) -> float:
        oldest_call = self._call_times[0]
        wait_seconds = oldest_call + self._period_seconds - now
        return max(0.0, wait_seconds)

    def _load_state(self) -> None:
        if self._state_key == "" or not RATE_LIMIT_STATE_PATH.exists():
            return
        try:
            payload = json.loads(RATE_LIMIT_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        values = payload.get(self._state_key, [])
        if not isinstance(values, list):
            return
        call_times = []
        for value in values:
            try:
                call_times.append(float(value))
            except (TypeError, ValueError):
                continue
        self._call_times = deque(sorted(call_times))

    def _save_state(self) -> None:
        if self._state_key == "":
            return
        payload: dict[str, object] = {}
        if RATE_LIMIT_STATE_PATH.exists():
            try:
                raw_payload = json.loads(RATE_LIMIT_STATE_PATH.read_text(encoding="utf-8"))
                if isinstance(raw_payload, dict):
                    payload = raw_payload
            except Exception:
                payload = {}
        payload[self._state_key] = list(self._call_times)
        RATE_LIMIT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = RATE_LIMIT_STATE_PATH.with_name(f"{RATE_LIMIT_STATE_PATH.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temp_path, RATE_LIMIT_STATE_PATH)


@lru_cache(maxsize=1)
def get_tushare_rate_limiter() -> TushareRateLimiter:
    max_calls = _int_env("MHK_TUSHARE_MAX_CALLS_PER_MINUTE", DEFAULT_MAX_CALLS_PER_MINUTE)
    return TushareRateLimiter(max_calls, RATE_LIMIT_PERIOD_SECONDS, "global")


def _api_rate_env_name(api_name: str) -> str:
    normalized = "".join(char if char.isalnum() else "_" for char in api_name.upper())
    return f"MHK_TUSHARE_{normalized}_MAX_CALLS_PER_MINUTE"


@lru_cache(maxsize=None)
def get_tushare_api_rate_limiter(api_name: str) -> TushareRateLimiter:
    default_calls, default_period = DEFAULT_API_RATE_LIMITS.get(api_name, (DEFAULT_MAX_CALLS_PER_MINUTE, RATE_LIMIT_PERIOD_SECONDS))
    max_calls = _int_env(_api_rate_env_name(api_name), default_calls)
    period_seconds = _float_env(f"{_api_rate_env_name(api_name).replace('MAX_CALLS_PER_MINUTE', 'RATE_PERIOD_SECONDS')}", default_period)
    return TushareRateLimiter(max_calls, period_seconds, api_name)


def get_tushare_rate_limit_stats() -> RateLimitStats:
    return get_tushare_rate_limiter().stats()


def call_tushare_api(api_name: str, func: Callable[..., T], *args: object, **kwargs: object) -> T:
    limiter = get_tushare_api_rate_limiter(api_name)
    return call_provider_api("tushare", api_name, limiter.call, func, *args, **kwargs)
