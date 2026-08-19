from __future__ import annotations


_KINDS = {
    "request_error",
    "timeout_error",
    "rate_limit_error",
    "permission_error",
    "parse_error",
    "schema_error",
    "contract_error",
}


class TushareMigrationError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        if kind not in _KINDS:
            raise ValueError(f"未知 Tushare migration 错误类型: {kind}")
        super().__init__(message)
        self.kind = kind
