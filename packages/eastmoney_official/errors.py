from __future__ import annotations


ERROR_KINDS = {
    "request_error",
    "timeout_error",
    "rate_limit_error",
    "permission_error",
    "parse_error",
    "schema_error",
    "contract_error",
}


class P0ProviderError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        if kind not in ERROR_KINDS:
            raise ValueError(f"未知 Provider 错误类型: {kind}")
        super().__init__(message)
        self.kind = kind
