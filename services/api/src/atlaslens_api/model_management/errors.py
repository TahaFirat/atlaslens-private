from __future__ import annotations


class ModelManagementError(RuntimeError):
    def __init__(self, code: str, *, subreason_code: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.subreason_code = subreason_code
