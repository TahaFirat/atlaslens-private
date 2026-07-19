from __future__ import annotations


class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message_key: str,
        title: str,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message_key = message_key
        self.title = title
        self.retry_after_seconds = retry_after_seconds


def bad_request(code: str, message_key: str, title: str = "Invalid request") -> AppError:
    return AppError(400, code, message_key, title)


def unprocessable(code: str, message_key: str, title: str = "Invalid image") -> AppError:
    return AppError(422, code, message_key, title)
