from __future__ import annotations

from pathlib import Path

import pytest

from atlaslens_api.demo_case import InvestorDemoCaseError
from atlaslens_api.demo_case_cli import _validate_database_url


def test_private_demo_cli_accepts_launcher_sqlite3_database(tmp_path: Path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'investor-demo.sqlite3').as_posix()}"

    assert _validate_database_url(database_url) == database_url


@pytest.mark.parametrize(
    "database_url",
    (
        "sqlite:///relative-demo.sqlite3",
        "sqlite:///:memory:",
        "postgresql://localhost/investor-demo",
    ),
)
def test_private_demo_cli_refuses_non_dedicated_database(database_url: str) -> None:
    with pytest.raises(InvestorDemoCaseError) as error:
        _validate_database_url(database_url)

    assert error.value.code == "demo_database_must_be_dedicated_sqlite"
