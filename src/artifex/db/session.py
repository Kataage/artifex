from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from artifex.db.migrations import upgrade_database


def _prepare_sqlite_parent(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return
    database = url.database
    if database is None or database in {":memory:", ""}:
        return
    Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


class Database:
    def __init__(self, database_url: str) -> None:
        _prepare_sqlite_parent(database_url)
        self.database_url = database_url
        self.engine: Engine = create_engine(database_url)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def migrate(self) -> None:
        upgrade_database(self.database_url)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
