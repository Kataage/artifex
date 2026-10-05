from __future__ import annotations

from importlib.resources import files

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text


def alembic_config(database_url: str) -> Config:
    config = Config()
    script_location = files("artifex.db").joinpath("alembic")
    config.set_main_option("script_location", str(script_location))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def upgrade_database(database_url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url), revision)


def current_revision(database_url: str) -> str | None:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            try:
                value = connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar_one_or_none()
            except Exception:
                return None
            return str(value) if value is not None else None
    finally:
        engine.dispose()
