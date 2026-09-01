"""SQLite access, and the content hash that proves the generator is deterministic.

`DecisionLog` is append-only. SQLAlchemy will happily let you update any row, so the
constraint is enforced here with a database trigger rather than by convention — a
convention is a comment, and the audit claim needs to survive somebody being in a hurry.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from .config import GENERATED_DIR

DEFAULT_DB_NAME = "deduction_desk.db"

# Rows in these tables are immutable once written. The trigger below enforces it.
APPEND_ONLY_TABLES = ("decision_log",)


def db_path(name: str = DEFAULT_DB_NAME) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_DIR / name


def make_engine(path: Path | None = None, *, echo: bool = False) -> Engine:
    target = path or db_path()
    engine = create_engine(f"sqlite:///{target}", echo=echo)

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):  # pragma: no cover - driver callback
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        # NORMAL rather than FULL: the batch writes tens of thousands of rows and this is
        # a reproducible artefact, not a system of record. If the machine loses power
        # mid-run you re-run it.
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


def _install_append_only_triggers(engine: Engine) -> None:
    """Make `DecisionLog` immutable at the database level.

    A trigger, not a code convention. The submission's audit claim is that any case can be
    reconstructed from this table alone; if some later code path can quietly rewrite a row,
    that claim is worth nothing and nobody would find out.
    """
    with engine.begin() as conn:
        for table in APPEND_ONLY_TABLES:
            for op in ("UPDATE", "DELETE"):
                conn.execute(
                    text(
                        f"""
                        CREATE TRIGGER IF NOT EXISTS no_{op.lower()}_{table}
                        BEFORE {op} ON {table}
                        BEGIN
                            SELECT RAISE(ABORT,
                                '{table} is append-only: {op} is not permitted');
                        END;
                        """
                    )
                )


def init_db(path: Path | None = None, *, reset: bool = False) -> Engine:
    """Create the schema. `reset=True` deletes the file first."""
    target = path or db_path()
    if reset and target.exists():
        target.unlink()
    engine = make_engine(target)
    SQLModel.metadata.create_all(engine)
    _install_append_only_triggers(engine)
    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """A committed session.

    `expire_on_commit=False` because callers routinely build a summary from the same
    objects they just persisted. With the default, every attribute read after the commit
    triggers a reload against a session that has already closed, which surfaces as
    `DetachedInstanceError` a long way from the cause. These are plain value objects
    written once and never concurrently mutated, so there is nothing to go stale.
    """
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


# ----------------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------------
def content_hash(engine: Engine, *, exclude: tuple[str, ...] = ()) -> str:
    """A stable hash of the database's contents.

    Deliberately hashes *contents*, not the file. Two SQLite files with identical rows can
    differ byte-for-byte — page allocation, free lists and vacuum state all vary — so a
    file checksum would report spurious non-determinism and send you hunting a bug that
    is not there. This walks every table in sorted order, every row sorted by primary key,
    and hashes the rendered values.
    """
    inspector_tables = sorted(SQLModel.metadata.tables)
    digest = hashlib.sha256()

    with engine.connect() as conn:
        for table_name in inspector_tables:
            if table_name in exclude:
                continue
            table = SQLModel.metadata.tables[table_name]
            order_cols = [c.name for c in table.primary_key.columns] or [
                c.name for c in table.columns
            ]
            cols = ", ".join(f'"{c.name}"' for c in table.columns)
            order = ", ".join(f'"{c}"' for c in order_cols)
            digest.update(f"\n## {table_name}\n".encode())
            for row in conn.execute(text(f'SELECT {cols} FROM "{table_name}" ORDER BY {order}')):
                digest.update(("\x1f".join("" if v is None else str(v) for v in row)).encode())
                digest.update(b"\x1e")

    return digest.hexdigest()


def table_counts(engine: Engine) -> dict[str, int]:
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for name in sorted(SQLModel.metadata.tables):
            counts[name] = int(conn.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar() or 0)
    return counts
