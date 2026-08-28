"""Read-only Streamlit diagnostics console.

The React operator console is the primary dashboard. This file intentionally
keeps only a small database inspector so Streamlit cannot mutate runtime state.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DB_FILE = ROOT / "data" / "bot.db"
MAX_ROWS = 100


def _table_names(db_file: Path = DB_FILE) -> list[str]:
    if not db_file.exists():
        return []
    uri = f"file:{db_file.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [str(row[0]) for row in rows]


def _read_table(table: str, db_file: Path = DB_FILE, limit: int = MAX_ROWS) -> tuple[list[str], list[tuple]]:
    if table not in set(_table_names(db_file)):
        return [], []
    uri = f"file:{db_file.as_posix()}?mode=ro"
    safe_limit = max(1, min(int(limit), MAX_ROWS))
    with sqlite3.connect(uri, uri=True) as conn:
        cursor = conn.execute(f'SELECT * FROM "{table}" LIMIT ?', (safe_limit,))
        columns = [str(item[0]) for item in cursor.description or []]
        rows = cursor.fetchall()
    return columns, rows


def _rows_as_dicts(columns: Iterable[str], rows: Iterable[tuple]) -> list[dict[str, object]]:
    keys = list(columns)
    return [dict(zip(keys, row)) for row in rows]


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("streamlit is not installed") from exc

    st.set_page_config(page_title="AutoTraderBot DB Inspector", layout="wide")
    st.title("AutoTraderBot DB Inspector")
    st.caption("Read-only diagnostics. Runtime actions are handled by the React console and API.")

    if not DB_FILE.exists():
        st.info("Database not found: " + str(DB_FILE))
        return

    tables = _table_names()
    if not tables:
        st.info("No tables found.")
        return

    selected = st.selectbox("Table", tables)
    columns, rows = _read_table(str(selected))
    st.dataframe(_rows_as_dicts(columns, rows), use_container_width=True)


if __name__ == "__main__":
    main()
