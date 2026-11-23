"""The table of runs on disk (QAbstractTableModel over library.all_runs())."""
from __future__ import annotations

import datetime as dt

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QBrush

import library
from . import theme

COLUMNS = ("code", "chapter", "variant", "state", "cases", "created", "run id")


def age(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        t = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return ts
    s = (dt.datetime.now(dt.timezone.utc) - t).total_seconds()
    return f"{s / 60:.0f} min ago" if s < 7200 else (f"{s / 3600:.0f} h ago" if s < 2 * 86400 else f"{s / 86400:.0f} d ago")


class RunsModel(QAbstractTableModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.rows: list[dict] = []

    def refresh(self) -> None:
        self.beginResetModel()
        self.rows = library.all_runs()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section]
        return None

    def run_at(self, row: int) -> dict:
        return self.rows[row]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r = self.rows[index.row()]
        col = COLUMNS[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            c = r.get("cases") or {}
            return {"code": r.get("code", "?"), "chapter": r["chapter"], "variant": r.get("variant") or "", "state": r.get("state") or "?",
                    "cases": f"{c.get('done', 0)}/{c.get('planned', '?')}" + (f" ({c.get('failed')} failed)" if c.get("failed") else ""),
                    "created": age(r.get("created_at")), "run id": r["run_id"]}[col]
        if role == Qt.ItemDataRole.ForegroundRole and col == "state":
            return QBrush(theme.state_color(r.get("state")))
        if role == Qt.ItemDataRole.ToolTipRole:
            return r["run"]
        return None
