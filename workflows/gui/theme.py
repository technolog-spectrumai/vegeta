"""Fusion style with a light or dark palette (the desktop's scheme by default), as Disk Sweeper does it."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import QApplication, QStyleFactory

LIGHT, DARK = "light", "dark"
_mode = LIGHT
STATE_COLORS = {LIGHT: {"completed": "#2e7d32", "running": "#1565c0", "partial": "#e65100", "prepared": "#6b6b6b", "failed": "#c62828", "cancelled": "#8e24aa"},
                DARK: {"completed": "#81c784", "running": "#64b5f6", "partial": "#ffb74d", "prepared": "#9a9a9a", "failed": "#ff6b6b", "cancelled": "#ce93d8"}}


def system_mode() -> str:
    try:
        return DARK if QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark else LIGHT
    except AttributeError:
        return LIGHT


def current_mode() -> str:
    return _mode


def state_color(state: str | None) -> QColor:
    return QColor(STATE_COLORS[_mode].get(state or "", "#6b6b6b"))


def _dark_palette() -> QPalette:
    p = QPalette()
    for role, c in {QPalette.ColorRole.Window: "#2b2b2b", QPalette.ColorRole.WindowText: "#e6e6e6", QPalette.ColorRole.Base: "#1f1f1f", QPalette.ColorRole.AlternateBase: "#2a2a2a",
                    QPalette.ColorRole.Text: "#e6e6e6", QPalette.ColorRole.Button: "#3a3a3a", QPalette.ColorRole.ButtonText: "#e6e6e6", QPalette.ColorRole.Highlight: "#3d8bd9",
                    QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: "#3a3a3a", QPalette.ColorRole.ToolTipText: "#e6e6e6"}.items():
        p.setColor(role, QColor(c))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#7a7a7a"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#7a7a7a"))
    return p


def apply_theme(mode: str) -> None:
    global _mode
    _mode = DARK if mode == DARK else LIGHT
    app = QApplication.instance()
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setPalette(_dark_palette() if _mode == DARK else app.style().standardPalette())
    app.setStyleSheet("QPushButton#runButton { font-weight: bold; padding: 8px 18px; background: #2e7d32; color: white; border-radius: 4px; }"
                      "QPushButton#runButton:disabled { background: #9a9a9a; }"
                      "QPushButton#stopButton { font-weight: bold; padding: 8px 18px; background: #c62828; color: white; border-radius: 4px; }"
                      "QPushButton#stopButton:disabled { background: #9a9a9a; }"
                      "QLabel[role='muted'] { color: #8a8a8a; } QLabel[role='warning'] { color: #e65100; font-weight: bold; }")
