"""The desktop interface, offscreen (skipped where PyQt6 is not installed): the tree, the chapter panel, the command
preview, the runs table, the helpers. Nothing is launched."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import PyQt6.QtWidgets  # noqa: F401
except ImportError as exc:                      # not installed, or Qt's libraries missing (libEGL) on a headless machine
    pytest.skip(f"PyQt6 not usable here: {exc}", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflows"))


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_command_preview_and_helpers(app):
    import library
    from gui.runner import command_for, notebook_line, resume_argv
    book, name, info = library.by_code("K2")
    argv = command_for(info, {"--variant": "4Q-2", "--quality": "fine", "--processors": "4"}, mode="prepare", limits='--max-cases 1 --cases "α 12°"', cache=False)
    assert argv[:3] == ["workflows/kestrel/airframe_cfd.py", "--chapter", "stall"] and "--prepare-only" in argv and "--no-cache" in argv and "α 12°" in argv
    assert resume_argv(ROOT / "nowhere") is None


def test_window_tree_and_chapter_panel(app):
    from gui.app import MainWindow
    w = MainWindow()
    w.show()
    assert w.tree.topLevelItemCount() >= 1 and w.chapter[2]["code"] == "K1"
    w.select_code("k10")
    assert w.chapter_title.text().startswith("K10") and "K9" in w.chapter_pre.text() and "--chapter joint" in w.command.text()
    w.mode.setCurrentIndex(1)
    w.cache.setChecked(False)
    assert "--prepare-only" in w.command.text() and "--no-cache" in w.command.text()
    w.select_code("Z9")                                   # unknown: the status bar says so, the selection stays
    assert "no book with code" in w.status_label.text() and w.chapter[2]["code"] == "K10"
    assert w.runs_model.columnCount() == 7
    assert "✔" in w.legend.text() and w.chapter_runs.text().startswith("runs so far")
    w.close()


def test_batch_tab_collects_steps_and_writes_yaml(app, tmp_path):
    import batch
    from gui.app import MainWindow
    w = MainWindow()
    w.select_code("K1")
    w.add_to_batch()
    w.select_code("K11")
    w.limits.setText("--max-minutes 30")
    w.add_to_batch()
    assert [s.chapter for s in w.batch_steps] == ["K1", "K11"] and w.batch_table.rowCount() == 2
    w.batch_table.selectRow(1)
    w._move_step(-1)
    assert [s.chapter for s in w.batch_steps] == ["K11", "K1"]
    path = w.save_batch(tmp_path / "b.yaml")
    b = batch.load(path)
    assert [s.chapter for s in b.steps] == ["K11", "K1"] and b.steps[0].limits == "--max-minutes 30"
    w._clear_batch()
    w.load_batch(str(path))
    assert len(w.batch_steps) == 2 and w.batch_path == path
