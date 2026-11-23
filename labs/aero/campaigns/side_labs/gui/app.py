"""Main window: the tree of books and chapters, the chosen chapter's options and Run / Stop, the live log, the runs."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QFont, QKeySequence
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QTableView, QTableWidget, QTableWidgetItem, QTabWidget, QToolBar,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

import batch as _batch

import library
import runkit
from . import theme
from .models import RunsModel
from .runner import Launcher, RunFollower, command_for, resume_argv, stop_run

ROOT = library.ROOT
CODE_ROLE = Qt.ItemDataRole.UserRole


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vegeta workflows")
        self.resize(1280, 820)
        theme.apply_theme(theme.system_mode())
        self.launcher = Launcher(self)
        self.follower = RunFollower(self)
        self.follower.status.connect(self._on_status)
        self.follower.log_lines.connect(self._on_log_lines)
        self.follower.finished.connect(self._on_run_finished)
        self.current_run: Path | None = None
        self.chapter: tuple[str, str, dict] | None = None
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(5000)
        self.live_timer.timeout.connect(self.refresh)

        self._build_toolbar()
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_tree())
        split.addWidget(self._build_chapter_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        bottom = self._build_bottom()
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(split)
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 2)
        self.setCentralWidget(vsplit)
        self._build_status_bar()
        self.refresh()
        self.select_code(library.BOOKS[next(iter(library.BOOKS))]["code"] + "1")

    # ---------------------------------------------------------------------------------------------- construction
    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        def action(text, shortcut, slot, tip):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setToolTip(f"{tip} ({shortcut})" if shortcut else tip)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a
        self.act_refresh = action("Refresh", "F5", self.refresh, "Re-read the runs on disk")
        self.act_runs_dir = action("Open runs folder", "Ctrl+O", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(library.RUNS))), "Open labs/aero/campaigns/side_labs/runs in the file manager")
        self.act_import = action("Import legacy results", None, self.import_legacy, "The first CFD campaign's file into K1's folder (once)")
        self.act_cache = action("Cache…", None, self.show_cache, "The cached cases per book; clean them")
        self.act_theme = action("Light / dark", "Ctrl+Shift+L", self.toggle_theme, "Switch the theme")

    def _build_tree(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 0, 6)
        title = QLabel("Books and chapters — pick a chapter (or type its code, e.g. K2)")
        lay.addWidget(title)
        row = QHBoxLayout()
        self.code_edit = QLineEdit()
        self.code_edit.setPlaceholderText("chapter code, e.g. K2 or k10")
        self.code_edit.setMaximumWidth(220)
        self.code_edit.returnPressed.connect(lambda: self.select_code(self.code_edit.text()))
        go = QPushButton("Go")
        go.clicked.connect(lambda: self.select_code(self.code_edit.text()))
        row.addWidget(self.code_edit)
        row.addWidget(go)
        row.addStretch(1)
        lay.addLayout(row)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["chapter", "status (latest run per variant)", "what it is"])
        self.tree.setColumnWidth(0, 230)
        self.tree.setColumnWidth(1, 300)
        self.legend = QLabel("✔ completed   ▶ running   ◐ partial / cancelled   ○ prepared   ✘ failed   — not run")
        self.legend.setProperty("role", "muted")
        lay.addWidget(self.legend)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.tree.itemDoubleClicked.connect(lambda *_: self.run_chapter())
        lay.addWidget(self.tree, 1)
        return box

    def _build_chapter_panel(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        self.chapter_title = QLabel("—")
        f = self.chapter_title.font()
        f.setPointSize(f.pointSize() + 3)
        f.setBold(True)
        self.chapter_title.setFont(f)
        self.chapter_desc = QLabel("")
        self.chapter_desc.setWordWrap(True)
        self.chapter_meta = QLabel("")
        self.chapter_meta.setWordWrap(True)
        self.chapter_meta.setProperty("role", "muted")
        self.chapter_pre = QLabel("")
        self.chapter_pre.setWordWrap(True)
        self.chapter_pre.setProperty("role", "warning")
        lay.addWidget(self.chapter_title)
        lay.addWidget(self.chapter_desc)
        lay.addWidget(self.chapter_meta)
        lay.addWidget(self.chapter_pre)
        self.chapter_runs = QLabel("")
        self.chapter_runs.setWordWrap(True)
        lay.addWidget(self.chapter_runs)

        self.options_box = QGroupBox("Options")
        self.options_form = QFormLayout(self.options_box)
        self.option_widgets: dict[str, QWidget] = {}
        lay.addWidget(self.options_box)

        run_box = QGroupBox("Run")
        form = QFormLayout(run_box)
        self.mode = QComboBox()
        self.mode.addItems(["run the solver", "prepare only (write the cases, no solver)", "list the cases only"])
        self.mode.currentIndexChanged.connect(self._update_command)
        self.cache = QCheckBox("use the cache (cases this book already solved are read back)")
        self.cache.setChecked(True)
        self.cache.stateChanged.connect(self._update_command)
        self.ignore_pre = QCheckBox("ignore prerequisites (recorded in the manifest)")
        self.ignore_pre.stateChanged.connect(self._update_command)
        self.limits = QLineEdit()
        self.limits.setPlaceholderText('e.g. --max-cases 2 --max-minutes 90 --cases "α 12°"')
        self.limits.textChanged.connect(self._update_command)
        form.addRow("mode", self.mode)
        form.addRow("", self.cache)
        form.addRow("", self.ignore_pre)
        form.addRow("limits", self.limits)
        self.command = QLabel("")
        self.command.setWordWrap(True)
        self.command.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.command.setFont(QFont("monospace"))
        form.addRow("command", self.command)
        lay.addWidget(run_box)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("▶  Run chapter")
        self.run_button.setObjectName("runButton")
        self.run_button.setToolTip("Start the chapter's script detached: closing this window never stops it (Ctrl+R)")
        self.run_button.clicked.connect(self.run_chapter)
        self.stop_button = QPushButton("■  Stop run")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setToolTip("SIGINT to the running script: the solver stops, finished cases are kept, resume later")
        self.stop_button.clicked.connect(self.stop_current)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        self.batch_button = QPushButton("＋  Add to batch")
        self.batch_button.setToolTip("Append this chapter, with these options, to the batch (the Batch tab below): save it as batch.yaml, run it here or on another machine (Ctrl+B)")
        self.batch_button.clicked.connect(self.add_to_batch)
        buttons.addWidget(self.batch_button)
        buttons.addStretch(1)
        lay.addLayout(buttons)
        act_b = QAction(self)
        act_b.setShortcut(QKeySequence("Ctrl+B"))
        act_b.triggered.connect(self.add_to_batch)
        self.addAction(act_b)
        act = QAction(self)
        act.setShortcut(QKeySequence("Ctrl+R"))
        act.triggered.connect(self.run_chapter)
        self.addAction(act)
        lay.addStretch(1)
        return box

    def _build_bottom(self) -> QWidget:
        self.tabs = QTabWidget()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20000)
        self.log_view.setFont(QFont("monospace"))
        self.tabs.addTab(self.log_view, "Log of the current run")

        runs = QWidget()
        rl = QVBoxLayout(runs)
        rl.setContentsMargins(0, 0, 0, 0)
        self.runs_model = RunsModel(self)
        self.runs_table = QTableView()
        self.runs_table.setModel(self.runs_model)
        self.runs_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.runs_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.runs_table.setAlternatingRowColors(True)
        self.runs_table.verticalHeader().hide()
        header = self.runs_table.horizontalHeader()
        for c in range(6):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.runs_table.doubleClicked.connect(lambda _i: self.follow_selected())
        rl.addWidget(self.runs_table, 1)
        rb = QHBoxLayout()
        for text, slot, tip in (("Follow", self.follow_selected, "Show this run's log and progress"), ("Resume", self.resume_selected, "Continue an interrupted or partial run"),
                                ("Stop", self.stop_selected, "SIGINT to a running run"),
                                ("Open folder", self.open_selected, "The run directory in the file manager")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            rb.addWidget(b)
        rb.addStretch(1)
        rl.addLayout(rb)
        self.tabs.addTab(runs, "Runs on disk")
        self.tabs.addTab(self._build_batch_tab(), "Batch")
        return self.tabs

    # ---------------------------------------------------------------------------------------------- the batch
    def _build_batch_tab(self) -> QWidget:
        """An ordered list of chapter runs, saved as batch.yaml and run by ``wf batch`` — here, detached, or on
        another machine (a cloud box with the solvers): the file is the hand-over."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(QLabel("name"))
        self.batch_name = QLineEdit("kestrel batch")
        self.batch_name.setMaximumWidth(260)
        head.addWidget(self.batch_name)
        self.batch_stop = QCheckBox("stop at the first failed step")
        self.batch_stop.setChecked(True)
        head.addWidget(self.batch_stop)
        self.batch_file_label = QLabel("not saved yet")
        self.batch_file_label.setProperty("role", "muted")
        head.addWidget(self.batch_file_label, 1)
        lay.addLayout(head)
        self.batch_table = QTableWidget(0, 6)
        self.batch_table.setHorizontalHeaderLabels(["#", "chapter", "variant", "options", "mode", "limits"])
        self.batch_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.batch_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.batch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.batch_table.verticalHeader().hide()
        hh = self.batch_table.horizontalHeader()
        for c in range(5):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.batch_table, 1)
        self.batch_steps: list = []
        self.batch_path: Path | None = None
        row = QHBoxLayout()
        for text, slot, tip in (("▲", lambda: self._move_step(-1), "Earlier"), ("▼", lambda: self._move_step(+1), "Later"), ("Remove", self._remove_step, "Remove the selected step"),
                                ("Clear", self._clear_batch, "Empty the batch"), ("Load…", self.load_batch, "Read a batch.yaml"),
                                ("Save batch.yaml…", self.save_batch, "Write the batch to a YAML file: wf batch <file> runs it, here or on another machine"),
                                ("Show commands", self.dry_run_batch, "The commands the batch would run, in order, in the log"),
                                ("▶  Run batch", self.run_batch, "Save, then run the steps one after the other, detached; the log and progress appear in the Log tab")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            if text.startswith("▶"):
                b.setObjectName("runButton")
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        return box

    def _batch(self) -> "_batch.Batch":
        return _batch.Batch(self.batch_name.text().strip() or "batch", list(self.batch_steps), self.batch_stop.isChecked())

    def _refresh_batch_table(self) -> None:
        t = self.batch_table
        t.setRowCount(len(self.batch_steps))
        for i, st in enumerate(self.batch_steps):
            opts = " ".join(f"{k}={v}" for k, v in st.options.items()) + ("" if st.cache else " no-cache") + (" ignore-prerequisites" if st.ignore_prerequisites else "")
            try:
                _, name, info = _batch.resolve(st)
                ch = f"{info['code']} {name}"
            except KeyError:
                ch = f"{st.chapter} (unknown)"
            for c, txt in enumerate((str(i + 1), ch, st.variant, opts.strip(), st.mode, st.limits)):
                t.setItem(i, c, QTableWidgetItem(txt))
        self.tabs.setTabText(2, f"Batch ({len(self.batch_steps)})" if self.batch_steps else "Batch")

    def add_to_batch(self) -> None:
        if not self.chapter:
            return
        _, name, info = self.chapter
        opts = {k.lstrip("-"): v for k, v in self._options().items() if v not in (None, "") and k != "--variant"}
        mode = "prepare" if self.mode.currentIndex() == 1 else "run"
        self.batch_steps.append(_batch.Step(info["code"], self._options().get("--variant", ""), opts, mode, self.cache.isChecked(), self.limits.text().strip(),
                                            self.ignore_pre.isChecked()))
        self._refresh_batch_table()
        self.batch_table.selectRow(len(self.batch_steps) - 1)
        self.tabs.setCurrentIndex(2)
        self.status_label.setText(f"{info['code']} {name} added to the batch ({len(self.batch_steps)} step(s))")

    def _move_step(self, d: int) -> None:
        i = self.batch_table.currentRow()
        j = i + d
        if i < 0 or not (0 <= j < len(self.batch_steps)):
            return
        self.batch_steps[i], self.batch_steps[j] = self.batch_steps[j], self.batch_steps[i]
        self._refresh_batch_table()
        self.batch_table.selectRow(j)

    def _remove_step(self) -> None:
        i = self.batch_table.currentRow()
        if 0 <= i < len(self.batch_steps):
            del self.batch_steps[i]
            self._refresh_batch_table()

    def _clear_batch(self) -> None:
        self.batch_steps.clear()
        self.batch_path = None
        self.batch_file_label.setText("not saved yet")
        self._refresh_batch_table()

    def load_batch(self, path=None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load a batch", str(ROOT), "batch files (*.yaml *.yml)")
            if not path:
                return
        try:
            b = _batch.load(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Batch", f"{path}:\n{exc}")
            return
        self.batch_steps = list(b.steps)
        self.batch_name.setText(b.name)
        self.batch_stop.setChecked(b.stop_on_failure)
        self.batch_path = Path(path)
        self.batch_file_label.setText(str(self.batch_path))
        self._refresh_batch_table()
        self.tabs.setCurrentIndex(2)

    def save_batch(self, path=None) -> Path | None:
        if not self.batch_steps:
            QMessageBox.information(self, "Batch", "The batch is empty: pick a chapter, set its options and press 'Add to batch'.")
            return None
        probs = _batch.validate(self._batch())
        if probs:
            QMessageBox.warning(self, "Batch", "\n".join(probs))
            return None
        if not path:
            default = str(self.batch_path or (ROOT / f"{runkit.slug(self.batch_name.text().strip() or 'batch')}.yaml"))
            path, _ = QFileDialog.getSaveFileName(self, "Save the batch as YAML", default, "batch files (*.yaml *.yml)")
            if not path:
                return None
        self.batch_path = _batch.save(self._batch(), path)
        self.batch_file_label.setText(f"{self.batch_path}   —   run it anywhere:  wf batch {self.batch_path.name}")
        self.status_label.setText(f"batch saved: {self.batch_path}")
        return self.batch_path

    def dry_run_batch(self) -> None:
        if not self.batch_steps:
            return
        lines: list[str] = []
        _batch.run(self._batch(), dry_run=True, log=lines.append)
        self.log_view.clear()
        self.log_view.setPlainText("\n".join(lines))
        self.tabs.setCurrentIndex(0)

    def run_batch(self) -> None:
        path = self.save_batch(self.batch_path) if self.batch_path else self.save_batch()
        if path is None:
            return
        b = self._batch()
        if QMessageBox.question(self, "Run batch", f"Run the {len(b.steps)} step(s) of {path.name} one after the other, detached?\n\n$ wf batch {path}\n\n"
                                "Each step is a chapter run with its own run directory; the batch folder under labs/aero/campaigns/side_labs/runs/_batches/ keeps batch.json with every "
                                "output file. The same file runs on any machine with the solvers.") != QMessageBox.StandardButton.Yes:
            return
        before = {p.name for p in (library.RUNS / "_batches").glob("*")} if (library.RUNS / "_batches").is_dir() else set()
        self.log_view.clear()
        ok, cmd, _ = self.launcher.launch(["labs/aero/campaigns/side_labs/cli.py", "batch", str(path)], label=f"batch_{runkit.slug(b.name)}")
        self.log_view.appendPlainText(f"$ {cmd}")
        if not ok:
            return
        bdir = None
        for _ in range(20):
            QApplication.processEvents()
            new = [p for p in (library.RUNS / "_batches").glob("*") if p.name not in before and p.is_dir()] if (library.RUNS / "_batches").is_dir() else []
            if new:
                bdir = max(new, key=lambda p: p.stat().st_mtime)
                break
            time.sleep(0.25)
        if bdir is None:
            self.log_view.appendPlainText("(the batch folder appears under labs/aero/campaigns/side_labs/runs/_batches/ in a moment; press Refresh and follow its runs from the table)")
            QTimer.singleShot(4000, self.refresh)
            return
        self.follow(bdir)
        self.refresh()

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(320)
        self.progress.setTextVisible(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.heartbeat_label = QLabel("")
        self.heartbeat_label.setProperty("role", "muted")
        sb.addWidget(self.status_label, 1)
        sb.addPermanentWidget(self.heartbeat_label)
        sb.addPermanentWidget(self.progress)

    # ---------------------------------------------------------------------------------------------- the tree
    def refresh(self) -> None:
        selected = self.chapter[2]["code"] if self.chapter else None
        live = False
        self.tree.clear()
        for book, b in library.BOOKS.items():
            chs = library.chapters(book)
            top = QTreeWidgetItem([f"{book}  ({b['code']})", b["title"], ""])
            top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(top)
            kinds = []
            for info in chs.values():
                if info["kind"] not in kinds:
                    kinds.append(info["kind"])
            for kind in kinds:
                k = QTreeWidgetItem([kind.upper(), library.kind_of(bname, kind)["title"], ""])
                k.setFlags(k.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                top.addChild(k)
                for name, info in chs.items():
                    if info["kind"] != kind:
                        continue
                    cs = library.chapter_status(book, name)
                    pre = ("  [after " + ", ".join(chs[p]["code"] for p in info["prerequisites"]) + "]") if info["prerequisites"] else ""
                    it = QTreeWidgetItem([f"{cs['mark']}  {info['code']}  {name}", cs["summary"], info["description"] + pre])
                    it.setData(0, CODE_ROLE, info["code"])
                    it.setToolTip(2, info["description"])
                    it.setToolTip(1, "\n".join(f"{v}: {r['state']} — {r['run_id']}" for v, r in cs["latest"].items()) or "no run yet")
                    if cs["runs"]:
                        it.setForeground(1, theme.state_color(cs["overall"]))
                        it.setForeground(0, theme.state_color(cs["overall"]))
                    k.addChild(it)
                    if cs["overall"] == "running":
                        live = True
        self.tree.expandAll()
        self.runs_model.refresh()
        if selected:
            self.select_code(selected)
        self.live_timer.start() if live else self.live_timer.stop()      # while a run is live the tree follows it

    def select_code(self, code: str) -> None:
        try:
            book, name, info = library.by_code(code)
        except KeyError as exc:
            self.status_label.setText(str(exc))
            return
        self.chapter = (book, name, info)
        self.code_edit.setText(info["code"])
        it = self.tree.invisibleRootItem()
        stack = [it.child(i) for i in range(it.childCount())]
        while stack:
            node = stack.pop()
            if node.data(0, CODE_ROLE) == info["code"]:
                self.tree.blockSignals(True)
                self.tree.setCurrentItem(node)
                self.tree.blockSignals(False)
                break
            stack.extend(node.child(i) for i in range(node.childCount()))
        self._show_chapter()

    def _on_tree_selection(self) -> None:
        items = self.tree.selectedItems()
        if items and items[0].data(0, CODE_ROLE):
            self.select_code(items[0].data(0, CODE_ROLE))

    def _show_chapter(self) -> None:
        book, name, info = self.chapter
        kind = library.kind_of(book, info["kind"])
        chs = library.chapters(book)
        self.chapter_title.setText(f"{info['code']}  {name}")
        self.chapter_desc.setText(info["description"])
        try:
            ns = library.needs_status(book, name)["line"]
        except Exception as exc:                                          # the panel never fails on a probe
            ns = f"not checked ({exc})"
        self.chapter_meta.setText(f"{kind['title']}\nneeds: {kind['needs']} · time: {kind['time']}\non this machine: {ns}")
        self.chapter_pre.setText(("prerequisites (same book): " + ", ".join(f"{chs[p]['code']} ({p})" for p in info["prerequisites"])) if info["prerequisites"] else "")
        self.chapter_pre.setVisible(bool(info["prerequisites"]))
        cs = library.chapter_status(book, name)
        self.chapter_runs.setText(("runs so far: " + cs["summary"] + f"  ({cs['counts']['completed']} completed, {cs['counts']['running']} running, {cs['counts']['failed']} failed, {cs['counts']['other']} other)")
                                  if cs["runs"] else "runs so far: none")
        while self.options_form.rowCount():
            self.options_form.removeRow(0)
        self.option_widgets = {}
        for flag, choices, default in kind["options"]:
            if choices:
                w = QComboBox()
                w.addItems(list(choices))
                w.setCurrentText(default)
                w.currentIndexChanged.connect(self._update_command)
            else:
                w = QLineEdit(str(default))
                w.setMaximumWidth(80)
                w.textChanged.connect(self._update_command)
            self.option_widgets[flag] = w
            self.options_form.addRow(flag.lstrip("-"), w)
        self._update_command()

    def _options(self) -> dict:
        out = {}
        for flag, w in self.option_widgets.items():
            out[flag] = w.currentText() if isinstance(w, QComboBox) else w.text().strip()
        return out

    def _argv(self) -> list[str]:
        _, _, info = self.chapter
        mode = ("run", "prepare", "list")[self.mode.currentIndex()]
        argv = command_for(info, self._options(), mode=mode, limits=self.limits.text(), cache=self.cache.isChecked())
        if self.ignore_pre.isChecked():
            argv.append("--ignore-prerequisites")
        return argv

    def _update_command(self, *_) -> None:
        if not self.chapter:
            return
        import shlex
        self.command.setText(shlex.join([".venv/bin/python", *self._argv()]))

    # ---------------------------------------------------------------------------------------------- running
    def run_chapter(self) -> None:
        if not self.chapter:
            return
        book, name, info = self.chapter
        argv = self._argv()
        variant = self._options().get("--variant", "")
        if "--list" in argv:
            self._run_listing(argv)
            return
        missing = [] if self.ignore_pre.isChecked() else library.check_prerequisites(book, name, variant=variant)
        if missing:
            QMessageBox.warning(self, "Prerequisites", f"{info['code']} needs a completed run of its prerequisite chapter(s) for {variant} first:\n\n" + "\n".join(missing)
                                + "\n\nRun them first, or tick 'ignore prerequisites'.")
            return
        import shlex
        if QMessageBox.question(self, "Run chapter", f"Start {info['code']} {name} detached?\n\n{shlex.join(['.venv/bin/python', *argv])}\n\nIt keeps running when this window closes; "
                                "the log and progress appear below.") != QMessageBox.StandardButton.Yes:
            return
        self.log_view.clear()
        self.status_label.setText(f"starting {info['code']}…")
        QApplication.processEvents()
        ok, cmd, run_dir = self.launcher.launch(argv, label=f"{info['code']}_{name}")
        self.log_view.appendPlainText(f"$ {cmd}")
        if not ok:
            self.status_label.setText("could not start the script")
            return
        if run_dir is None:
            self.log_view.appendPlainText("(the run directory appears once the cases are built; press Refresh and follow it from the runs table)")
            QTimer.singleShot(4000, self.refresh)
            return
        self.follow(run_dir)
        self.refresh()

    def _run_listing(self, argv: list[str]) -> None:
        import subprocess
        self.log_view.clear()
        self.status_label.setText("listing the cases…")
        QApplication.processEvents()
        out = subprocess.run([sys.executable, str(ROOT / argv[0]), *argv[1:]], capture_output=True, text=True, cwd=ROOT, timeout=600)
        self.log_view.setPlainText(out.stdout + out.stderr)
        self.tabs.setCurrentIndex(0)
        self.status_label.setText("Ready")

    def follow(self, run_dir: Path) -> None:
        self.current_run = Path(run_dir)
        self.log_view.clear()
        self.log_view.appendPlainText(f"— following {self.current_run} —")
        self.follower.follow(self.current_run)
        self.tabs.setCurrentIndex(0)

    def _on_log_lines(self, lines: list[str]) -> None:
        for ln in lines:
            self.log_view.appendPlainText(ln)

    def _on_status(self, st: dict) -> None:
        c = st.get("cases") or {}
        planned, done = int(c.get("planned") or c.get("n") or 0), int(c.get("done") or 0)
        state = st.get("state", "?")
        if planned:
            self.progress.setRange(0, planned)
            self.progress.setValue(int(min(done, planned)))
            self.progress.setFormat(f"{done:g}/{planned} cases" + (f" · ~{st['eta_text'].rsplit('~', 1)[-1]}" if st.get("eta_text") and state == "running" else ""))
        else:
            self.progress.setRange(0, 0 if state == "running" else 1)
        hb = st.get("heartbeat") or {}
        if state == "running" and hb:
            adv = "advancing" if hb.get("advancing") else f"quiet for {hb.get('quiet_for_s', 0):.0f} s"
            self.heartbeat_label.setText(f"{hb.get('case', '')[:40]} · {hb.get('case_elapsed_s', 0) / 60:.1f} min · {adv} · {hb.get('last_progress', '')[:40]}")
        else:
            self.heartbeat_label.setText("")
        name = self.current_run.name if self.current_run else ""
        eta = f" — about {st['eta_text'].split(', ', 1)[-1]}, finishing ~{st.get('eta_at', '')[-5:]}" if st.get("eta_text") and state == "running" else ""
        self.status_label.setText(f"{name}: {state} — {done:g} of {planned} cases" + (f", {c.get('failed')} failed" if c.get("failed") else "") + (f", {c.get('reused')} reused" if c.get("reused") else "") + eta)
        self.stop_button.setEnabled(state == "running")

    def _on_run_finished(self, st: dict) -> None:
        self.refresh()

    def stop_current(self) -> None:
        if self.current_run:
            self.log_view.appendPlainText(stop_run(self.current_run))

    # ---------------------------------------------------------------------------------------------- runs table
    def _selected_run(self) -> dict | None:
        rows = self.runs_table.selectionModel().selectedRows() if self.runs_table.selectionModel() else []
        return self.runs_model.run_at(rows[0].row()) if rows else None

    def follow_selected(self) -> None:
        r = self._selected_run()
        if r:
            self.follow(Path(r["run"]))

    def stop_selected(self) -> None:
        r = self._selected_run()
        if r:
            self.log_view.appendPlainText(stop_run(Path(r["run"])))
            self.tabs.setCurrentIndex(0)

    def resume_selected(self) -> None:
        r = self._selected_run()
        if not r:
            return
        argv = resume_argv(Path(r["run"]))
        if argv is None:
            QMessageBox.information(self, "Resume", "This run cannot be resumed from here (no script for its kind).")
            return
        if r.get("state") == "running":
            QMessageBox.information(self, "Resume", "This run is still running.")
            return
        ok, cmd, _ = self.launcher.launch(argv, label=f"resume_{r['run_id']}")
        self.log_view.clear()
        self.log_view.appendPlainText(f"$ {cmd}")
        if ok:
            self.follow(Path(r["run"]))

    def open_selected(self) -> None:
        r = self._selected_run()
        if r:
            QDesktopServices.openUrl(QUrl.fromLocalFile(r["run"]))

    # ---------------------------------------------------------------------------------------------- tools
    def import_legacy(self) -> None:
        import subprocess
        out = subprocess.run([sys.executable, str(ROOT / library.IMPORTS["import_legacy"]["script"])], capture_output=True, text=True, cwd=ROOT, timeout=600)
        self.log_view.setPlainText(out.stdout + out.stderr)
        self.tabs.setCurrentIndex(0)
        self.refresh()

    def show_cache(self) -> None:
        st = runkit.Cache.stats()
        text = "\n".join(f"{k}: {v['cases']} case(s), {v['kB']} kB" for k, v in st.items()) or "empty"
        box = QMessageBox(self)
        box.setWindowTitle("Cache")
        box.setText("Solved cases reused across runs of a book (side_labs/cache/):\n\n" + text)
        clean = box.addButton("Clean cache", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is clean and st:
            n = runkit.Cache.clean()
            self.status_label.setText(f"removed {n} cached case(s); the runs keep their native results")

    def toggle_theme(self) -> None:
        theme.apply_theme(theme.LIGHT if theme.current_mode() == theme.DARK else theme.DARK)
        self.runs_model.refresh()


def main(argv=None) -> int:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Vegeta workflows")
    w = MainWindow()
    w.show()
    return app.exec()
