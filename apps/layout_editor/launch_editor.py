"""Tkinter launcher for the warehouse layout mask editor."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import Tk, messagebox, simpledialog, ttk

from apps.layout_editor.editor_matplotlib import GridEditor as MatplotlibGridEditor
from apps.layout_editor.editor_tkinter import GridEditor as TkinterGridEditor
from whl_core.layout_io import empty_mask_bundle, load_mask, mask_to_grid
from whl_core.paths import MASK_DIR
from whl_core.registry import add_layout, delete_layout, list_layouts
from whl_visualization.layout_plot import plot_layout_grid

EDITOR_BACKENDS = {
    "Tkinter canvas": TkinterGridEditor,
    "Matplotlib": MatplotlibGridEditor,
}

UNKNOWN = "unknown"


@dataclass(frozen=True)
class LayoutRow:
    """Display row for a registered layout mask."""

    layout_id: int
    filename: str
    layout_name: str
    rows: str
    cols: str
    aisle_width: str
    modified: str
    missing: bool = False

    def values(self) -> tuple[str, str, str, str, str, str, str]:
        """Return Treeview string values."""
        filename = f"{self.filename} [missing]" if self.missing else self.filename
        return (
            str(self.layout_id),
            filename,
            self.layout_name,
            self.rows,
            self.cols,
            self.aisle_width,
            self.modified,
        )


def _npz_filename(name: str) -> str:
    """Return a normalized ``.npz`` filename for a layout name."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Layout name must not be empty.")
    return clean_name if clean_name.lower().endswith(".npz") else f"{clean_name}.npz"


def _mask_path(filename: str) -> Path:
    """Return the repository-local mask path for a registry filename."""
    return MASK_DIR / _npz_filename(filename)


def layout_filename_available(
    name: str,
    layouts: dict[int, str] | None = None,
    mask_dir: Path = MASK_DIR,
) -> bool:
    """Return whether a proposed layout filename is unused."""
    try:
        filename = _npz_filename(name)
    except ValueError:
        return False

    registered = set((layouts or list_layouts()).values())
    return filename not in registered and not (mask_dir / filename).exists()


def prepare_duplicate_masks(source_masks: dict, new_name: str) -> dict:
    """Return an editable copy of source masks with a new layout name."""
    filename = _npz_filename(new_name)
    duplicate = {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in source_masks.items()
    }
    duplicate["name"] = Path(filename).stem
    return duplicate


def _format_modified_time(path: Path) -> str:
    """Return a compact local modified-time string for a path."""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def layout_row_from_registry_entry(
    layout_id: int,
    filename: str,
    mask_dir: Path = MASK_DIR,
) -> LayoutRow:
    """Build selector metadata for one registry entry without raising on errors."""
    path = mask_dir / _npz_filename(filename)
    if not path.exists():
        return LayoutRow(
            layout_id=layout_id,
            filename=_npz_filename(filename),
            layout_name="missing",
            rows="missing",
            cols="missing",
            aisle_width="missing",
            modified="missing",
            missing=True,
        )

    try:
        masks = load_mask(path)
    except Exception:
        return LayoutRow(
            layout_id=layout_id,
            filename=path.name,
            layout_name=UNKNOWN,
            rows=UNKNOWN,
            cols=UNKNOWN,
            aisle_width=UNKNOWN,
            modified=_format_modified_time(path),
        )

    return LayoutRow(
        layout_id=layout_id,
        filename=path.name,
        layout_name=str(masks.get("name", "")) or UNKNOWN,
        rows=str(masks.get("rows", "")) or UNKNOWN,
        cols=str(masks.get("cols", "")) or UNKNOWN,
        aisle_width=str(masks.get("aisle_width", "")) or UNKNOWN,
        modified=_format_modified_time(path),
    )


def layout_rows_from_registry(
    layouts: dict[int, str],
    mask_dir: Path = MASK_DIR,
) -> list[LayoutRow]:
    """Return selector rows for registry entries."""
    return [
        layout_row_from_registry_entry(layout_id, filename, mask_dir)
        for layout_id, filename in sorted(layouts.items())
    ]


def filter_layout_rows(rows: list[LayoutRow], query: str) -> list[LayoutRow]:
    """Filter rows by filename or layout name."""
    clean_query = query.strip().lower()
    if not clean_query:
        return rows
    return [
        row
        for row in rows
        if clean_query in row.filename.lower()
        or clean_query in row.layout_name.lower()
    ]


class LayoutSelectorDialog:
    """Treeview selector for registry layouts."""

    columns = (
        "id",
        "filename",
        "layout_name",
        "rows",
        "cols",
        "aisle_width",
        "modified",
    )

    def __init__(
        self,
        parent: tk.Misc,
        rows: list[LayoutRow],
        title: str,
        action_label: str,
    ) -> None:
        self.rows = rows
        self.filtered_rows = rows
        self.selected: tuple[int, str] | None = None
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("860x420")
        self.window.transient(parent)
        self.window.grab_set()

        frame = ttk.Frame(self.window, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Search").pack(anchor="w")
        self.search_var = tk.StringVar(master=self.window)
        search = ttk.Entry(frame, textvariable=self.search_var)
        search.pack(fill="x", pady=(0, 8))
        search.focus_set()

        table_frame = ttk.Frame(frame)
        table_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            table_frame,
            columns=self.columns,
            show="headings",
            selectmode="browse",
        )
        y_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview,
        )
        x_scroll = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.tree.xview,
        )
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        headings = {
            "id": "ID",
            "filename": "Filename",
            "layout_name": "Layout name",
            "rows": "Rows",
            "cols": "Cols",
            "aisle_width": "Aisle width",
            "modified": "Last modified",
        }
        widths = {
            "id": 50,
            "filename": 210,
            "layout_name": 160,
            "rows": 70,
            "cols": 70,
            "aisle_width": 90,
            "modified": 140,
        }
        for column in self.columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="w", stretch=True)

        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Cancel", command=self.cancel).pack(side="right")
        ttk.Button(actions, text=action_label, command=self.confirm).pack(
            side="right",
            padx=(0, 8),
        )

        self.tree.tag_configure("missing", foreground="#b00020")
        self.search_var.trace_add("write", self._on_filter_change)
        self.tree.bind("<Double-1>", lambda _event: self.confirm())
        self.window.bind("<Return>", lambda _event: self.confirm())
        self.window.bind("<Escape>", lambda _event: self.cancel())
        self._populate(rows)

    def _populate(self, rows: list[LayoutRow]) -> None:
        self.tree.delete(*self.tree.get_children())
        self.filtered_rows = rows
        for row in rows:
            tags = ("missing",) if row.missing else ()
            self.tree.insert(
                "",
                "end",
                iid=str(row.layout_id),
                values=row.values(),
                tags=tags,
            )

        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])

    def _on_filter_change(self, *_args) -> None:
        self._populate(filter_layout_rows(self.rows, self.search_var.get()))

    def confirm(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo(
                "Select layout",
                "Select a layout first.",
                parent=self.window,
            )
            return
        layout_id = int(selection[0])
        row = next(row for row in self.rows if row.layout_id == layout_id)
        self.selected = (row.layout_id, row.filename)
        self.window.destroy()

    def cancel(self) -> None:
        self.selected = None
        self.window.destroy()

    def show(self) -> tuple[int, str] | None:
        """Show the selector and return the selected layout ID and filename."""
        self.window.wait_window()
        return self.selected


class LayoutEditorLauncher:
    """Small Tkinter launcher for common layout-mask editor workflows."""

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("WHL Layout Editor")
        self.root.geometry("340x400")
        self.editor_backend = tk.StringVar(value="Tkinter canvas")
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="WHL Layout Editor").pack(pady=(0, 14))
        ttk.Label(frame, text="Editor").pack(anchor="w")
        ttk.Combobox(
            frame,
            textvariable=self.editor_backend,
            values=tuple(EDITOR_BACKENDS),
            state="readonly",
        ).pack(fill="x", pady=(0, 12))
        ttk.Button(
            frame,
            text="Create new layout",
            command=self.create_new_layout,
        ).pack(fill="x", pady=4)
        ttk.Button(
            frame,
            text="Duplicate existing layout",
            command=self.duplicate_existing_layout,
        ).pack(fill="x", pady=4)
        ttk.Button(
            frame,
            text="Open/edit existing layout",
            command=self.open_existing_layout,
        ).pack(fill="x", pady=4)
        ttk.Button(
            frame,
            text="Delete layout",
            command=self.delete_existing_layout,
        ).pack(fill="x", pady=4)
        ttk.Button(
            frame,
            text="Preview layout",
            command=self.preview_layout,
        ).pack(fill="x", pady=4)
        ttk.Button(
            frame,
            text="Exit",
            command=self.root.destroy,
        ).pack(
            fill="x",
            pady=(16, 0),
        )

    def _editor_class(self):
        """Return the selected editor backend class."""
        return EDITOR_BACKENDS[self.editor_backend.get()]

    def _open_editor(self, masks: dict, name: str, path: Path):
        """Open the selected editor backend with a shared constructor contract."""
        editor_class = self._editor_class()
        kwargs = {
            "masks": masks,
            "name": name,
            "save_path": path,
        }
        if editor_class is TkinterGridEditor:
            kwargs["master"] = self.root
        editor = editor_class(**kwargs)
        editor.show()
        return editor

    def _ask_new_layout_name(self, title: str, initialvalue: str = "") -> str | None:
        """Ask for a non-empty, unused layout name."""
        while True:
            name = simpledialog.askstring(
                title,
                "New layout name:",
                initialvalue=initialvalue,
                parent=self.root,
            )
            if name is None:
                return None

            clean_name = name.strip()
            if not clean_name:
                messagebox.showwarning(title, "Layout name must not be empty.")
                continue
            if not layout_filename_available(clean_name):
                messagebox.showwarning(
                    title,
                    f"{_npz_filename(clean_name)} already exists. Choose another name.",
                )
                continue
            return clean_name

    def _select_layout(self, title: str, action_label: str) -> tuple[int, str] | None:
        layouts = list_layouts()
        if not layouts:
            messagebox.showinfo(title, "No layouts are registered yet.")
            return None

        rows = layout_rows_from_registry(layouts)
        return LayoutSelectorDialog(self.root, rows, title, action_label).show()

    def create_new_layout(self) -> None:
        rows = simpledialog.askinteger("Create layout", "Rows:", minvalue=1)
        if rows is None:
            return
        cols = simpledialog.askinteger("Create layout", "Columns:", minvalue=1)
        if cols is None:
            return
        aisle_width = simpledialog.askinteger(
            "Create layout",
            "Aisle width:",
            initialvalue=2,
            minvalue=1,
        )
        if aisle_width is None:
            return
        name = simpledialog.askstring("Create layout", "Layout name:")
        if not name:
            return

        filename = _npz_filename(name)
        path = _mask_path(filename)
        if path.exists() and not messagebox.askyesno(
            "Overwrite layout?",
            f"{filename} already exists. Open it for overwrite?",
        ):
            return

        masks = empty_mask_bundle(rows, cols, aisle_width, Path(filename).stem)
        editor = self._open_editor(masks, Path(filename).stem, path)

        if editor.saved_path is not None:
            layouts = list_layouts()
            if filename not in layouts.values():
                add_layout(filename)
            messagebox.showinfo("Saved", f"Saved {editor.saved_path}")

    def duplicate_existing_layout(self) -> None:
        selected = self._select_layout("Duplicate layout", "Duplicate")
        if selected is None:
            return

        _, source_filename = selected
        source_path = _mask_path(source_filename)
        if not source_path.exists():
            messagebox.showerror(
                "Duplicate layout",
                f"Source mask file not found:\n{source_path}",
            )
            return

        new_name = self._ask_new_layout_name(
            "Duplicate layout",
            initialvalue=f"{Path(source_filename).stem}_copy",
        )
        if new_name is None:
            return

        target_filename = _npz_filename(new_name)
        target_path = _mask_path(target_filename)
        source_masks = load_mask(source_path)
        duplicate_masks = prepare_duplicate_masks(source_masks, target_filename)

        editor = self._open_editor(
            duplicate_masks,
            Path(target_filename).stem,
            target_path,
        )
        if editor.saved_path is not None:
            layouts = list_layouts()
            if target_filename not in layouts.values():
                add_layout(target_filename)
            messagebox.showinfo(
                "Duplicated",
                f"Created {target_filename} from {source_filename}",
            )

    def open_existing_layout(self) -> None:
        selected = self._select_layout("Open/edit layout", "Open")
        if selected is None:
            return
        _, filename = selected
        path = _mask_path(filename)
        if not path.exists():
            messagebox.showerror("Open/edit layout", f"Mask file not found:\n{path}")
            return

        masks = load_mask(path)
        self._open_editor(masks, Path(filename).stem, path)

    def delete_existing_layout(self) -> None:
        selected = self._select_layout("Delete layout", "Delete")
        if selected is None:
            return
        layout_id, filename = selected
        if not messagebox.askyesno("Delete layout", f"Delete {filename}?"):
            return

        removed = delete_layout(layout_id, reindex=True)
        path = _mask_path(filename)
        if path.exists():
            path.unlink()

        if removed is None:
            messagebox.showwarning("Delete layout", "Registry entry was not found.")
        else:
            messagebox.showinfo("Delete layout", f"Deleted {filename}")

    def preview_layout(self) -> None:
        selected = self._select_layout("Preview layout", "Preview")
        if selected is None:
            return
        _, filename = selected
        path = _mask_path(filename)
        if not path.exists():
            messagebox.showerror("Preview layout", f"Mask file not found:\n{path}")
            return

        masks = load_mask(path)
        grid = mask_to_grid(masks)
        title = (
            f"{Path(filename).stem} | aisle_width={masks['aisle_width']} | "
            f"{masks['rows']}x{masks['cols']}"
        )
        plot_layout_grid(grid, title=title, show_coords=True)


def main() -> None:
    """Run the Tkinter layout editor launcher."""
    root = Tk()
    LayoutEditorLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()


__all__ = [
    "EDITOR_BACKENDS",
    "LayoutEditorLauncher",
    "LayoutRow",
    "LayoutSelectorDialog",
    "filter_layout_rows",
    "layout_filename_available",
    "layout_row_from_registry_entry",
    "layout_rows_from_registry",
    "main",
    "prepare_duplicate_masks",
]
