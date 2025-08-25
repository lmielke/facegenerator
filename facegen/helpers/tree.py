# tree.py
"""
WHY: Minimal, consistent tree builder that honors sts.* settings:
- ignore_dirs (with globs), abrev_dirs, ignore_files (verbosity thresholds).
- Collects matched files and (optionally) dumps contents by verbosity.
- Colorization is optional and isolated.
"""

import os, re, fnmatch, contextlib  # stdlib only, fast imports
from typing import Set, List, Tuple, Dict, Iterable, Optional
from colorama import Fore, Style

import facegen.settings as sts


try:
    # Optional helper for atomic chdir during mk_dirs_hierarchy
    from facegen.helpers.collections import temp_chdir as _temp_chdir
except Exception:
    @contextlib.contextmanager
    def _temp_chdir(path: str):
        """
        WHY: Fallback chdir ctx mgr; used only if facegen helper is absent.
        """
        cwd = os.getcwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(cwd)


styles_dict: Dict[str, Dict[str, Dict[str, object]]] = {
    "default": {
        "dir":  {"sym": "|--",  "col": f"{Fore.WHITE}"},
        "file": {"sym": "|-",   "col": f"{Fore.WHITE}"},
        "fold": {"sym": "▼",    "col": f"{Fore.YELLOW}"},
        "disc": {"sym": "|...", "col": f"{Style.DIM}{Fore.WHITE}"},
        "ext":  {"sym": [".py"], "col": [f"{Fore.BLUE}"]},
    },
}


class Tree:
    file_types: Dict[str, str] = {
        ".py": "Python", ".yml": "YAML", ".yaml": "YAML",
        ".md": "Markdown", ".txt": "Text",
    }

    def __init__(self, *args, style: str = "default", **kwargs):
        """
        WHY: One Tree to render hierarchy, collect matches, and read content.
        """
        self._apply_style(*args, style=style, **kwargs)
        self.indent = "    "
        self.matched_files: List[str] = []
        self.verbose = self.handle_verbosity(*args, **kwargs)
        self._out: Optional[List[str]] = None

    def __call__(self, *args, **kwargs) -> dict:
        """
        WHY: Convenience wrapper that returns a dict payload.
        """
        tree, contents = self.mk_tree(*args, **kwargs)
        selected = self.load_matched_files(*args, **kwargs)
        return {
            "tree": tree,
            "contents": contents,
            "file_matches": list(self.matched_files),
            "selected_files": selected,
        }

    # --- verbosity / style --------------------------------------------------

    def handle_verbosity(self, *args, verbose: int = 0, **kwargs) -> int:
        """
        WHY: Verbosity controls content-dump depth; confirm very large dumps.
        """
        if verbose >= 7:
            print(f"{Fore.YELLOW}WARNING: {verbose = } "
                  f"Output might excede the console length!{Style.RESET_ALL}"
                  f"{Style.RESET_ALL}")
            cont = input("Continue with a selected verbosity? "
                         "[1, 2, ..., ENTER keeps current]: ")
            if cont.isdigit():
                return int(cont)
        return verbose

    def _apply_style(self, *args, style: str, **kwargs) -> None:
        st = styles_dict.get(style, styles_dict["default"])
        for name, style_map in st.items():
            for k, v in style_map.items():
                setattr(self, f"{name}_{k}", v)

    # --- public API ---------------------------------------------------------

    def mk_tree(
        self,
        *args,
        project_dir: str | None = None,
        max_depth: int | None = 6,
        ignores: set[str] | None = None,
        file_match_regex: str | None = None,
        work_file_name: str | None = None,
        colorized: bool = False,
        **kwargs,
    ) -> tuple[str, str]:
        """
        WHY: Walk project_dir, honoring sts.*; return (tree, contents).
        """
        if project_dir is None and args and isinstance(args[0], str):
            project_dir = args[0]
        prj = project_dir or getattr(sts, "project_dir", os.getcwd())
        ign = set(ignores) if ignores else set(getattr(sts, "ignore_dirs", set()))
        self.matched_files.clear()

        tree, contents = ["<hierarchy>"], ["<file_contents>"]
        self._out = tree
        base_level = prj.count(os.sep)

        for root, dirs, files in os.walk(prj, topdown=True):
            level = root.count(os.sep) - base_level
            subdir = os.path.basename(root)
            ind = self.indent * level

            if self._is_ignored(subdir, ign):
                tree.append(f"{ind}{self.disc_sym} {self.fold_sym} {subdir}")
                dirs[:] = []
                continue

            if max_depth is not None and level >= max_depth:
                tree.append(f"{ind}{self.disc_sym} {self.fold_sym} {subdir}")
                dirs[:] = []
                continue

            tree.append(f"{ind}{self.dir_sym}{self.fold_sym} {subdir}")
            self._emit_files(
                *args, root=root, files=files, ind=ind, level=level,
                file_match_regex=file_match_regex, contents=contents, **kwargs,
            )

        tree.append("</hierarchy>")
        contents.append("</file_contents>")

        if work_file_name:
            self._promote_workfile(*args, work_file_name=work_file_name, **kwargs)

        t = "\n".join(tree)
        if colorized:
            t = self._colorize(t, *args, **kwargs)
        return t, "\n".join(contents)

    # --- emit / matches / contents -----------------------------------------

    def _emit_files(
        self,
        *args,
        root: str,
        files: Iterable[str],
        ind: str,
        level: int,
        file_match_regex: Optional[str],
        contents: List[str],
        **kwargs,
    ) -> None:
        log_dir = self._is_abbrev_dir(root, *args, **kwargs)
        listed = 0
        for f in files:
            if log_dir and listed >= 1:
                self._line(f"{ind}{self.indent}{self.disc_sym}", *args, **kwargs)
                break
            self._line(f"{ind}{self.indent}{self.file_sym} {f}", *args, **kwargs)
            full = os.path.join(root, f)

            if file_match_regex and re.search(file_match_regex, f):
                self._track_match(*args, path=full, **kwargs)

            if self.verbose > level:
                if self._ignored_file(f, *args, **kwargs):
                    listed += 1
                    continue
                fc = self.load_file_content(*args, file_path=full, **kwargs)
                contents.append(
                    f"{Fore.CYAN}\n<file name='{f}' path='{full}'>{Fore.RESET}\n{fc}"
                )
            listed += 1

    def _track_match(self, *args, path: str | None = None, **kwargs) -> None:
        if path and path not in self.matched_files:
            self.matched_files.append(path)

    def _promote_workfile(self, *args, work_file_name: str, **kwargs) -> None:
        base = work_file_name
        idx = next(
            (i for i, p in enumerate(self.matched_files)
             if os.path.splitext(os.path.basename(p))[0] == base),
            None,
        )
        if idx is None:
            return
        p = self.matched_files.pop(idx)
        self.matched_files.insert(0, p)

    def load_matched_files(
        self,
        *args,
        default_ignore_files: Iterable[str] | None = None,
        **kwargs,
    ) -> List[dict]:
        """
        WHY: Load content for matched files; optional path-prefix filter.
        """
        sel: List[dict] = []
        prefixes = tuple(default_ignore_files or ())
        for p in self.matched_files:
            if prefixes and p.startswith(prefixes):
                continue
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    c = fh.read()
            except UnicodeDecodeError:
                print(f"{Fore.RED}Error reading file: {p}{Fore.RESET}")
                continue
            ext = os.path.splitext(p)[1]
            ftype = self.file_types.get(ext, "Text")
            sel.append({"file_path": p, "file_type": ftype, "file_content": c})
        return sel

    # --- helpers: ignore / abbrev / IO -------------------------------------

    def _is_ignored(self, subdir: str, ignores: Set[str], *args, **kwargs) -> bool:
        """
        WHY: Support exact matches, suffix-like, and glob patterns.
        """
        return any(
            subdir == pat
            or subdir.endswith(pat.lstrip("*"))
            or fnmatch.fnmatch(subdir, pat)
            for pat in ignores
        )

    def _ignored_file(self, fname: str, *args, **kwargs) -> bool:
        """
        WHY: Hide technical files until verbosity reaches required level.
        Rule: if self.verbose < level and any(pattern in name) -> ignore.
        Case-insensitive for robustness.
        """
        rules: dict[int, set[str]] = getattr(sts, "ignore_files", {})
        f = fname.casefold()
        for min_show_level, pats in rules.items():
            if self.verbose < min_show_level and any(p.casefold() in f for p in pats):
                return True
        return False

    def _is_abbrev_dir(self, root: str, *args, **kwargs) -> bool:
        """
        WHY: Abbreviate directory listing if leaf name is in sts.abrev_dirs.
        """
        sub = os.path.basename(root)
        abr = set(getattr(sts, "abrev_dirs", set()))
        return sub in abr

    def load_file_content(self, *args, file_path: str, **kwargs) -> str:
        """
        WHY: Read file as text, suppress noisy errors unless verbose>=1.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            if self.verbose >= 1:
                print(f"{Fore.RED}Read error:{Fore.RESET} {e}")
            return ""

    def _line(self, s: str, *args, **kwargs) -> None:
        """
        WHY: Append a rendered line to the active tree buffer.
        """
        buf = getattr(self, "_out", None)
        if buf is not None:
            buf.append(s)

    # --- color / normalize / parse / mk-dirs --------------------------------

    def _colorize(self, tree: str, *args, style: str = "default", **kwargs) -> str:
        """
        WHY: Inject ANSI colors based on style symbols and file extensions.
        """
        st = styles_dict.get(style, styles_dict["default"])
        out: List[str] = []
        for line in tree.split("\n"):
            for name, m in st.items():
                sym, col = m.values()
                if name == "ext":
                    for sfx, c in zip(sym, col):
                        line = re.sub(
                            rf"(\S*{re.escape(sfx)})",
                            f"{c}\\1{Style.RESET_ALL}",
                            line,
                        )
                else:
                    line = line.replace(sym, f"{col}{sym}{Style.RESET_ALL}")
            out.append(line)
        return "\n".join(out).strip()

    def uncolorize(self, tree: str, *args, **kwargs) -> str:
        """
        WHY: Strip basic Colorama sequences from a rendered tree.
        """
        cols = [
            Fore.YELLOW, Fore.GREEN, Fore.RED, Fore.BLUE, Fore.CYAN,
            Fore.WHITE, Style.RESET_ALL, Style.DIM, Style.RESET_ALL,
        ]
        out = []
        for line in tree.split("\n"):
            for c in cols:
                line = line.replace(c, "")
            out.append(line)
        return "\n".join(out).strip()

    def _normalize_tree(self, tree: str, *args, **kwargs) -> str:
        """
        WHY: Left-trim to minimal indent so parse_tree works on pasted trees.
        """
        lines = tree.split("\n")
        if not lines:
            return ""
        pad = self.indent
        levels = [len(l) - len(l.lstrip(pad)) for l in lines if l.strip()]
        min_ind = min(levels) if levels else 0
        norm = [l[min_ind:] if l.strip() else l for l in lines]
        return "\n".join(norm).strip()

    def _cleanup_line(self, line: str, *args, **kwargs) -> Tuple[str, bool]:
        """
        WHY: Remove style symbols; return (text, is_dir).
        """
        line = line.strip()
        is_dir = line.startswith(self.dir_sym)
        for s in (self.disc_sym, self.dir_sym, self.file_sym, self.fold_sym):
            line = line.replace(s, "")
        return line.strip(), is_dir

    def parse_tree(self, tree: str, *args, **kwargs) -> List[Tuple[str, bool]]:
        """
        WHY: Convert textual tree into (path, is_dir) tuples for mk_dirs_hierarchy.
        """
        paths: List[Tuple[str, bool]] = []
        temps: List[str] = []
        norm = self._normalize_tree(tree, *args, **kwargs)
        for line in norm.split("\n"):
            if not line or line.startswith(self.disc_sym) or line.startswith("<"):
                continue
            level = line.count(self.indent)
            txt, is_dir = self._cleanup_line(line, *args, **kwargs)
            if not txt or txt == self.disc_sym:
                continue
            if len(temps) < level:
                temps.append(txt)
            else:
                temps = temps[:level] + [txt]
            paths.append((os.path.join(*temps), is_dir))
        return paths

    def mk_dirs_hierarchy(
        self,
        dirs: Iterable[Tuple[str, bool]],
        tgt_path: str,
        *args,
        **kwargs,
    ) -> Optional[str]:
        """
        WHY: Materialize (path, is_dir) tuples under tgt_path; return first dir.
        """
        start_dir = None
        with _temp_chdir(tgt_path):
            for path, is_dir in dirs:
                if os.path.exists(path):
                    continue
                if is_dir:
                    os.makedirs(path)
                    start_dir = start_dir or os.path.join(tgt_path, path)
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(f"#{os.path.basename(path)}\n\n")
        return start_dir
