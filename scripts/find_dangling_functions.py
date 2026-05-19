#!/usr/bin/env python3
"""Find likely dangling functions with a conservative AST scan.

This script is intentionally heuristic. It reports candidates for manual review;
do not blindly delete everything it prints.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOTS = ["SimpleLLMFunc"]
SKIP_DIRS = {".git", ".venv", "__pycache__", "dist", "build", ".pytest_cache"}
HOOK_NAMES = {
    "on_run_start",
    "before_llm_call",
    "after_llm_call",
    "before_tool_batch",
    "after_tool_batch",
    "before_finalize",
    "collect_context_mutations",
    "finalize",
}
TUI_NAMES = {
    "compose",
    "on_mount",
    "on_input_submitted",
    "on_key",
    "watch_",
}
COMMON_PROTOCOL_NAMES = {
    "__init__",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    "__iter__",
    "__next__",
    "__call__",
    "__str__",
    "__repr__",
    "__len__",
    "__bool__",
    "__getitem__",
    "__setitem__",
    "__contains__",
}
DECORATOR_ALLOWLIST = {
    "property",
    "staticmethod",
    "classmethod",
    "abstractmethod",
    "tool",
    "llm_function",
    "llm_chat",
}


@dataclass(frozen=True)
class FunctionInfo:
    module: str
    qualname: str
    name: str
    path: Path
    lineno: int
    is_method: bool
    decorators: tuple[str, ...]
    exported: bool = False


class ModuleVisitor(ast.NodeVisitor):
    def __init__(self, module: str, path: Path) -> None:
        self.module = module
        self.path = path
        self.stack: list[str] = []
        self.functions: list[FunctionInfo] = []
        self.exports: set[str] = set()
        self.name_refs: dict[str, int] = {}
        self.name_calls: dict[str, int] = {}
        self.attr_refs: dict[str, int] = {}
        self.attr_calls: dict[str, int] = {}
        self.imported_names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qual_parts = [*self.stack, node.name]
        decorators = tuple(_decorator_name(item) for item in node.decorator_list)
        self.functions.append(
            FunctionInfo(
                module=self.module,
                qualname=".".join(qual_parts),
                name=node.name,
                path=self.path,
                lineno=node.lineno,
                is_method=bool(self.stack),
                decorators=decorators,
            )
        )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Name(self, node: ast.Name) -> None:
        self.name_refs[node.id] = self.name_refs.get(node.id, 0) + 1

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.attr_refs[node.attr] = self.attr_refs.get(node.attr, 0) + 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            self.name_calls[func.id] = self.name_calls.get(func.id, 0) + 1
        elif isinstance(func, ast.Attribute):
            self.attr_calls[func.attr] = self.attr_calls.get(func.attr, 0) + 1
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imported_names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imported_names.add(alias.asname or alias.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                self.exports.update(_literal_string_list(node.value))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.target.id == "__all__" and node.value is not None:
            self.exports.update(_literal_string_list(node.value))
        self.generic_visit(node)


@dataclass
class ScanResult:
    functions: list[FunctionInfo]
    exported_names: set[str]
    name_refs: dict[str, int]
    name_calls: dict[str, int]
    attr_refs: dict[str, int]
    attr_calls: dict[str, int]
    imported_names: set[str]


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return "<unknown>"


def _literal_string_list(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set()
    values: set[str] = set()
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.add(item.value)
    return values


def _module_name(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def _iter_py_files(roots: Iterable[str]) -> Iterable[Path]:
    for root_str in roots:
        root = Path(root_str)
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        for path in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            yield path


def scan(roots: Iterable[str]) -> ScanResult:
    functions: list[FunctionInfo] = []
    exported_names: set[str] = set()
    name_refs: dict[str, int] = {}
    name_calls: dict[str, int] = {}
    attr_refs: dict[str, int] = {}
    attr_calls: dict[str, int] = {}
    imported_names: set[str] = set()

    for path in _iter_py_files(roots):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            print(f"PARSE_ERROR {path}: {exc}")
            continue
        visitor = ModuleVisitor(_module_name(path), path)
        visitor.visit(tree)
        functions.extend(visitor.functions)
        exported_names.update(visitor.exports)
        imported_names.update(visitor.imported_names)
        _merge_counts(name_refs, visitor.name_refs)
        _merge_counts(name_calls, visitor.name_calls)
        _merge_counts(attr_refs, visitor.attr_refs)
        _merge_counts(attr_calls, visitor.attr_calls)

    functions = [
        FunctionInfo(
            module=f.module,
            qualname=f.qualname,
            name=f.name,
            path=f.path,
            lineno=f.lineno,
            is_method=f.is_method,
            decorators=f.decorators,
            exported=f.name in exported_names,
        )
        for f in functions
    ]
    return ScanResult(
        functions=functions,
        exported_names=exported_names,
        name_refs=name_refs,
        name_calls=name_calls,
        attr_refs=attr_refs,
        attr_calls=attr_calls,
        imported_names=imported_names,
    )


def _merge_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def classify(info: FunctionInfo, result: ScanResult) -> tuple[str, str]:
    name = info.name
    if name in COMMON_PROTOCOL_NAMES or name.startswith("__"):
        return "do-not-remove", "protocol/dunder"
    if info.exported:
        return "do-not-remove", "exported in __all__"
    if name in result.imported_names:
        return "manual-review", "imported somewhere"
    if any(dec in DECORATOR_ALLOWLIST for dec in info.decorators):
        return "manual-review", f"decorated: {','.join(info.decorators)}"
    if name in HOOK_NAMES:
        return "manual-review", "hook method name"
    if name in TUI_NAMES or any(name.startswith(prefix) for prefix in TUI_NAMES if prefix.endswith("_")):
        return "manual-review", "TUI/lifecycle-like method"
    if name.startswith("test_"):
        return "do-not-remove", "test function"

    call_count = result.name_calls.get(name, 0) + result.attr_calls.get(name, 0)
    ref_count = result.name_refs.get(name, 0) + result.attr_refs.get(name, 0)

    if call_count == 0 and ref_count == 0:
        if info.is_method and not name.startswith("_"):
            return "manual-review", "public method with no static references"
        return "safe-candidate", "no static refs/calls"
    if call_count == 0 and ref_count <= 1 and name.startswith("_"):
        return "manual-review", f"private with low refs/calls refs={ref_count}"
    return "used", f"calls={call_count} refs={ref_count}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument(
        "--category",
        choices=["safe-candidate", "manual-review", "do-not-remove", "used", "all"],
        default="safe-candidate",
    )
    args = parser.parse_args()

    result = scan(args.roots)
    rows = []
    for info in result.functions:
        category, reason = classify(info, result)
        if args.category != "all" and category != args.category:
            continue
        rows.append((category, str(info.path), info.lineno, info.qualname, reason))

    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    print(
        f"scanned_functions={len(result.functions)} exported_names={len(result.exported_names)} category={args.category}"
    )
    for category, path, lineno, qualname, reason in rows:
        print(f"{category}\t{path}:{lineno}\t{qualname}\t{reason}")


if __name__ == "__main__":
    main()
