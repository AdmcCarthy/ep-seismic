"""This library imports only the standard library, its declared dependencies and the libraries it depends on: never the host repository (`src`) or a knowledge store."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "ep_seismic"
ALLOWED_LIBRARIES = ['ep_physics_base', 'ep_seismic']
FORBIDDEN = {"src", "knowledge", "ep_knowledge"}


def imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
    return out


def test_no_import_of_the_host_or_a_knowledge_store():
    for f in PACKAGE.rglob("*.py"):
        found = imports(f) & FORBIDDEN
        assert not found, f"{f.name} imports {found}"


def test_other_libraries_are_only_the_declared_ones():
    for f in PACKAGE.rglob("*.py"):
        libs = {m for m in imports(f) if m.startswith("ep_")}
        assert libs <= set(ALLOWED_LIBRARIES), f"{f.name} imports {libs - set(ALLOWED_LIBRARIES)}"
