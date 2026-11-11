"""Publication gate: is this repository safe to make public?  Blockers fail (exit 1); warnings are for a person to read.

    uv run python scripts/publication_gate.py

Blockers: a machine path, a secret, a data or binary file, a large file, a private e-mail address.  Warnings: names of private sources and partners, references to private folders.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
SKIP_FILES = {"uv.lock", "publication_gate.py"}
DATA_EXT = {".parquet", ".feather", ".xlsx", ".xls", ".pdf", ".zip", ".las", ".dlis", ".segy", ".sgy", ".shp", ".gpkg", ".csv", ".png", ".jpg", ".db", ".sqlite"}
BLOCK = {"machine path": re.compile(r"[A-Za-z]:[\\/](?:appl|Users)[\\/]"), "secret": re.compile(r"ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY|api[_-]?key\s*[:=]\s*['\"][^'\"]{8,}"),
         "e-mail": re.compile(r"[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+\.[\w.]+")}
WARN = {"private source or partner": re.compile(r"\b(Volve|Equinor|Statoil|NSTA|Idiom)\b"), "private folder": re.compile(r"\b(personal|working)/|knowledge/(packs|requests)"), "Sodir data": re.compile(r"\b(Sodir|NPD)\b")}


def main() -> int:
    blockers: list[str] = []
    warnings: dict[str, list[str]] = {}
    for f in sorted(ROOT.rglob("*")):
        if not f.is_file() or set(f.relative_to(ROOT).parts) & SKIP_DIRS or f.name in SKIP_FILES:
            continue
        rel = f.relative_to(ROOT).as_posix()
        if f.suffix.lower() in DATA_EXT:
            blockers.append(f"{rel}: data or binary file ({f.suffix})")
            continue
        if f.stat().st_size > 400_000:
            blockers.append(f"{rel}: larger than 400 KB")
            continue
        try:
            text = f.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            blockers.append(f"{rel}: not text")
            continue
        for label, pattern in BLOCK.items():
            for m in pattern.finditer(text):
                blockers.append(f"{rel}: {label}: {m.group(0)[:60]!r}")
                break
        for label, pattern in WARN.items():
            n = len(pattern.findall(text))
            if n:
                warnings.setdefault(label, []).append(f"{rel} ({n})")
    print(f"publication gate: {len(blockers)} blockers")
    for b in blockers[:60]:
        print("  BLOCKER", b)
    for label, files in warnings.items():
        print(f"  warning, {label}: {len(files)} files, e.g. {', '.join(files[:4])}")
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
