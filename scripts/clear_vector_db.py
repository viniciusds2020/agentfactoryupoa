"""Clear all persisted vector collections (FAISS/chroma path) in one command.

Usage:
  python scripts/clear_vector_db.py --yes
  python scripts/clear_vector_db.py --path data/chroma --yes
  python scripts/clear_vector_db.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _default_vector_root() -> Path:
    # Keep it dependency-free and predictable for CLI use.
    return Path("data/chroma")


def _clear_root(root: Path, dry_run: bool) -> tuple[int, int]:
    deleted_dirs = 0
    deleted_files = 0

    if not root.exists():
        return deleted_dirs, deleted_files

    for entry in root.iterdir():
        if entry.is_dir():
            if not dry_run:
                shutil.rmtree(entry, ignore_errors=False)
            deleted_dirs += 1
        else:
            if not dry_run:
                entry.unlink(missing_ok=True)
            deleted_files += 1

    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)

    return deleted_dirs, deleted_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear all persisted vector collections.")
    parser.add_argument(
        "--path",
        default=str(_default_vector_root()),
        help="Vector DB root folder to clear (default: data/chroma).",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted.")
    args = parser.parse_args()

    root = Path(args.path)

    if not args.yes and not args.dry_run:
        print(f"About to delete ALL vector data under: {root.resolve()}")
        reply = input("Type 'YES' to continue: ").strip()
        if reply != "YES":
            print("Aborted.")
            return 1

    deleted_dirs, deleted_files = _clear_root(root, dry_run=args.dry_run)
    action = "Would delete" if args.dry_run else "Deleted"
    print(f"{action}: {deleted_dirs} directories, {deleted_files} files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

