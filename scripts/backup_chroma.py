"""
backup_chroma.py — Back up and restore the ChromaDB vector store.

WHY THIS EXISTS
---------------
Re-indexing from scratch burns Cohere API quota (15 084 chunks × ~350 tokens
= ~5.3M tokens, which exhausts a free trial).  Before any chunker or indexer
change that would invalidate the existing DB, run:

    python scripts/backup_chroma.py backup

This copies the raw ChromaDB files to data/chroma_db_backup_<timestamp>/.
To roll back:

    python scripts/backup_chroma.py restore [--from data/chroma_db_backup_20260528_120000]

USAGE
-----
    # Create a timestamped backup
    python scripts/backup_chroma.py backup

    # Restore from the most recent backup (auto-detected)
    python scripts/backup_chroma.py restore

    # Restore from a specific backup
    python scripts/backup_chroma.py restore --from data/chroma_db_backup_20260528_120000

    # List available backups
    python scripts/backup_chroma.py list

    # Verify a backup is intact (chunk count matches live DB)
    python scripts/backup_chroma.py verify

WHAT IS BACKED UP
-----------------
ChromaDB persists to a directory of SQLite + HNSW index files.
The entire directory is copied with shutil.copytree — no special
serialisation needed.  The backup is a self-contained ChromaDB instance
you can open directly with chromadb.PersistentClient(path=backup_dir).
"""

import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime


# ── Paths ────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    """Return the repo root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def _live_db_path() -> Path:
    # Try to read from config first; fall back to the default
    try:
        sys.path.insert(0, str(_project_root()))
        from src.config import settings
        return Path(settings.chroma_db_path)
    except Exception:
        return _project_root() / "data" / "chroma_db"


def _backup_dir_for(timestamp: str) -> Path:
    return _project_root() / "data" / f"chroma_db_backup_{timestamp}"


def _list_backups() -> list[Path]:
    data_dir = _project_root() / "data"
    return sorted(data_dir.glob("chroma_db_backup_*"), reverse=True)


# ── Chunk-count verification ──────────────────────────────────────────────────

def _count_chunks(db_path: Path) -> int:
    """Open a ChromaDB at db_path and return total chunk count."""
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        col = client.get_collection("algorithms")
        return col.count()
    except Exception as e:
        return -1


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_backup(args) -> int:
    live = _live_db_path()
    if not live.exists():
        print(f"ERROR: live DB not found at {live}")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = _backup_dir_for(ts)

    print(f"Backing up {live} → {dest} …")
    shutil.copytree(live, dest)

    count = _count_chunks(dest)
    count_str = str(count) if count >= 0 else "unknown (open failed)"
    print(f"✅  Backup complete — {count_str} chunks — {dest}")
    return 0


def cmd_restore(args) -> int:
    live = _live_db_path()

    if args.source:
        src = Path(args.source)
    else:
        backups = _list_backups()
        if not backups:
            print("ERROR: No backups found in data/")
            return 1
        src = backups[0]
        print(f"Auto-selected most recent backup: {src}")

    if not src.exists():
        print(f"ERROR: Backup not found at {src}")
        return 1

    # Safety: rename live DB rather than delete, so you can undo the restore
    if live.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        displaced = live.parent / f"chroma_db_pre_restore_{ts}"
        print(f"Moving current DB → {displaced}")
        shutil.move(str(live), str(displaced))

    print(f"Restoring {src} → {live} …")
    shutil.copytree(src, live)

    count = _count_chunks(live)
    count_str = str(count) if count >= 0 else "unknown"
    print(f"✅  Restore complete — {count_str} chunks live at {live}")
    return 0


def cmd_list(args) -> int:
    backups = _list_backups()
    if not backups:
        print("No backups found.")
        return 0
    print(f"{'Backup path':<55}  {'chunks':>8}")
    print("-" * 66)
    for b in backups:
        count = _count_chunks(b)
        count_str = str(count) if count >= 0 else "?"
        print(f"{str(b):<55}  {count_str:>8}")
    return 0


def cmd_verify(args) -> int:
    live = _live_db_path()
    backups = _list_backups()

    live_count = _count_chunks(live)
    print(f"Live DB ({live}): {live_count} chunks")

    if not backups:
        print("No backups to compare against.")
        return 0

    latest = backups[0]
    bk_count = _count_chunks(latest)
    print(f"Latest backup ({latest.name}): {bk_count} chunks")

    if live_count == bk_count:
        print("✅  Counts match — backup looks intact.")
    else:
        print(f"⚠️   Counts differ ({live_count} vs {bk_count}) — backup may be stale or from a different index run.")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Back up and restore the AlgoRAG ChromaDB vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("backup",  help="Create a timestamped backup of the live DB")

    p_restore = sub.add_parser("restore", help="Restore from a backup")
    p_restore.add_argument(
        "--from", dest="source", metavar="PATH",
        help="Path to backup dir (default: most recent)",
    )

    sub.add_parser("list",   help="List available backups with chunk counts")
    sub.add_parser("verify", help="Compare live DB chunk count against latest backup")

    args = parser.parse_args()
    dispatch = {
        "backup":  cmd_backup,
        "restore": cmd_restore,
        "list":    cmd_list,
        "verify":  cmd_verify,
    }
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
