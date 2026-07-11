#!/usr/bin/env python3
"""List tasks in management/docs/tasks.md (one or all sections).

Usage:
    python3 list_tasks.py                  # all sections
    python3 list_tasks.py --section 进行中   # one section
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mgmt_io
import task_schema


def print_section(lines, section: str) -> int:
    rows = mgmt_io.list_rows(lines, section)
    cols = task_schema.SECTION_COLS[section]
    print(f"\n## {section}  ({len(rows)} task{'s' if len(rows) != 1 else ''})")
    if not rows:
        print("  (empty)")
        return 0
    print("  " + " | ".join(cols))
    for r in rows:
        print("  " + " | ".join(r.get(c, "") for c in cols))
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", default=None, help="only list this section (default: all)")
    args = ap.parse_args()

    path = mgmt_io.tasks_md()
    lines = mgmt_io.read_lines(path)
    sections = [task_schema.resolve_section(args.section)] if args.section else task_schema.SECTION_ORDER
    total = 0
    for s in sections:
        total += print_section(lines, s)
    print(f"\n{total} task(s) total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
