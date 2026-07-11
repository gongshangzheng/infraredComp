#!/usr/bin/env python3
"""Delete a task row from management/docs/tasks.md.

Usage:
    python3 delete_task.py --section 进行中 --name "轮廓提取优化"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mgmt_io
import task_schema


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", required=True, help="section the task is in (进行中/待开始/已完成)")
    ap.add_argument("--name", required=True, help="task name (first cell, exact match)")
    args = ap.parse_args()

    section = task_schema.resolve_section(args.section)
    path = mgmt_io.tasks_md()
    lines = mgmt_io.read_lines(path)
    new_lines = mgmt_io.delete_row(lines, section, args.name)
    mgmt_io.write_lines(path, new_lines)
    print(f"✓ deleted task {args.name!r} from {section}  ({mgmt_io.rel(path)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
