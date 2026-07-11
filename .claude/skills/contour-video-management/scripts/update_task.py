#!/usr/bin/env python3
"""Update a task row (or move it between sections) in management/docs/tasks.md.

Only fields passed on the CLI are changed; omitted fields keep their current
value. ``--move-to`` relocates the row to another section, remapping columns
(任务/负责人 carry over; moving to 已完成 defaults 完成日期 to today, 产出 to 备注).

Usage:
    # change status / owner
    python3 update_task.py --section 进行中 --name "轮廓提取优化" --status 🔴 --owner 李四
    # rename
    python3 update_task.py --section 进行中 --name "轮廓提取优化" --new-name "轮廓提取 v2"
    # mark done (move 进行中 -> 已完成)
    python3 update_task.py --section 进行中 --name "轮廓提取优化" --move-to 已完成 --output "PR#12"
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
    ap.add_argument("--section", required=True, help="section the task is currently in")
    ap.add_argument("--name", required=True, help="current task name (first cell, exact match)")
    ap.add_argument("--new-name", default=None, help="rename the task")
    ap.add_argument("--owner", default=None)
    ap.add_argument("--start", default=None, help="开始日期 / 预计开始")
    ap.add_argument("--end", default=None, help="截止日期")
    ap.add_argument("--status", default=None, help="状态 emoji")
    ap.add_argument("--priority", default=None, help="优先级")
    ap.add_argument("--note", default=None, help="备注")
    ap.add_argument("--complete-date", default=None, help="完成日期 (已完成)")
    ap.add_argument("--output", default=None, help="产出 (已完成)")
    ap.add_argument("--move-to", default=None, help="move task to another section (进行中/待开始/已完成)")
    args = ap.parse_args()

    section = task_schema.resolve_section(args.section)
    path = mgmt_io.tasks_md()
    lines = mgmt_io.read_lines(path)

    if args.move_to:
        to_section = task_schema.resolve_section(args.move_to)
        remap: dict[str, str] = {}
        if args.new_name:
            remap["任务"] = args.new_name
        if args.owner:
            remap["负责人"] = args.owner
        if args.complete_date:
            remap["完成日期"] = args.complete_date
        if args.output:
            remap["产出"] = args.output
        new_lines = mgmt_io.move_row(lines, section, args.name, to_section, remap)
        mgmt_io.write_lines(path, new_lines)
        print(f"✓ moved task {args.name!r}: {section} -> {to_section}  ({mgmt_io.rel(path)})")
        return 0

    # collect only explicitly-passed fields -> column updates
    updates: dict[str, str] = {}
    if args.new_name is not None:
        updates["任务"] = args.new_name
    if args.owner is not None:
        updates["负责人"] = args.owner
    if args.start is not None:
        updates["开始日期"] = args.start
        updates["预计开始"] = args.start
    if args.end is not None:
        updates["截止日期"] = args.end
    if args.status is not None:
        updates["状态"] = args.status
    if args.priority is not None:
        updates["优先级"] = args.priority
    if args.note is not None:
        updates["备注"] = args.note
    if args.complete_date is not None:
        updates["完成日期"] = args.complete_date
    if args.output is not None:
        updates["产出"] = args.output

    # filter updates to columns that exist in this section (silently skip others)
    cols = task_schema.SECTION_COLS[section]
    safe_updates = {c: v for c, v in updates.items() if c in cols}
    if not safe_updates:
        print("no updates given (nothing matched this section's columns); nothing to do.", file=sys.stderr)
        return 1

    new_lines = mgmt_io.update_row(lines, section, args.name, safe_updates)
    mgmt_io.write_lines(path, new_lines)
    print(f"✓ updated task {args.name!r} in {section}  ({mgmt_io.rel(path)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
