#!/usr/bin/env python3
"""Add a task row to management/docs/tasks.md.

Operates on the THREE-section board (进行中 / 待开始 / 已完成). The section
determines which columns the row gets; friendly flags map to the right column
for the chosen section (e.g. --start -> 开始日期 for 进行中, 预计开始 for 待开始).

Self-locating: run from anywhere; resolves the repo root from its own path.
Mirrors the same file in ~/ProjFlow (identical tasks.md schema).

Usage:
    python3 add_task.py --section 进行中 --name "轮廓提取优化" --owner 张三 \\
        --start 2026-07-11 --end 2026-07-18 --status 🟢 --note "sobel 降噪"
    python3 add_task.py --section 待开始 --name "AV1 baseline" --priority P1
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running as `python3 add_task.py` from anywhere: put this script's dir
# (which holds mgmt_io.py / task_schema.py) on sys.path before importing them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mgmt_io
import task_schema


def build_values(section: str, a: argparse.Namespace) -> dict[str, str]:
    cols = task_schema.SECTION_COLS[section]
    flag_map = {
        "任务": a.name,
        "负责人": a.owner,
        "开始日期": a.start,
        "预计开始": a.start,
        "截止日期": a.end,
        "完成日期": a.complete_date or mgmt_io.TODAY,
        "状态": a.status,
        "优先级": a.priority,
        "产出": a.output,
        "备注": a.note,
    }
    return {c: flag_map.get(c, "") for c in cols}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", required=True, help="进行中 | 待开始 | 已完成 (or in_progress/pending/completed)")
    ap.add_argument("--name", required=True, help="任务名 (first column)")
    ap.add_argument("--owner", default="", help="负责人")
    ap.add_argument("--start", default="", help="开始日期 / 预计开始 (YYYY-MM-DD)")
    ap.add_argument("--end", default="", help="截止日期 (YYYY-MM-DD)")
    ap.add_argument("--status", default="🟢", help="状态 emoji (进行中; default 🟢)")
    ap.add_argument("--priority", default="", help="优先级 (待开始, e.g. P1)")
    ap.add_argument("--note", default="", help="备注")
    ap.add_argument("--complete-date", default="", help="完成日期 (已完成; default today)")
    ap.add_argument("--output", default="", help="产出 (已完成)")
    args = ap.parse_args()

    section = task_schema.resolve_section(args.section)
    if section == "已完成" and not args.complete_date and not args.start:
        # ok — defaults to today
        pass
    values = build_values(section, args)
    path = mgmt_io.tasks_md()
    lines = mgmt_io.read_lines(path)
    new_lines = mgmt_io.add_row(lines, section, values)
    mgmt_io.write_lines(path, new_lines)
    print(f"✓ added task {args.name!r} -> {section}  ({mgmt_io.rel(path)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
