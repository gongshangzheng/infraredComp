#!/usr/bin/env python3
"""Update a task node in management/docs/projects/{slug}/tasks.json.

Only fields passed on the CLI are changed; omitted fields keep their current
value. ``--status`` is the kanban "move between buckets" lever (status drives
which bucket a task appears in — completed/active/planned/paused/blocked).

Self-locating: run from anywhere; resolves the repo root from its own path.
Mirrors the same file in ~/infraredComp (identical tasks.json schema).

Usage:
    # change status / assignee
    uv run python update_task.py --slug projflow --id t2-3 --status active --assignee 李四
    # rename + dates
    uv run python update_task.py --slug projflow --id t2-3 --title "轮廓提取 v2" --end 2026-07-20
    # mark done
    uv run python update_task.py --slug projflow --id t2-3 --status completed
    # description: --description REPLACES the whole field; --append-description APPENDS
    uv run python update_task.py --slug projflow --id t2-3 --description "全新描述"
    uv run python update_task.py --slug projflow --id t2-3 --append-description "【进度@2026-07-15】wrapper 已就位，crash 待解"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mgmt_io
import task_schema

# CLI flag -> task node field; --status is normalized via resolve_status
FLAG_FIELD = {
    "title": "title",
    "assignee": "assignee",
    "start": "startDate",
    "end": "endDate",
    "description": "description",
    "note_path": "notePath",
    "priority": "priority",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True, help="project slug")
    ap.add_argument("--id", required=True, help="task id (exact match)")
    ap.add_argument("--status", default=None, help="completed/active/planned/paused/blocked")
    ap.add_argument("--title", default=None)
    ap.add_argument("--assignee", default=None)
    ap.add_argument("--start", default=None, help="startDate (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="endDate (YYYY-MM-DD)")
    ap.add_argument("--description", default=None, help="整体替换 description（原内容被覆盖）")
    ap.add_argument("--append-description", default=None, help="追加到 description 末尾（与 --description 互斥，用换行连接）")
    ap.add_argument("--note-path", default=None, help="相对项目目录的笔记 markdown 路径")
    ap.add_argument("--priority", default=None)
    ap.add_argument("--hidden", action="store_true", default=None,
                    help="标记为不展示（项目树默认隐藏）")
    ap.add_argument("--no-hidden", dest="hidden", action="store_false",
                    help="取消不展示标记")
    args = ap.parse_args()

    tree = mgmt_io.read_tasks(args.slug)
    tasks = tree.get("tasks", [])
    parent_list, idx, task = mgmt_io.find_task_by_id(tasks, args.id)
    if task is None:
        sys.exit(f"error: task {args.id!r} not found in project {args.slug!r}")

    updates = {}
    if args.status is not None:
        updates["status"] = task_schema.resolve_status(args.status)
    for flag, field in FLAG_FIELD.items():
        val = getattr(args, flag)
        if val is not None:
            updates[field] = val
    if args.hidden is True:
        updates["hidden"] = True

    # --append-description: append to existing description (mutually exclusive
    # with --description, which replaces via FLAG_FIELD above).
    if args.append_description is not None:
        if args.description is not None:
            sys.exit("error: --description and --append-description are mutually exclusive")
        existing = (task.get("description") or "").rstrip()
        joined = (existing + "\n" + args.append_description) if existing else args.append_description
        updates["description"] = joined

    if not updates and args.hidden is not False:
        print("no updates given; nothing to do.", file=sys.stderr)
        return 1

    # immutable: replace the node in its parent list
    new_task = {**task, **updates}
    removed_keys = []
    if args.hidden is False and "hidden" in task:
        new_task.pop("hidden", None)
        removed_keys.append("hidden")
    parent_list[idx] = new_task
    mgmt_io.write_tasks(args.slug, tree)
    path = mgmt_io.tasks_json_path(args.slug)
    changed = sorted(set(list(updates.keys()) + removed_keys))
    print(f"✓ updated task {args.id!r}: {', '.join(changed)}  ({mgmt_io.rel(path)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
