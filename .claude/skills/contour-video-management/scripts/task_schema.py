"""Task-board section constants for tasks.md.

infraredComp and ProjFlow use an IDENTICAL tasks.md schema (verified in both
repos): three sections in fixed order, each with a distinct column set. The
generic ``mgmt_io`` auto-detects columns at runtime, so these constants are
only used to (a) validate CLI args and (b) map friendly flag names
(--status, --priority, ...) to the right column.

Section order MUST match ``server/parsers/tasks_parser.py`` section_keys:
    in_progress (进行中) -> pending (待开始) -> completed (已完成)
"""

from __future__ import annotations

import sys

SECTION_ORDER = ["进行中", "待开始", "已完成"]

SECTION_COLS: dict[str, list[str]] = {
    "进行中": ["任务", "负责人", "开始日期", "截止日期", "状态", "备注"],
    "待开始": ["任务", "负责人", "预计开始", "截止日期", "优先级", "备注"],
    "已完成": ["任务", "负责人", "完成日期", "产出", "备注"],
}

SECTION_ALIASES = {
    "in_progress": "进行中", "ongoing": "进行中", "进行中": "进行中",
    "pending": "待开始", "todo": "待开始", "待开始": "待开始",
    "completed": "已完成", "done": "已完成", "已完成": "已完成",
}


def resolve_section(s: str) -> str:
    """Map a CLI section arg (alias or canonical) to the canonical name."""
    key = s.strip().lower()
    if key in SECTION_ALIASES:
        return SECTION_ALIASES[key]
    for canon in SECTION_ORDER:
        if s.strip() == canon:
            return canon
    sys.exit(
        f"error: unknown section {s!r}. Expected one of {SECTION_ORDER} "
        f"(or aliases: in_progress/pending/completed)."
    )
