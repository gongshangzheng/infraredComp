---
name: contour-video-management
description: |
  infraredComp 项目管理模块操作指南。用于团队成员管理、日报/周报/月报、任务看板、里程碑、会议纪要等 CRUD 操作（脚本直接改 markdown，后端只读）。
  触发场景：(1) 添加/修改/删除团队成员，(2) 创建/更新/删除报表，(3) 管理任务看板(增删改查+跨段移动)，(4) 创建/更新/删除会议纪要，(5) 了解项目结构
---

# infraredComp 项目管理模块

本 skill 提供 infraredComp 项目管理模块（`management/`）的完整操作指南，以及一套**自定位**的 CRUD 脚本，直接读写 `management/` 下的 markdown 文件。

> **架构前提**：后端 `server/routers/management.py` 是**只读**的（16 个 GET 端点，解析 markdown 暴露给前端）。所有"增删改"由本 skill 的脚本直接修改 markdown 文件完成，再由只读 API 暴露。这与 ProjFlow 镜像。

## 脚本一览（`.claude/skills/contour-video-management/scripts/`）

脚本 **self-locating**（用 `parents[4]` 解析仓库根），同一份文件在 infraredComp 与 ProjFlow 都能跑（两库 `tasks.md` schema 一致）。纯标准库，**推荐用 `uv run python` 运行**（项目是 uv 管理）。

> **Windows 必看**：`python3` 在 Windows 上是 Microsoft Store 的占位 stub，非交互运行直接 exit 49、不可用；且脚本末尾 `print("✓ added ...")` 的 `✓`（U+2713）在 gbk stdout 下会 `UnicodeEncodeError`——崩溃发生在 `write_lines` **之后**，任务其实已写入文件，但 `&&` 链会断、后续命令不执行。Windows 下务必加前缀：`PYTHONUTF8=1 uv run python <script>`。多个脚本串行写同一个 `tasks.md`（每次读全文→写全文），只能 `&&` 串行，不能并行。

| 实体 | 新增 | 修改 | 删除 | 查询 |
|------|------|------|------|------|
| 任务 task | `add_task.py` | `update_task.py`（含跨段 move） | `delete_task.py` | `list_tasks.py` |
| 成员 member | `add_member.py` | `update_member.py` | `delete_member.py` | — |
| 会议 meeting | `create_meeting.py` | `update_meeting.py` | `delete_meeting.py` | — |
| 报表 report | `create_report.py` | `update_report.py` | `delete_report.py` | — |

## 项目结构

```text
management/
├── team/           # 团队成员
│   ├── README.md   # 成员列表表(姓名|英文标识|角色|入职日期) + 暂无占位
│   └── {id}.md     # 个人档案(基本信息表 + 技术栈 + 负责模块 + 备注)
├── daily/          # 日报  YYYY/MM/DD-{author}.md
├── weekly/         # 周报  YYYY/W{NN}-{author}.md
├── monthly/        # 月报  YYYY/{MM}-{author}.md
└── docs/
    ├── tasks.md    # 任务看板(三段: 进行中/待开始/已完成)
    ├── milestones.md # 里程碑
    ├── projects/   # 项目树 {slug}/README.md + tasks.json + notes/
    └── meetings/   # 会议纪要 YYYY-MM-DD.md
```

只读 API（`GET /api/management/*`，端口 8091）：`team`、`daily`、`weekly`、`monthly`、`tasks`、`milestones`、`meetings`、`projects` 等，详见 `server/routers/management.py`。脚本改完 markdown，前端经 API 即可看到。

---

## 1. 任务看板 CRUD

`management/docs/tasks.md` 三段，**每段列不同**（后端 `tasks_parser` 按表顺序映射 in_progress/pending/completed）：

```markdown
## 进行中
| 任务 | 负责人 | 开始日期 | 截止日期 | 状态 | 备注 |

## 待开始
| 任务 | 负责人 | 预计开始 | 截止日期 | 优先级 | 备注 |

## 已完成
| 任务 | 负责人 | 完成日期 | 产出 | 备注 |
```

```bash
SD=.claude/skills/contour-video-management/scripts

# 新增任务（进行中）
uv run python $SD/add_task.py --section 进行中 --name "轮廓提取优化" \
  --owner 张三 --start 2026-07-11 --end 2026-07-18 --status 🟢 --note "sobel 降噪"
# 新增任务（待开始，带优先级）
uv run python $SD/add_task.py --section 待开始 --name "AV1 baseline" --owner 李四 --priority P1

# 修改字段（只改传了的）
uv run python $SD/update_task.py --section 进行中 --name "轮廓提取优化" --status 🔴 --owner 李四
# 改名
uv run python $SD/update_task.py --section 进行中 --name "轮廓提取优化" --new-name "轮廓提取 v2"
# 完成（跨段 move 进行中 -> 已完成，完成日期默认今天，产出可指定）
uv run python $SD/update_task.py --section 进行中 --name "轮廓提取优化" --move-to 已完成 --output "PR#12"

# 删除
uv run python $SD/delete_task.py --section 进行中 --name "轮廓提取优化"

# 查询
uv run python $SD/list_tasks.py                 # 全部
uv run python $SD/list_tasks.py --section 待开始
```

`--section` 接受中文（进行中/待开始/已完成）或英文别名（in_progress/pending/completed、ongoing/todo/done）。状态 emoji：🟢 正常 / 🟡 风险 / 🔴 阻塞 / ✅ 完成。

---

## 2. 团队成员 CRUD

```bash
# 新增（自动去掉 README 的"暂无"占位行，并生成 {id}.md 档案）
uv run python $SD/add_member.py --name 张三 --id zhangsan --role 算法工程师 \
  --join-date 2026-01-15 --research "红外视频压缩" --tech "Python,PyTorch,ffmpeg" --modules "评测,论文"

# 修改（按 --id 定位；--new-id 会改 id 并重命名档案文件）
uv run python $SD/update_member.py --id zhangsan --role "高级算法工程师" --join-date 2026-01-15
uv run python $SD/update_member.py --id zhangsan --name 张三丰 --new-id zhangsanfeng

# 删除（同时删 README 行 + 档案文件；--keep-profile 保留档案）
uv run python $SD/delete_member.py --id zhangsan
uv run python $SD/delete_member.py --id zhangsan --keep-profile
```

---

## 3. 报表 CRUD

文件路径：daily `YYYY/MM/DD-{author}.md`、weekly `YYYY/W{NN}-{author}.md`、monthly `YYYY/{MM}-{author}.md`。

```bash
# 创建
uv run python $SD/create_report.py --type daily   --author zhangsan --date 2026-07-11
uv run python $SD/create_report.py --type weekly  --author zhangsan --year 2026 --week 28
uv run python $SD/create_report.py --type monthly --author zhangsan --year 2026 --month 07

# 更新（追加工作/计划条目，或重写备注）
uv run python $SD/update_report.py --type daily --author zhangsan --date 2026-07-11 \
  --append-work "完成轮廓提取 baseline" --append-plan "跑 AV1 CRF 扫描" --note "ok"

# 删除
uv run python $SD/delete_report.py --type daily   --author zhangsan --date 2026-07-11
uv run python $SD/delete_report.py --type weekly  --author zhangsan --year 2026 --week 28
uv run python $SD/delete_report.py --type monthly --author zhangsan --year 2026 --month 07
```

---

## 4. 会议纪要 CRUD

文件：`management/docs/meetings/YYYY-MM-DD.md`。

```bash
# 创建
uv run python $SD/create_meeting.py --date 2026-07-11 \
  --participants "张三、李四" --recorder 张三 --topics "进度回顾,方案讨论" \
  --decision "确认用 sobel" --todo "张三:跑 CRF 扫描"

# 更新（换参会人/记录人，追加决议/待办）
uv run python $SD/update_meeting.py --date 2026-07-11 --participants "张三、李四、王五" \
  --append-decision "追加决议" --append-todo "李四:调研 X"

# 删除
uv run python $SD/delete_meeting.py --date 2026-07-11
```

---

## 5. 里程碑 / 项目树

`milestones.md` 单表（名称|目标日期|状态|备注）；项目树 `docs/projects/{slug}/`（README.md + tasks.json + notes/）。这两块暂无专用 CRUD 脚本，直接编辑 markdown 即可，只读 API `GET /api/management/milestones`、`GET /api/management/projects[/{slug}[/tasks|/notes/{path}]]` 会自动解析。

## 关键约定

- **只读后端**：management 后端只读 markdown；所有写操作走本 skill 脚本（直接改文件 + 原子写）。
- **parser 兼容**：脚本的表格编辑器**保留表头**、按段实际列数生成行，与 `server/parsers/tasks_parser.py`（按表顺序 + 按表头名取列）兼容。
- **空表保留**：`markdown_table.parse_markdown_tables` 已修——空表（header+separator 无数据行）也保留，保证 `tasks_parser` 的位置映射（in_progress/pending/completed）不因某段为空而错位。
- **跨段 move**：`update_task --move-to` 自动按目标段列重映射（任务/负责人带过去；→已完成 完成日期默认今天、产出可来自 `--output` 或原备注）。
- **自定位**：脚本用 `parents[4]` 解析仓库根，从任意 cwd 运行均可；同一份文件在 infraredComp 与 ProjFlow 通用。

## 常用命令

```bash
SD=.claude/skills/contour-video-management/scripts
uv run python $SD/list_tasks.py                      # 看任务看板
uv run python $SD/add_task.py --section 进行中 --name "X" --owner Y --start 2026-07-11 --end 2026-07-18
curl --noproxy '*' http://localhost:8091/api/management/tasks   # 前端所见
```
