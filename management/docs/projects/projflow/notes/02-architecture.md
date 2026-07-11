# 架构说明

## 目录约定

```
management/docs/
├── team/          # 团队成员档案
├── daily/ weekly/ monthly/   # 报表
├── meetings/      # 会议纪要
└── projects/      # 项目树数据
    └── {slug}/
        ├── README.md     # 含 YAML frontmatter 的项目说明
        ├── tasks.json    # 任务树
        └── notes/        # 任务笔记 markdown
```

## 解析层

- `projects_parser.py` 负责项目列表、README 解析、任务树级联完成计算
- 笔记读取做了路径穿越防护（拒绝 `..` 开头路径）

## 前端

- `Projects.vue`：项目 Tab + 任务树 + README/任务详情双栏
- `ProjectTaskNode.vue`：递归渲染 git 风格分支线节点
