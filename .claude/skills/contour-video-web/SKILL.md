---
name: contour-video-web
description: |
  infraredComp Web 全栈(FastAPI server/ + Vue3 web/)开发指南。用于后端 API 开发、前端页面开发、服务启动调试等。
  触发场景：(1) 启动后端+前端服务 (2) 开发后端 API 端点 (3) 开发前端页面/视图 (4) 调试 API 与前端联调 (5) 查看日志
---

# infraredComp Web 全栈开发指南

本 skill 提供 `server/`(FastAPI 后端)+ `web/`(Vue3 前端)的完整开发指南。仓库根(含 `pyproject.toml` 的目录),架构镜像 ProjFlow。所有路径相对仓库根;先 `cd` 到仓库根再运行,或用 `$(git rev-parse --show-toplevel)` 定位。

## 项目结构

```text
infraredComp/
├── server/                      # FastAPI 后端(:8091)
│   ├── main.py                  # app + CORS + 挂载 routers(management/papers/benchmark/evaluation) + /api/health
│   ├── config.py                # 路径常量 + CORS_ORIGINS + OUTPUTS_DIR(=results/video/)
│   ├── db.py                    # papers SQLite(schema/upsert_paper/row_to_dict)
│   ├── parsers/                 # management markdown 解析器
│   │   ├── markdown_table.py team_parser.py report_parser.py
│   │   └── tasks_parser.py milestones_parser.py projects_parser.py
│   ├── routers/
│   │   ├── management.py        # /api/management/* (16 GET,只读)
│   │   ├── papers.py            # /api/papers/* (10 端点)
│   │   ├── benchmark.py         # /api/benchmark/* (读 results.json + /runs 列 contour)
│   │   └── evaluation.py        # /api/evaluation/* (contour-video 适配:codecs/序列/results+output_video/outputs 按需视频/run)
│   └── utils/file_utils.py      # safe_resolve/scan_directory/read_file
├── web/                         # Vue3 + Naive UI + vue-echarts(:3001)
│   ├── vite.config.js           # base /infraredComp/ + proxy /api→:8091
│   └── src/
│       ├── api/                 # request.js + {management,papers,benchmark,evaluation}.js
│       ├── stores/theme.js      # Pinia dark/light store(localStorage + <html data-theme>)
│       ├── components/common/   # EmptyState/MarkdownRenderer/StatusBadge/VideoModal
│       ├── views/{management,papers,evaluation}/  # 评测子页 7 个(见下)
│       ├── router/index.js      # 路由(/evaluation/* 子路由组,meta.module 驱动侧边栏)
│       └── layouts/MainLayout.vue  # 侧边栏(评测体系子菜单) + 主题切换按钮 + breadcrumb
├── management/                  # management 数据(markdown)
├── data/papers.db              # papers SQLite
├── results/video/              # 评测产物:results.json + bitstreams/(压缩码流) + recon/(重建帧)
│   └── contour 存储按方法分目录: datasets/contour/<source>/<method>/
└── start_services.sh            # 一键启动
```

### 评测子页（/evaluation/*，7 个，取代原单页 /benchmark）
`EvalRun`(运行,触发 run_osu_baseline 后按需看视频+指标) · `EvalResults`(指标表+图,行内「查看视频」按需弹) · `EvalMethodCompare`(方法对比矩阵:canny/sobel 各方法下 baseline) · `EvalOutputs`(浏览 bitstreams/recon,按需播放) · `ModelManage`(codec 配置) · `DatasetManage`(序列) · `ConfigManage`(评测配置)。菜单「评测体系」下 7 children。`/benchmark` 保留 redirect→`/evaluation/results`。

### 评测模块的上下游关系（见 upstream-sync skill）
共享脚手架（6 视图 + VideoModal + api/evaluation.js + evaluation.py 通用契约 + /outputs 视频端点）在 **ProjFlow 上游**（commit `c41bb78`）。infraredComp 经 `git checkout upstream/main -- web/src/views/evaluation/ web/src/components/common/VideoModal.vue web/src/api/evaluation.js` 取回（保留 git 血缘）；**后端 `evaluation.py` 是 infraredComp 定制**（接 contour-video 数据源:models=codecs、datasets=序列、results=results.json+output_video、outputs=bitstreams/recon、run=run_osu_baseline）——不直接取上游通用版。EvalMethodCompare 是 infraredComp 独有（contour-video 双阶段多方法对比），未进上游。

## 启动服务

```bash
# 一键(后端 :8091 + 前端 :3001)
bash start_services.sh

# 手动分开
cd <repo-root>
uv run uvicorn server.main:app --host 0.0.0.0 --port 8091          # 后端
cd web && pnpm dev                                                  # 前端(需 Node 22+;start_services.sh 自动选 node@25/24/22)
```

访问:前端 `http://localhost:3001/infraredComp/`,后端文档 `http://localhost:8091/api/docs`。
前端经 Vite proxy `/api` → `localhost:8091`。

## 1. API 端点

| 模块 | 端点 | 说明 |
|------|------|------|
| health | `GET /api/health` | `{status:ok}` |
| management | `GET /api/management/{team,team/:id,daily,daily/:date/:author,weekly,weekly/:y/:w/:author,monthly,monthly/:y/:m/:author,tasks,milestones,meetings,meetings/:date,projects,projects/:slug,projects/:slug/tasks,projects/:slug/notes/:path}` | 16 个只读端点,parser 解析 markdown |
| papers | `GET /api/papers`(limit/offset/source/category)、`/stats/summary`、`/{id}`、`/{id}/note`;`PUT /{id}/note`、`/{id}/blog`、`/{id}/star`、`/{id}/pin`;`POST /{id}/summarize`(桩) | SQLite |
| benchmark | `GET /api/benchmark/results`(codec/sequence/crf 过滤)、`/results/compare`、`/runs`(列 contour manifest,兼容 `<source>/<method>/` 与旧扁平)、`POST /run`(桩) | 只读 results.json |
| evaluation | `GET /api/evaluation/{models,datasets,configs,methods,results,results/compare,results/{id},outputs,outputs/{path}}` `POST /run` | models=codecs、datasets=序列+contour、results=results.json(每条附 output_video)、`/outputs/{path}` 流式 FileResponse(按需服务 bitstreams/recon,safe_resolve 防穿越)、`/run` 异步触发 run_osu_baseline |

curl 示例:

```bash
curl --noproxy '*' http://localhost:8091/api/health
curl --noproxy '*' 'http://localhost:8091/api/papers?limit=5'
curl --noproxy '*' http://localhost:8091/api/management/team
curl --noproxy '*' http://localhost:8091/api/benchmark/results
curl --noproxy '*' 'http://localhost:8091/api/benchmark/results?codec=x264&crf=23'
```

> `--noproxy '*'` 绕过系统 http_proxy,避免 localhost 走代理。

## 2. 后端开发约定(镜像 ProjFlow)

- **路径单一来源**:`server/config.py` 导出 `BASE_DIR`+各模块 `*_DIR`/`PAPERS_DB`/`RESULTS_VIDEO_JSON`,router 不硬编码路径。
- **三层**:`router`(HTTP)→ `parser`/`db`(数据转换)→ `utils/file_utils`(I/O)。
- **只读后端**:management/benchmark 后端只读持久化数据;papers 仅 note/star/pin/blog 改动。benchmark 执行走 CLI。
- **执行/报告解耦**:`/api/benchmark/*` 只读 `results/video/results.json`,不依赖 runner 在线。
- **路径穿越防护**:`utils/file_utils.py::safe_resolve`(realpath+containment)+ router 内 regex 校验(`_DATE_RE`/`_SLUG_RE` 等)。
- **新增 router**:在 `server/routers/` 加文件定义 `router = APIRouter(prefix="/api/<m>", tags=[...])`,在 `server/main.py` `import` 并 `app.include_router(...)`。

## 3. 前端开发约定(镜像 ProjFlow)

- **一个 api 模块对应一个 router**:`web/src/api/{management,papers,benchmark,evaluation}.js` ↔ `server/routers/{...}.py`。
- **dark/light 主题**:Pinia store `web/src/stores/theme.js`(localStorage key `infraredcomp-theme`,默认跟随 `prefers-color-scheme`);App.vue 绑 `:theme="naiveTheme"`(dark 用 darkTheme)+ `<html data-theme>` 同步;MainLayout 右上角切换按钮;`styles/index.scss` 的 `:root` + `:root[data-theme=dark]` token 块是暗色基础(视图样式用 `var(--color-card/border/text-*)` 等 token,勿硬编码 `#fff/#f8fafc`)。从 ProjFlow 上游移植(见 upstream-sync)。
- **输出视频按需加载**(核心):统一用 `components/common/VideoModal.vue` —— `<video v-if="show" :src preload="none">`,**仅在弹窗打开 + 用户按 play 时才请求字节**。调用方(EvalRun/EvalResults/EvalOutputs/EvalMethodCompare)点击时才 `videoSrc = getOutputUrl(path); videoShow = true`。视频 URL = `/api/evaluation/outputs/<relpath>`(vite proxy 转 backend)。**禁止页面预加载所有视频**。
- **评测子页 7 个**(见项目结构节),路由 `/evaluation/*`,`/benchmark` redirect→`/evaluation/results`。
- **vue-echarts**:EvalResults/EvalMethodCompare 用 `vue-echarts` + `echarts`(已在 package.json)。
- **axios**:`web/src/api/request.js`,`baseURL:'/api'`,timeout 15000,响应拦截器返回 `response.data`。
- **视图**:`<script setup>` + Naive UI 按需 import + `ref`/`onMounted`;空态用 `EmptyState.vue`,markdown 用 `MarkdownRenderer.vue`,状态用 `StatusBadge.vue`,视频弹窗用 `VideoModal.vue`。
- **路由**:`web/src/router/index.js`,每路由带 `meta:{title,module}`,`router.beforeEach` 设 `document.title`;`meta.module=evaluation` 驱动 `MainLayout` 侧边栏「评测体系」高亮与面包屑。
- **侧边栏**:`MainLayout.vue` 的 `menuOptions`(评测体系含 7 children)+ `allKeys`(activeKey 匹配)+ `moduleMap`(面包屑)。
- **样式**:复用 `web/src/styles/variables.scss` token(`$primary-color #4f46e5` 等)+ `var(--color-*)` 运行时 token(暗色适配)。

## 4. 论文导入(迁移种子数据)

```bash
# 从 web/src/data/papers.json 导入到 data/papers.db(字段映射见 scripts/import_papers.py)
uv run python scripts/import_papers.py
```

## 5. 调试 / 查看日志

```bash
# 后端启动失败查日志
tail -50 backend.log
# 前端
tail -50 frontend.log

# 端口占用
lsof -nP -iTCP:8091 -sTCP:LISTEN
lsof -nP -iTCP:3001 -sTCP:LISTEN

# 启动后端单进程调试
cd <repo-root> && uv run uvicorn server.main:app --port 8091 --reload

# 前端构建检查
cd web && npx vite build                                           # 需 PATH 含 Node 22+
```

## 关键约定

- **CORS**:`server/config.py::CORS_ORIGINS` 允许 `localhost:3001/5173/3002`(Vite dev 默认 3001)。
- **Pages 部署**:`.github/workflows/deploy.yml` 静态部署 `web/dist` 到 GitHub Pages(base `/infraredComp/`)。本地 dev 走 proxy;Pages 上无后端时 API 视图降级 EmptyState(生产托管后端为后续)。
- **node 版本**:pnpm 需 Node 22+(系统默认 node 20 会崩 `node:sqlite`)。`start_services.sh` 自动 pick `node@25`/`@24`/`@22`;CI 用 Node 24。
- **数据目录**:management(markdown)/papers(SQLite 在 `data/`)/benchmark(`results/video/results.json`)各模块自带数据目录,后端只读(论文笔记除外)。

## 常用命令

```bash
# 启动
bash start_services.sh
# 健康检查
curl --noproxy '*' http://localhost:8091/api/health
# 三模块各取一个端点
curl --noproxy '*' http://localhost:8091/api/management/projects
curl --noproxy '*' 'http://localhost:8091/api/papers?limit=3'
curl --noproxy '*' http://localhost:8091/api/benchmark/results
# 重启后端
lsof -nP -iTCP:8091 -sTCP:LISTEN -t | xargs kill; uv run uvicorn server.main:app --port 8091
# 论文迁移
uv run python scripts/import_papers.py
```
