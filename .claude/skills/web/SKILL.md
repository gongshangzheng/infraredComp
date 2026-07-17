---
name: web
description: |
  infraredComp Web 全栈(FastAPI server/ + Vue3 web/)开发指南。用于后端 API 开发、前端页面开发、服务启动调试等。
  触发场景：(1) 启动后端+前端服务 (2) 开发后端 API 端点 (3) 开发前端页面/视图 (4) 调试 API 与前端联调 (5) 查看日志
---

# infraredComp Web 全栈开发指南

`server/`(FastAPI 后端)+ `web/`(Vue3 前端)开发指南。仓库根(含 `pyproject.toml` 的目录)。所有路径相对仓库根;先 `cd` 到仓库根再运行,或用 `$(git rev-parse --show-toplevel)` 定位。

## 项目结构

```text
infraredComp/
├── server/                      # FastAPI 后端(:8091)
│   ├── main.py                  # app + CORS + 挂 routers(management/papers/benchmark/evaluation/training) + /api/health
│   ├── config.py                # 路径常量 + CORS_ORIGINS + OUTPUTS_DIR(=results/video/)
│   ├── db.py                    # papers SQLite
│   ├── parsers/                 # management markdown 解析器
│   │   ├── markdown_table.py team_parser.py report_parser.py
│   │   └── tasks_parser.py milestones_parser.py projects_parser.py
│   ├── routers/
│   │   ├── management.py        # /api/management/* (16 GET,只读)
│   │   ├── papers.py            # /api/papers/* (10 端点)
│   │   ├── benchmark.py         # /api/benchmark/* (读 results.json + /runs 列 contour)
│   │   ├── evaluation.py        # /api/evaluation/* (contour-video 适配:codecs+DL image+ssf2020/dcvc_rt,kind=learned-video /datasets/datasets/{id}/datasets/{id}/media /results+output_video /outputs 按需 /run 触发 run_osu_baseline 或 run_all_subprocess; checkpoint→eval 打通; dataset_detail 返回 view_source/view_video)
│   │   └── training.py          # /api/training/* (CompressAI/ELIC /runs+loss_series/checkpoints/outputs; POST /run 触发 scripts/train_model.py)
│   └── utils/file_utils.py      # safe_resolve/scan_directory/read_file
├── web/                         # Vue3 + Naive UI + vue-echarts(:3001)
│   ├── vite.config.js           # base /infraredComp/ + proxy /api→:8091
│   └── src/
│       ├── api/                 # request.js + {management,papers,benchmark,evaluation,train,datasetPreview}.js
│       ├── stores/theme.js      # Pinia dark/light store(localStorage + <html data-theme>)
│       ├── components/common/   # EmptyState/MarkdownRenderer/StatusBadge
│       ├── views/{management,papers,evaluation,training}/
│       ├── router/index.js
│       └── layouts/MainLayout.vue
├── management/                  # management 数据(markdown)
├── data/papers.db              # papers SQLite
├── results/video/              # 评测产物 + 展示 mp4 缓存
│   ├── results.json            # OSU/demo 评测结果
│   ├── xiph_cif.json           # Xiph 评测结果(独立文件)
│   ├── bitstreams/              # 压缩码流
│   ├── recon/                   # 重建帧
│   ├── source/                  # 原始展示 mp4(惰性生成,缓存)
│   └── contour_mp4/             # 轮廓展示 mp4(惰性生成,缓存)
├── datasets/
│   ├── raw/                     # 原始视频
│   │   ├── xiph_cif/            # Xiph 6 段 CIF y4m + manifest
│   │   └── osu_color_thermal/   # OSU 6 段 seq{1..6}.mp4 + manifest
│   └── contour/<source>/<method>/  # 阶段1 产物:**只有 contour.mp4 + manifest.json**(无 PNG 帧)
├── start_services.ps1           # Windows 一键启动(uvicorn :8091 + pnpm dev :3001)
└── start_services.sh            # Linux/Mac 备用(项目根)
```

## 启动服务

后端 **8091**、前端 **3001**(vite proxy `/api` → 8091)。

```bash
# Windows(推荐):
powershell -File start_services.ps1
# 后端日志: backend.log  前端日志: frontend.log

# Linux/Mac:
bash start_services.sh

# 手动分开
cd <repo-root>
uv run uvicorn server.main:app --host 0.0.0.0 --port 8091   # 后端
cd web && pnpm dev                                           # 前端(需 Node 22+)
```

**uv 网络炸(代理 10061)** 时绕过:`.venv/Scripts/python.exe -m uvicorn server.main:app --port 8091` 或用 conda compression env。

访问:前端 `http://localhost:3001/infraredComp/`,后端文档 `http://localhost:8091/api/docs`。

## 1. 评测视图(侧栏「评测体系」)

```
/evaluation/datasets           DatasetManage     数据集家族列表(原始 + 轮廓) + 状态
/evaluation/datasets/:id       DatasetDetail      **序列展开:左=原始视频 / 右=轮廓视频**(多方法 n-tabs)
/evaluation/run                EvalRun            选 dataset+codecs+crfs+sequences+mode 触发 baseline
/evaluation/speed              SpeedResults       视频网格(speed run 主观看)
                                 (旧 /evaluation/results + /evaluation/compare + /evaluation/outputs 重定向到此)
```

`mode` 只影响"数据集子集(--sequences)+ 跳哪个展示页",不在跑代码分叉;**speed/formal 默认都不截断帧**。详见 evaluation skill §3。

**EvalRun** 选 DL image codec(CompressAI/ELIC, kind='dl')时出现 checkpoint 选择器(选 trained 不选=pretrained)+ quality;选 learned-video(ssf2020)时同样;DCVC-RT(无 trained checkpoint)显示 setup note。

**POST /api/evaluation/run** 接 `dataset_id`/`codecs`/`crfs`/`method`/`sequences`/`mode`,按 dataset 选脚本(xiph_cif → `run_all_subprocess.py`,osu → `run_osu_baseline.py`),Popen 传 CLI 参数。`frames` 固定为 None(从不传 --frames)。

## 2. 数据集页面展示管线(`/evaluation/datasets/:id`)

序列展开:左=原始视频、右=轮廓视频(多方法 n-tabs),都是可播放 mp4。媒体来源:
- **原始视频(浏览器可播)**:`source/{seq}.mp4` 经 `/api/evaluation/outputs/source/{seq}.mp4`(由 `_ensure_source_video` 从 raw 截到 contour 帧窗口生成)。
- **轮廓视频**:`contour_mp4/{seq}_{method}.mp4` 经 `/api/evaluation/outputs/contour_mp4/{seq}_{method}.mp4`(由 `_ensure_contour_video` 从 `contour.mp4`(无损,stage1 产物)转有损;**只读 contour.mp4,无 PNG 回退**)。
- **`seq.file` 字段(给 `/datasets/{id}/media/{path}`)**:后端 `_load_raw_datasets` 剥 `datasets/` 前缀后返回相对 DATASETS_DIR 的路径(如 `raw/xiph_cif/akiyo_cif.y4m`)。

`_ensure_*` 用 `safe_resolve` 防穿越;惰性 ffmpeg 生成,缓存于 `_VIDEO_CACHE` + 磁盘。重启后端清 `_VIDEO_CACHE` → 首次 `/datasets/{id}` 重新生成。

**重要:浏览器可能缓存旧展示 mp4**(URL 不变),改完管线后**硬刷新**页面。

## 3. 训练视图(侧栏「训练体系」)

```
/training/run                  TrainRun
/training/results              TrainResults(loss 曲线 + checkpoint 文件列表)
/training/models               TrainModelManage
/training/datasets             TrainDatasetManage
/training/configs              TrainConfigManage
```

`scripts/train_model.py` 实例化 CompressAI image model / ELIC,Adam + RD loss,每 epoch 写 `results/training/metrics.json` + `checkpoints/{model}__q{q}__{ts}.pth` + `logs/{run_id}.log`。**命名 `{model}__q{q}__{ts}.pth` 让 eval `/api/evaluation/models` 自动发现**。

**checkpoint→eval 打通**:`evaluation.py /models` 的 DL/learned-video codec 带 `checkpoint` 字段 = `_trained_checkpoints_for(model_id)`;EvalRun 选 checkpoint_id → 解析路径 + 返回 image benchmark CLI note。`benchmark/learned.py:_load_model` + `elic_model.py:load_elic_model` 加 checkpoint_path override。

## 4. API 端点(关键)

| 模块 | 端点 | 说明 |
|------|------|------|
| health | `GET /api/health` | `{status:ok}` |
| management | `GET /api/management/{team,team/:id,daily,daily/:date/:author,weekly,weekly/:y/:w/:author,monthly,monthly/:y/:m/:author,tasks,milestones,meetings,meetings/:date,projects,projects/:slug,projects/:slug/tasks,projects/:slug/notes/:path}` | 16 个只读 |
| papers | `GET /api/papers`(limit/offset/source/category)、`/stats/summary`、`/{id}`、`/{id}/note`;`PUT /{id}/note`、`/{id}/blog`、`/{id}/star`、`/{id}/pin`;`POST /{id}/summarize`(桩) | SQLite |
| benchmark | `GET /api/benchmark/results`(codec/sequence/crf 过滤)、`/results/compare`、`/runs`(列 contour manifest,兼容 `<source>/<method>/`) | 只读 results.json |
| evaluation | `GET /api/evaluation/{models,datasets,configs,methods,results,results/compare,results/aggregate,results/{id},outputs,outputs/{path}}` `GET /datasets`、`/datasets/{id}`、`/datasets/{id}/media/{path:path}` `POST /run`、`/datasets/{id}/download` | models=codecs+DL+learned-video, datasets=序列+轮廓+imagenet 图像,results 聚合多 json,outputs 流式,/run 异步触发 |
| training | `GET /api/training/{models,datasets,runs,run/{id},checkpoints,outputs}` `POST /run`、`/models/{id}/delete`、`/checkpoints/{id}/delete` | run 触发 scripts/train_model.py,checkpoints→eval 打通 |

curl 示例(绕过系统代理):

```bash
curl --noproxy '*' http://localhost:8091/api/health
curl --noproxy '*' 'http://localhost:8091/api/papers?limit=5'
curl --noproxy '*' 'http://localhost:8091/api/evaluation/datasets/xiph_cif'
curl --noproxy '*' 'http://localhost:8091/api/evaluation/outputs/source/akiyo_cif.mp4'
```

> `--noproxy '*'` 绕过系统 http_proxy,避免 localhost 走代理。

## 5. 后端开发约定(镜像 ProjFlow)

- **路径单一来源**:`server/config.py` 导出 `BASE_DIR`+各模块 `*_DIR`/`PAPERS_DB`,router 不硬编码。
- **三层**:`router`(HTTP)→ `parser`/`db`(数据转换)→ `utils/file_utils`(I/O)。
- **只读后端**:management/benchmark 后端只读持久化数据;papers 允许 note/star/pin/blog;evaluation 触发 run 用 Popen。
- **执行/报告解耦**:`/api/benchmark/*`、`/api/evaluation/results` 只读 `results/video/*.json`,不依赖 runner 在线。
- **路径穿越防护**:`utils/file_utils.py::safe_resolve`(realpath+containment)。
- **新增 router**:在 `server/routers/` 加文件定义 `router = APIRouter(prefix="/api/<m>", tags=[...])`,在 `server/main.py` `import` 并 `app.include_router(...)`。

## 6. 前端开发约定(镜像 ProjFlow)

- **一个 api 模块对应一个 router**:`web/src/api/<m>.js` ↔ `server/routers/<m>.py`。
- **dark/light 主题**:Pinia store `web/src/stores/theme.js`(localStorage key `infraredcomp-theme`,默认 `prefers-color-scheme`);App.vue 绑 `:theme="naiveTheme"`(dark 用 darkTheme)+ `<html data-theme>` 同步;MainLayout 右上切换;`styles/index.scss` `:root` + `:root[data-theme=dark]` token 块;视图样式用 `var(--color-card/border/text-*)`,勿硬编码。
- **输出视频按需加载(核心)**:`<video preload="none">` 不点开不取字节。选结果/输出时才赋 `src`;`src` = `/api/evaluation/outputs/<relpath>`(vite proxy 转 backend)。**禁止预加载所有视频**。
- **路由**:`web/src/router/index.js`,每路由带 `meta:{title,module}`,`meta.module=evaluation` 驱动 MainLayout 侧栏高亮与面包屑。
- **侧边栏**:`MainLayout.vue` 的 `menuOptions`(评测体系含 datasets/run/speed/formal 等 children + 训练体系 5 children)+ `allKeys`(activeKey 匹配)+ `moduleMap`(面包屑)。
- **样式**:复用 `web/src/styles/variables.scss` token(`$primary-color #4f46e5`)+ `var(--color-*)` 运行时 token(暗色适配)。
- **数据集页布局**:`DatasetDetail.vue` 序列展开两栏 grid `1fr 1fr`(左原始 / 右轮廓),多方法用 n-tabs。`outputUrl(view_source/view_video) → /api/evaluation/outputs/...`;`getDatasetMediaUrl(datasetId, path)` 给 frame gallery(回退用)。

## 7. 论文导入(迁移种子数据)

```bash
uv run python scripts/import_papers.py
# 从 web/src/data/papers.json 导入到 data/papers.db
```

## 8. 调试 / 查看日志

```bash
# 后端启动失败查日志
tail -50 logs/backend.log
# 前端
tail -50 logs/frontend.log

# 端口占用
# Windows:
netstat -ano | grep 8091
# Linux/Mac:
lsof -nP -iTCP:8091 -sTCP:LISTEN

# 重启后端
# Windows:
netstat -ano | grep ":8091" | grep LISTENING    # 找 PID
taskkill //F //PID <PID>
# Linux/Mac:
lsof -nP -iTCP:8091 -sTCP:LISTEN -t | xargs kill
uv run uvicorn server.main:app --port 8091   # 或 .venv/Scripts/python.exe -m uvicorn ...

# 前端构建检查
cd web && node_modules/.bin/vite build           # 不需要 pnpm 在 PATH
```

## 9. 数据集/模型/训练数据流

```
datasets/raw/<dataset>/<seq>.<ext>        ← download scripts (Xiph 公开,OSU 常 403)
    │
    ▼  stage1 (extract_contour_video)
datasets/contour/<seq>/<method>/
    ├── contour.mp4            ← 无损 (libx264 -qp 0 -pix_fmt yuv420p),**无 PNG**
    └── manifest.json
    │
    ▼  stage2 (run_benchmark / bench_one / run_all_video_models 从 contour.mp4 解码临时帧)
results/video/
    ├── results.json           ← OSU/demo
    ├── xiph_cif.json          ← Xiph(独立)
    ├── bitstreams/            ← 压缩码流
    ├── recon/                 ← 重建帧
    ├── source/                ← 原始展示 mp4(惰性)
    └── contour_mp4/           ← 轮廓展示 mp4(惰性)
```

## 关键约定

- **CORS**:`server/config.py::CORS_ORIGINS` 允许 `localhost:3001/5173/3002`。
- **Pages 部署**:`.github/workflows/deploy.yml` 静态部署 `web/dist` 到 GitHub Pages(base `/infraredComp/`)。本地 dev 走 proxy;Pages 上无后端时 API 视图降级 EmptyState。
- **node 版本**:pnpm 需 Node 22+。`start_services.ps1` 自动选 node@25/24/22;CI 用 Node 24。
- **数据目录**:management(markdown)/papers(SQLite)/evaluation(results/video/*.json)各模块自带目录。
- **contour 管线产物是 `contour.mp4`,不是 PNG**;阶段2 从视频解码临时帧(详见 evaluation skill §1-2)。

## 常用命令

```bash
# 启动
powershell -File start_services.ps1          # Windows
bash start_services.sh                       # Linux/Mac

# 健康检查
curl --noproxy '*' http://localhost:8091/api/health

# 评测
curl --noproxy '*' 'http://localhost:8091/api/evaluation/datasets/xiph_cif'
curl --noproxy '*' 'http://localhost:8091/api/evaluation/results/aggregate?dataset=Xiph-CIF-natural'

# 重启后端
netstat -ano | grep ":8091" | grep LISTENING   # Windows 找 PID
taskkill //F //PID <PID>
.venv/Scripts/python.exe -m uvicorn server.main:app --port 8091  # 绕过 uv

# 论文迁移
uv run python scripts/import_papers.py
```
