# AGENTS.md — infraredComp 架构锚点

infraredComp 是一个红外/轮廓视频压缩评测平台,镜像 ProjFlow 全栈架构:
**FastAPI 后端**(`server/`)+ **Vue3 前端**(`web/`)+ 三个业务模块
(management / papers / contour-video benchmark)。前后端解耦,后端只读持久化数据。

## 顶层布局

```
infraredComp/
├── server/                 # FastAPI 后端(:8091)
│   ├── main.py             # app + CORS + 挂载 routers + /api/health
│   ├── config.py           # 所有路径常量(单一来源)
│   ├── db.py               # papers SQLite(schema/upsert/row_to_dict)
│   ├── parsers/            # management 的 markdown 解析器
│   │   ├── markdown_table.py team_parser.py report_parser.py
│   │   ├── tasks_parser.py milestones_parser.py projects_parser.py
│   ├── routers/
│   │   ├── management.py   # /api/management/* (16 GET,只读 markdown)
│   │   ├── papers.py       # /api/papers/* (10 端点,SQLite)
│   │   └── benchmark.py    # /api/benchmark/* (读 results/video/results.json)
│   └── utils/file_utils.py # safe_resolve/scan_directory/read_file
├── management/             # 项目管理数据(markdown,与 ProjFlow 同格式)
│   ├── team/ daily/ weekly/ monthly/   # YYYY/MM/DD-作者.md 等
│   └── docs/{tasks.md,milestones.md,meetings/,projects/<slug>/}
├── papers/                 # 论文模块数据(config/docs/scripts);SQLite 在 data/
├── data/                   # papers.db(SQLite)
├── benchmark/              # 图像压缩 benchmark(既有,legacy)+ video/
│   └── video/              # ★ 两阶段轮廓视频压缩评测(evaluation 对应模块)
│       ├── config.py ffmpeg_util.py data.py
│       ├── extractors/{base,canny,sobel}.py   # 阶段1 可插拔提取器
│       ├── codecs/{base,x264,x265,svtav1,vp9}.py  # 阶段2 codec
│       ├── stage1_extract.py  # 原始视频→无损轮廓帧序列
│       ├── stage2_benchmark.py # 轮廓视频→视频 codec 压缩→评测
│       ├── metrics.py aggregate.py visualize.py html_report.py
│       ├── artifact_io.py __main__.py verify.py
├── datasets/{raw,<method>}  # raw=原始输入;<method>(canny/sobel/hed/...)=阶段1 轮廓产物(datasets/<method>/<source>/)
├── results/video/          # 阶段2 产物:results.json + bitstreams/ + recon/ + charts/
├── web/                    # Vue3 + Naive UI + Chart.js(:3001)
│   └── src/{api,views,router,layouts,components,styles}/
├── scripts/{download_dataset.py, import_papers.py}
├── start_services.sh       # 一键启 server+web
├── pyproject.toml          # uv 管理;含 fastapi/uvicorn/pydantic/pyyaml/torch/compressai
└── README.md
```

## 数据流(三大模块)

| 模块 | 数据源 | 后端端点 | 前端 |
|------|--------|----------|------|
| management | `management/` markdown 文件 | `/api/management/*`(只读,parser 解析) | `views/management/*` |
| papers | `data/papers.db` SQLite | `/api/papers/*`(增改 note/star/pin) | `views/papers/*` |
| benchmark | `results/video/results.json`(runner 写) | `/api/benchmark/*`(只读 JSON) | `views/benchmark/BenchmarkResults.vue` |

**执行与报告解耦**:benchmark 的执行走 CLI(`python -m benchmark.video`),后端 `/api/benchmark/*`
只读 `results.json`,不依赖 runner 在线(同 ProjFlow evaluation 边界)。

## 轮廓视频评测两阶段(核心)

1. **阶段1 提取**:`原始视频(mp4/帧目录)→ [可插拔提取器:canny/sobel] → 无损灰度 PNG 帧序列 + manifest`
   产物存 `datasets/<method>/<source>/`,既是阶段2 输入也是质量基准。
2. **阶段2 评测**:`轮廓视频 → [x264/x265/svtav1/vp9 @ 多 CRF] → 重建 → 逐帧 PSNR/SSIM + 码率 + fps + 时序一致性`
   产物 `results/video/results.json`(+ bitstreams/recon/charts/report.html)。

## 启动

```bash
bash start_services.sh                 # 后端:8091 + 前端:3001
# 或分开:
uv run uvicorn server.main:app --port 8091      # 后端
cd web && pnpm dev                              # 前端(需 Node22+,用 node@25)
```

前端经 Vite proxy `/api` → `localhost:8091`(见 `web/vite.config.js`)。

## 常用命令

```bash
# 轮廓视频评测
uv run python -m benchmark.video.verify                          # 端到端自检
uv run python -m benchmark.video --input datasets/raw/xxx.mp4 \
  --method canny --crfs 18,23,28,33 --codecs x264,x265,svtav1,vp9  # 全流程
uv run python -m benchmark.video --input datasets/raw/xxx.mp4 --method sobel --extract-only  # 仅阶段1
uv run python -m benchmark.video --input datasets/canny/demo --skip-extract                 # 仅阶段2

# 论文导入(从 web/src/data/papers.json → data/papers.db)
uv run python scripts/import_papers.py

# 后端验证
curl --noproxy '*' http://localhost:8091/api/health
curl --noproxy '*' http://localhost:8091/api/benchmark/results
curl --noproxy '*' 'http://localhost:8091/api/papers?limit=5'
curl --noproxy '*' http://localhost:8091/api/management/team
```

## 约定

- **路径单一来源**:所有路径常量在 `server/config.py` / `benchmark/video/config.py`,不在业务代码硬编码。
- **数据集位置可配置**:`DATASETS_DIR` 经 `INFRACOMP_DATASETS_DIR` 环境变量重定位(默认 `<repo>/datasets`),`benchmark/video/config.py`、`server/config.py`、`scripts/download_dataset.py`、`benchmark/runner.py`、`benchmark/demo.py` 同步读取;获取/接入数据集见 `datasets/README.md` 与 `/dataset-management` skill。
- **只读后端**:management/benchmark 后端只读;papers 仅 note/star/pin/blog 改动。所有写操作除论文笔记外经 CLI。
- **路径穿越防护**:`server/utils/file_utils.py::safe_resolve` + 各 router 的 regex 校验。
- **ffmpeg 统一 `-pix_fmt yuv420p`**:所有 codec 编码统一像素格式(可移植、chroma 一致、PSNR 可比),
  decode 用 `gray` 出单通道 PNG;奇数尺寸经 `pad` 滤镜补偶,重建裁回原尺寸再算指标。
- **前端**:一个 api 模块对应一个后端 router(`api/management.js`↔`routers/management.py` 等);
  axios `baseURL:/api`,响应拦截器 unwrap `response.data`(同 ProjFlow)。
