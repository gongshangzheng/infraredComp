---
name: dataset-management
description: Manage datasets in infraredComp — configure dataset location via INFRACOMP_DATASETS_DIR, download the FLIR ADAS / OSU Color-Thermal / Xiph derf datasets, and add raw/contour inputs (incl. Y4M) for the contour-video benchmark. Use when obtaining or configuring datasets, or before running the benchmark.
---

# datasets 管理

说明 infraredComp 中如何获取、配置与接入数据集。完整索引见 `datasets/README.md`(tracked)。

## 数据集位置

根目录可经 `INFRACOMP_DATASETS_DIR` 重定位(默认 `<repo>/datasets`),被以下位置**同步读取**:

- 路径常量:`benchmark/video/config.py`、`server/config.py`
- 下载脚本:`scripts/download_dataset.py`、`scripts/download_osu_color_thermal.py`、`scripts/download_xiph_natural.py`
- 业务代码:`benchmark/runner.py`、`benchmark/demo.py`

```bash
export INFRACOMP_DATASETS_DIR=/data/infrared   # 把大数据集放到仓库外(换盘/省备份)
```

约定:路径常量**单一来源**,勿在业务代码硬编码。

## 大数据集:D:/data + symlink(本机策略)

大数据集(BSDS500、SA-Co-VEval 等,**数十 MB ~ 数十 GB**)的**真实数据物理存放在本机 `D:/data/<dataset>/`(仓库外、不进 git)**,再以**目录 junction**软链接到项目 `datasets/<dataset>/`,代码统一经 `datasets/<dataset>/...` 访问。好处:数据不占仓库体积、换盘/备份只动 `D:/data`、多项目可共享同一份 `D:/data`。

- 存放:`D:/data/<dataset>/`(真实数据,git clone / modelscope 下载 / 手动均可)
- 软链接:`datasets/<dataset>/` → `D:/data/<dataset>/.../<data-root>`(Windows 用 **junction**,**不需管理员**;symlink /D 需开发者模式)
- git:`.gitignore` 忽略 `datasets/<dataset>/` 整个 junction 目录(git 不 traverse 真实数据),只在 `datasets/README.md` 记索引

```bash
# 1. 数据落到 D:/data(以 BSDS500 为例)
git clone --depth 1 https://github.com/BIDS/BSDS500.git /d/data/BSDS500

# 2. junction 到项目 datasets/(PowerShell,无需管理员)
powershell -Command "New-Item -ItemType Junction -Path 'D:\code\infraredComp\datasets\BSDS500' -Target 'D:\data\BSDS500\BSDS500\data'"

# 3. .gitignore 加 datasets/BSDS500/,datasets/README.md 记索引
```

**ModelScope 数据集**(如 SA-Co-VEval)用 modelscope python SDK 下到 `D:/data/`(库已装:`uv add modelscope`):
```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('facebook/SACo-VEval', cache_dir='D:/data/SACo-VEval/.cache')
# 或 CLI: uv run modelscope download --dataset facebook/SACo-VEval
```

已落地:
- `D:/data/BSDS500/`(BSDS500,389MB,images+groundTruth+ucm2)→ junction `datasets/BSDS500/` ✅
- SA-Co-VEval(`facebook/SACo-VEval`,~32GB)待下,已记 task,计划同样落 `D:/data/SACo-VEval/` + junction

## 三个数据域(勿混为一谈)

### 1. FLIR ADAS 1.3(红外图像,legacy 图像 benchmark)
- 来源:Kaggle [`deepnewbie/flir-thermal-images-dataset`](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset)
- 体积:~15GB;许可证:FLIR ADAS V1.3 License(见下载后的 PDF,仅用于研究评测)
- 获取:`uv run python scripts/download_dataset.py`(需 Kaggle 凭证;`--version N` 锁版本,`--force` 覆盖)
- 用途:`benchmark/runner.py` / `benchmark/demo.py` 读取 `thermal_16_bit` 下的 16-bit TIFF

### 2. OSU Color-Thermal(热红外视频,contour-video baseline 用)
- 来源:OTCBVS Dataset 03 — [vcipl-okstate.org](https://vcipl-okstate.org/pbvs/bench/Data/03/download.html)
- 模态:6 段热红外序列(`1a.zip`..`6a.zip`,仅 thermal 通道,color 故意跳过)→ 归一化 `seq1..6.mp4`
- 许可证:OTCBVS,仅研究/教学,使用须引用致谢(Davis & Sharma 2007)
- 获取:`uv run python scripts/download_osu_color_thermal.py`(幂等;`--force` 重下;`--dry-run` 预览)
- 一键 baseline:`uv run python scripts/run_osu_baseline.py`(下载 + 6 段评测 + 单一多序列 `results.json`)
- 路径:`${INFRACOMP_DATASETS_DIR}/raw/osu_color_thermal/seq{1..6}.mp4`
- **注意**:`vcipl-okstate.org` 对部分网络/IP 返回 403 "administrative rules"(非代码问题),若失败需换网络或改用其他数据集
- **OSU 在评测列表里被隐藏**(`_EVAL_HIDDEN_RAW = {"osu_color_thermal"}` 在 `server/routers/evaluation.py`),raw 目录保留但不展示;若想看 OSU 序列,先在 router 临时删掉这个集合。

### 3. raw / contour 视频(通用 contour-video 输入)
- raw = 用户自备输入,接受**视频文件**(`.mp4/.avi/.mov/.mkv/.m4v/.webm/.y4m`)或**帧目录**(`.png/.jpg/.jpeg/.tif/.tiff/.bmp`);位置不固定,运行时由 `--input` 指定
- contour = 阶段1 提取产物:**无损 `contour.mp4`(`libx264 -qp 0 -pix_fmt yuv420p`)+ `manifest.json`**,按方法分目录(`datasets/contour/<source>/<method>/`,canny/sobel 不互相覆盖),**不再保留 PNG 帧**。阶段2 从 contour.mp4 解码出临时帧跑评测,跑完清理。
- `--input` 接受视频或帧目录,目录会 glob VIDEO_EXTS(`benchmark/video/stage1_extract.py::expand_inputs`)
- 路径:`${INFRACOMP_DATASETS_DIR}/contour/<source>/<method>/contour.mp4`(如 `contour/akiyo_cif/canny/contour.mp4`)+ 同目录 `manifest.json`
- 迁移旧目录(从 PNG 帧格式转 contour.mp4 格式):`ffmpeg -y -framerate <fps> -i frame_%06d.png -vf pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black -c:v libx264 -qp 0 -pix_fmt yuv420p contour.mp4` → 删 PNG → manifest 加 `"video_path"`。

### 4. Xiph derf CIF(自然视频,contour-video baseline 用)
- 来源:[Xiph derf collection](https://media.xiph.org/video/derf/y4m/)(公开测试媒体)
- 模态:6 段 CIF(352×288,~30fps)Y4M 序列(`akiyo_cif / bus_cif / city_cif / flower_cif / foreman_cif / mobile_cif`)
- 许可证:Xiph derf 测试媒体,公开测试用途
- 获取:`uv run python scripts/download_xiph_natural.py`(幂等;`--force` 重下;`--dry-run` 预览)
- 一键 baseline:`uv run python scripts/run_natural_baseline.py`(下载 + 6 段评测 + **独立文件** `results/video/xiph_cif.json`)
- 路径:`${INFRACOMP_DATASETS_DIR}/raw/xiph_cif/<name>_cif.y4m`
- **真实帧数(数 y4m 的 FRAME 标记或 ffprobe `nb_read_packets`)**:
  - akiyo 300 / bus 150 / city 300 / flower 250 / foreman 300 / mobile 300(5–10s @30fps)
- **下载脚本目前不写 raw manifest 的 `frame_count`**(只填 fps/size_bytes);运行后若数据集页序列头显示 "0 帧",用 ffprobe 回填:
  ```bash
  FP=.venv/Lib/site-packages/static_ffmpeg/bin/win32/ffprobe.exe
  # conda compression python(无需 uv 网络):
  "$USERPROFILE/.conda/envs/compression/python.exe" -c "
  import json, subprocess
  from pathlib import Path
  mf=Path('datasets/raw/xiph_cif/manifest.json')
  m=json.loads(mf.read_text(encoding='utf-8'))
  for s in m['sequences']:
      n=subprocess.run(['$FP','-v','error','-select_streams','v:0','-count_packets',
          '-show_entries','stream=nb_read_packets','-of','csv=p=0',s['file']],
          capture_output=True,text=True).stdout.strip()
      s['frame_count']=int(n) if n.isdigit() else 0
  mf.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
  "
  ```
- 用途:自然视频方向性 baseline;每条 run 携带 `dataset="Xiph-CIF-natural"`,与 OSU/demo 共存于 `results/video/`

## 按数据集区分结果(多数据集共存)

每个 baseline 脚本写**独立 results 文件**(不互相覆盖),evaluation 模块聚合读:

| 数据集 | baseline 脚本 | 输出文件 | envelope `dataset` | per-run `dataset` |
|---|---|---|---|---|
| demo(a3273aa 提交) | `python -m benchmark.video --input demo.mp4` | `results/video/results.json` | (无) | `"default"`(从文件名推断) |
| OSU | `scripts/run_osu_baseline.py` | `results/video/results.json`(覆盖 demo) | `"OSU Color-Thermal (OTCBVS Dataset 03)"` | `"OSU Color-Thermal (OTCBVS Dataset 03)"` |
| Xiph CIF | `scripts/run_natural_baseline.py` | `results/video/xiph_cif.json`(**独立**) | `"Xiph-CIF-natural"` | `"Xiph-CIF-natural"` |
| xiph 全量(脚本入口) | `scripts/run_all_subprocess.py`(跑 stage1+stage2 with `--codecs`) | `results/video/xiph_cif.json` | `"Xiph-CIF-natural"` | `"Xiph-CIF-natural"` |

> 注意:`run_natural_baseline.py` 与 `run_all_subprocess.py` 都写 `results/video/xiph_cif.json`,**会互覆盖**。后者用 `(sequence, codec, crf)` 作 run key 增量跳过(改管线后要刷新指标需先 `rm` 旧 JSON)。

- **Per-run dataset 字段**:`VideoCompressionResult.dataset`(data.py),由 `benchmark_codec(..., dataset=...)` 填,`run_benchmark` 透传
- **Evaluation 聚合**:`server/routers/evaluation.py::_load_results` 扫 `results/video/*.json`,每 run `setdefault("dataset", envelope 或文件名)`
- **API 过滤**:`GET /api/evaluation/results?dataset=Xiph-CIF-natural` 按 `run.dataset` 过滤(而非 `sequence_name` 别名)
- **前端两结果页**:`/evaluation/speed`(SpeedResults,视频网格,filter 含数据集/方法/序列/codec/crf)+ `/evaluation/formal`(FormalResults,2-3 演示视频 + per-(codec,crf) 16 行平均表)。旧 `/evaluation/results`(EvalResults per-run 表)废弃,重定向到 `/evaluation/formal`。两页均有"数据集"选择器(从 runs distinct `dataset_name` 构建)。
- **回退兼容**:无 `dataset` 的旧 results 文件(`results.json`)被赋 `"default"`

## 接入新数据集跑 baseline

```bash
# 任意视频文件
uv run python -m benchmark.video --input /path/to/video.mp4 --method canny --crfs 18,23,28,33

# 帧目录(如 FLIR thermal_8_bit) — 注意:帧目录输入会先生成 contour.mp4 再评测
uv run python -m benchmark.video --input ${INFRACOMP_DATASETS_DIR}/FLIR_ADAS_1_3/video/thermal_8_bit --method canny --extract-only

# Y4M(自然视频,如 Xiph)
uv run python -m benchmark.video --input ${INFRACOMP_DATASETS_DIR}/raw/xiph_cif/akiyo_cif.y4m --method canny

# 整个 Xiph 目录(自动 glob *.y4m)
uv run python -m benchmark.video --input ${INFRACOMP_DATASETS_DIR}/raw/xiph_cif --method canny

# 仅阶段 2(复用已有 contour 产物,按方法分子目录)
uv run python -m benchmark.video --input ${INFRACOMP_DATASETS_DIR}/contour/demo/canny --skip-extract

# speed run(少量视频子集,**全帧不截断**;speed 靠 --sequences 加速,不靠 --frames)
uv run python scripts/run_natural_baseline.py --sequences akiyo_cif,bus_cif --codecs x264 --crfs 23

# formal test(全量,全帧;不传 --frames)
uv run python scripts/run_natural_baseline.py --codecs x264,x265,vp9
```

两模式评测逻辑统一(一套 stage1+stage2),差异只在 `--sequences` 子集 + 展示页(speed 网格 / formal 平均)。详见 `.claude/skills/evaluation/SKILL.md` §3。**默认不截断帧**;speed 加速靠 `--sequences` 子集,不靠 `--frames N` 限帧。

## ffmpeg 来源

`benchmark/video/ffmpeg_util.py` 通过三层 fallback 解析 ffmpeg/ffprobe,无需系统安装:

1. `INFRACOMP_FFMPEG_BIN` 环境变量(目录或 exe 全路径)
2. `shutil.which` 在 PATH 查找
3. **`static-ffmpeg` pip 包**(`uv add static-ffmpeg`),内置 ffmpeg+ffprobe 静态二进制(自动下载到 `.venv/Lib/site-packages/static_ffmpeg/bin/`)

视频编码默认 codec:`x264 / x265 / svtav1 / vp9`(`benchmark/video/codecs/`);static-ffmpeg win32 build **不含 libsvtav1**,baseline 自动跳过 svtav1,实际跑 x264/x265/vp9 三 codec。

## git 策略

`datasets/` 下按媒体扩展名忽略(`*.mp4 *.png *.jpg *.tif ...`),不依赖固定目录名;`datasets/README.md`、`datasets/manifest.json`、`datasets/contour/*/manifest.json` 等小文件被追踪。**运行产物**(`results/video/xiph_cif.json`、`bitstreams/`、`recon/`、`source/`、`contour_mp4/`、`_raw_frames/`、`*.png` 帧)也不进 git,各 baseline 脚本每次跑覆盖或写独立文件。

## Windows 已知坑

- **GBK 编码**:`print` 含 `•` / `↓` 等 unicode 在 Windows GBK console 崩 → 下载脚本已用 ASCII 字符;baseline 跑前可设 `PYTHONUTF8=1` 兜底
- **证书吊销**:`vcipl-okstate.org` 走 Windows schannel + OCSP,可能返回 `CRYPT_E_REVOCATION_OFFLINE`;Xiph 不需要(curl 已加 `--ssl-no-revoke` 兜底)
- **403 administrative rules**:`vcipl-okstate.org` 整站对部分网络拒绝(非代码问题,需换网络/VPN 或换数据集)
- **缺失 ffmpeg**:`uv add static-ffmpeg` 一键解决,无需系统安装
- **uv 网络炸(代理 10061)**:`uv run` 走 pypi 镜像(tuna)解析依赖失败时,直接用 `.venv/Scripts/python.exe -m <module>` 绕过 uv,或用 conda compression env `$USERPROFILE/.conda/envs/compression/python.exe`(无需 uv)

## 约定速查

- 路径常量:`benchmark/video/config.py` 与 `server/config.py` 是单一来源;**勿在业务代码硬编码 `datasets/`**
- 后端只读:`/api/benchmark/runs` 列 `contour/` 目录 + manifest;`/api/evaluation/*` 聚合多数据集
- ffmpeg 统一 `-pix_fmt yuv420p` 编码、`gray` 解码(详见 AGENTS.md)
- 下载幂等:已下文件 `--force` 才重下;`manifest.json` 记录每条 seq 元数据
- **contour 产物是 `contour.mp4` 不是 PNG**:所有 contour 目录、stage2 临时物化、`_ensure_contour_video` 输入、manifest `video_path` 字段都指 `contour.mp4`(详见 evaluation skill §1-2)