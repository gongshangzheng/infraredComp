<template>
  <div class="page-container eval-results">
    <!-- 筛选：方法 / 序列 / codec -->
    <n-card size="small">
      <n-space align="center" size="small" wrap>
        <span class="lbl">数据集</span>
        <n-select v-model:value="filters.dataset" :options="datasetOptions" placeholder="全部" clearable size="small" style="width: 150px" />
        <span class="lbl">提取方法</span>
        <n-select v-model:value="filters.method" :options="methodOptions" placeholder="全部方法" clearable size="small" style="width: 130px" />
        <span class="lbl">序列</span>
        <n-select v-model:value="filters.sequence" :options="sequenceOptions" placeholder="全部" clearable size="small" style="width: 140px" />
        <span class="lbl">codec</span>
        <n-select v-model:value="filters.codec" :options="codecOptions" placeholder="全部" clearable size="small" style="width: 120px" />
        <n-button size="small" @click="load">刷新</n-button>
      </n-space>
    </n-card>

    <!-- 常驻大播放框（不点开弹窗；src 仅在选择时赋值 + preload=none 按需加载） -->
    <n-card size="small" class="player-card">
      <template #header>
        <div class="flex-between">
          <h3>输出视频</h3>
          <span class="hint">{{ playerTitle || '选择下方任意结果/输出以播放' }}</span>
        </div>
      </template>
      <div class="player-wrap">
        <video v-if="playerSrc" :src="playerSrc" controls preload="none" playsinline />
        <div v-else class="player-placeholder">选择下方任意一条结果或输出文件，视频将在此处按需加载播放</div>
      </div>
      <n-descriptions v-if="currentRun" :column="4" size="small" label-placement="left" bordered style="margin-top: 12px">
        <n-descriptions-item label="序列">{{ currentRun.sequence_name }}</n-descriptions-item>
        <n-descriptions-item label="codec">{{ currentRun.codec }}</n-descriptions-item>
        <n-descriptions-item label="CRF">{{ currentRun.crf }}</n-descriptions-item>
        <n-descriptions-item label="方法">{{ currentRun.method }}</n-descriptions-item>
        <n-descriptions-item label="PSNR">{{ fmt(currentRun.psnr) }} dB</n-descriptions-item>
        <n-descriptions-item label="SSIM">{{ fmt(currentRun.ssim) }}</n-descriptions-item>
        <n-descriptions-item label="码率">{{ fmt(currentRun.bitrate_kbps) }} kb/s</n-descriptions-item>
        <n-descriptions-item label="压缩比">{{ fmt(currentRun.compression_ratio) }}×</n-descriptions-item>
      </n-descriptions>
    </n-card>

    <!-- 评测结果表 -->
    <n-card size="small" title="评测结果" style="margin-top: 12px">
      <n-spin :show="loading">
        <n-data-table v-if="filteredResults.length" :columns="resultColumns" :data="filteredResults" :bordered="false" size="small" striped />
        <EmptyState v-else description="暂无结果。运行评测后此处列出各 codec×CRF 的指标。" />
      </n-spin>
    </n-card>

    <!-- 方法对比矩阵（同一操作点不同方法的 baseline） -->
    <n-card size="small" title="方法对比 — 不同轮廓提取方法下的 baseline" style="margin-top: 12px">
      <template #header-extra><span class="hint">点单元格播放该方法输出</span></template>
      <table v-if="matrixRows.length && methods.length" class="matrix">
        <thead>
          <tr><th>序列</th><th>codec</th><th>CRF</th><th v-for="m in methods" :key="m">{{ m }}</th></tr>
        </thead>
        <tbody>
          <tr v-for="r in matrixRows" :key="r.key">
            <td>{{ r.sequence }}</td><td>{{ r.codec }}</td><td>{{ r.crf }}</td>
            <td v-for="m in methods" :key="m" class="cell" :class="{ active: currentRun && currentRun.method === m && currentRun.sequence_name === r.sequence && currentRun.codec === r.codec && currentRun.crf === r.crf }">
              <template v-if="r.byMethod[m]">
                <div class="cell-metrics">
                  <span title="PSNR">P {{ fmt(r.byMethod[m].psnr) }}</span>
                  <span class="dim" title="码率 kb/s">{{ fmt(r.byMethod[m].bitrate_kbps) }} kb/s</span>
                </div>
                <n-button size="tiny" type="primary" secondary @click="play(r.byMethod[m])">播放</n-button>
              </template>
              <span v-else class="dim">—</span>
            </td>
          </tr>
        </tbody>
      </table>
      <EmptyState v-else description="跑多个轮廓提取方法（canny/sobel）的 baseline 后，此处按方法对比 PSNR/码率 + 播放。" />
    </n-card>

    <!-- 输出文件 -->
    <n-card size="small" title="输出文件（bitstreams 压缩码流 / recon 重建帧）" style="margin-top: 12px">
      <n-spin :show="outputsLoading">
        <n-data-table v-if="outputs.length" :columns="outputColumns" :data="outputs" :bordered="false" size="small" striped />
        <EmptyState v-else description="暂无输出文件。评测产物在 results/video/{bitstreams,recon}/。" />
      </n-spin>
    </n-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, h } from 'vue'
import { NCard, NSpin, NSpace, NSelect, NButton, NDataTable, NDescriptions, NDescriptionsItem, useMessage } from 'naive-ui'
import EmptyState from '../../components/common/EmptyState.vue'
import { getEvalResults, getMethods, listOutputs, getOutputUrl } from '../../api/evaluation'

const message = useMessage()
const loading = ref(false)
const outputsLoading = ref(false)
const results = ref([])
const methods = ref([])
const outputs = ref([])
const filters = ref({ dataset: null, method: null, sequence: null, codec: null })

// 常驻播放器状态
const playerSrc = ref('')
const playerTitle = ref('')
const currentRun = ref(null)

const methodOptions = computed(() => methods.value.map(m => ({ label: m, value: m })))
const sequenceOptions = computed(() => [...new Set(results.value.map(r => r.sequence_name))].map(s => ({ label: s, value: s })))
const codecOptions = computed(() => [...new Set(results.value.map(r => r.codec))].map(c => ({ label: c, value: c })))
const datasetOptions = computed(() => [...new Set(results.value.map(r => r.dataset_name).filter(Boolean))].map(d => ({ label: d, value: d })))

const filteredResults = computed(() => {
  let list = results.value
  if (filters.value.dataset) list = list.filter(r => r.dataset_name === filters.value.dataset)
  if (filters.value.method) list = list.filter(r => r.method === filters.value.method)
  if (filters.value.sequence) list = list.filter(r => r.sequence_name === filters.value.sequence)
  if (filters.value.codec) list = list.filter(r => r.codec === filters.value.codec)
  return list
})

// 方法对比矩阵：行 = (sequence, codec, crf) 操作点，列 = 方法
const matrixRows = computed(() => {
  let list = results.value
  if (filters.value.dataset) list = list.filter(r => r.dataset_name === filters.value.dataset)
  if (filters.value.sequence) list = list.filter(r => r.sequence_name === filters.value.sequence)
  if (filters.value.codec) list = list.filter(r => r.codec === filters.value.codec)
  const map = new Map()
  for (const r of list) {
    const key = `${r.sequence_name}|${r.codec}|${r.crf}`
    if (!map.has(key)) map.set(key, { key, sequence: r.sequence_name, codec: r.codec, crf: r.crf, byMethod: {} })
    map.get(key).byMethod[r.method] = r
  }
  return [...map.values()].sort((a, b) => a.sequence.localeCompare(b.sequence) || a.codec.localeCompare(b.codec) || a.crf - b.crf)
})

const resultColumns = computed(() => [
  { title: '序列', key: 'sequence_name' },
  { title: '方法', key: 'method', width: 80 },
  { title: 'codec', key: 'codec', width: 80 },
  { title: 'CRF', key: 'crf', width: 60 },
  { title: 'PSNR', key: 'psnr', width: 80, render: (r) => fmt(r.psnr) },
  { title: 'SSIM', key: 'ssim', width: 70, render: (r) => fmt(r.ssim) },
  { title: '码率(kb/s)', key: 'bitrate_kbps', width: 100, render: (r) => fmt(r.bitrate_kbps) },
  { title: '压缩比', key: 'compression_ratio', width: 80, render: (r) => fmt(r.compression_ratio) },
  {
    title: '操作', key: 'actions', width: 110,
    render: (r) => r.output_video
      ? h(NButton, { size: 'small', type: 'primary', secondary: true, onClick: () => play(r) }, { default: () => '播放' })
      : h('span', { style: 'color: var(--color-text-dim)' }, '—'),
  },
])

const outputColumns = [
  { title: '路径', key: 'path' },
  { title: '类型', key: 'ext', width: 70 },
  { title: '大小', key: 'size_bytes', width: 100, render: (r) => fmtSize(r.size_bytes) },
  {
    title: '操作', key: 'actions', width: 100,
    render: (r) => r.is_video
      ? h(NButton, { size: 'small', type: 'primary', secondary: true, onClick: () => playOutput(r) }, { default: () => '播放' })
      : h('span', { style: 'color: var(--color-text-dim)' }, '—'),
  },
]

function fmt(v) { return (v == null || isNaN(v)) ? '-' : Number(v).toFixed(2) }
function fmtSize(b) {
  if (!b) return '-'
  const u = ['B', 'KB', 'MB', 'GB']; let i = 0, v = b
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(1)} ${u[i]}`
}

// 选择一条结果 -> 常驻播放框加载（preload=none：按 play 才取字节）
function play(run) {
  if (!run.output_video) { message.warning('该结果暂无输出视频'); return }
  currentRun.value = run
  playerSrc.value = getOutputUrl(run.output_video)
  playerTitle.value = `${run.sequence_name} · ${run.method} · ${run.codec} · crf${run.crf}`
}
function playOutput(o) {
  currentRun.value = null
  playerSrc.value = getOutputUrl(o.path)
  playerTitle.value = o.path
}

async function load() {
  loading.value = true
  outputsLoading.value = true
  try {
    const [res, meth, out] = await Promise.all([
      getEvalResults().catch(() => []),
      getMethods().catch(() => ({ methods: [] })),
      listOutputs().catch(() => ({ outputs: [] })),
    ])
    results.value = res || []
    methods.value = meth?.methods || []
    outputs.value = out?.outputs || []
  } catch (e) { message.error('加载失败') }
  loading.value = false
  outputsLoading.value = false
}

onMounted(load)
</script>

<style scoped lang="scss">
.eval-results { display: flex; flex-direction: column; gap: 0; }
.flex-between { display: flex; justify-content: space-between; align-items: center; }
.lbl { font-size: 13px; color: var(--color-text-secondary); }
.hint { font-size: 12px; color: var(--color-text-dim); }

.player-card .player-wrap {
  display: flex; justify-content: center; align-items: center;
  background: #000; border-radius: 8px; overflow: hidden; min-height: 280px;
}
.player-card video { width: 100%; max-height: 60vh; display: block; }
.player-placeholder {
  color: var(--color-text-dim); padding: 48px; font-size: 14px; text-align: center;
}

.matrix { width: 100%; border-collapse: collapse; font-size: 13px; }
.matrix th, .matrix td { border: 1px solid var(--color-border-light); padding: 8px; text-align: center; }
.matrix th { background: var(--color-elevated); }
.matrix td.cell.active { background: var(--color-selected); }
.matrix .cell-metrics { display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; font-size: 12px; }
.matrix .dim { color: var(--color-text-dim); }
</style>
