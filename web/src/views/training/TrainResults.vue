<template>
  <div class="page-container train-results">
    <!-- 筛选 -->
    <n-card size="small">
      <n-space align="center" size="small" wrap>
        <span class="lbl">模型</span>
        <n-select v-model:value="filters.model" :options="modelOptions" placeholder="全部" clearable size="small" style="width: 160px" />
        <span class="lbl">数据集</span>
        <n-select v-model:value="filters.dataset" :options="datasetOptions" placeholder="全部" clearable size="small" style="width: 140px" />
        <span class="lbl">状态</span>
        <n-select v-model:value="filters.status" :options="statusOptions" placeholder="全部" clearable size="small" style="width: 100px" />
        <n-button size="small" @click="load">刷新</n-button>
      </n-space>
    </n-card>

    <!-- 常驻 loss 曲线区：所有 run 叠加显示，可切指标 -->
    <n-card size="small" class="curve-card">
      <template #header>
        <div class="flex-between">
          <h3>训练曲线（所有 run 叠加）</h3>
          <n-space size="small" align="center">
            <span class="hint">指标</span>
            <n-select v-model:value="curveMetric" :options="metricOptions" size="small" style="width: 110px" />
          </n-space>
        </div>
      </template>
      <div v-if="curveOption" class="curve-wrap">
        <v-chart class="curve" :option="curveOption" autoresize />
      </div>
      <div v-else class="curve-placeholder">暂无 run（在「训练运行」页启动训练后曲线在此叠加显示）</div>
    </n-card>

    <!-- 训练结果列表（run + checkpoint 合并） -->
    <n-card size="small" title="训练结果列表" style="margin-top: 12px">
      <n-spin :show="loading">
        <n-data-table v-if="filteredRuns.length" :columns="runColumns" :data="filteredRuns" :bordered="false" size="small" striped :scroll-x="1100" />
        <EmptyState v-else description="暂无训练 run。在「训练运行」页启动训练后，结果 + 曲线在此。" />
      </n-spin>
    </n-card>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed, h, watch } from 'vue'
import { NCard, NSpin, NSpace, NSelect, NButton, NDataTable, useMessage } from 'naive-ui'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import EmptyState from '../../components/common/EmptyState.vue'
import { getTrainRuns, getTrainOutputUrl } from '../../api/training'
import { useRouter } from 'vue-router'

const router = useRouter()

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent])

const message = useMessage()
const loading = ref(false)
const runs = ref([])
const filters = ref({ model: null, dataset: null, status: null })
const currentRun = ref(null)
const currentRunId = ref(null)

// 跨浏览器刷新(F5)持久化筛选 + 选中 run（下拉框刷新不清空）
const STORE_KEY = 'infracomp:train-results'
function persistState() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ filters: filters.value, currentRunId: currentRunId.value }))
  } catch { /* localStorage 不可用时静默 */ }
}
function restoreState() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE_KEY) || '{}')
    if (s.filters) filters.value = { model: null, dataset: null, status: null, ...s.filters }
    if (s.currentRunId) currentRunId.value = s.currentRunId
  } catch { /* ignore */ }
}
watch([filters, () => currentRunId.value], persistState, { deep: true })
restoreState()

// 实时曲线：3s 轮询 getTrainRuns，同步选中 run 的 loss_series（曲线随 epoch 增长）
const RUNNING = new Set(['running', 'started'])
let pollTimer = null
const isRunning = (r) => !!r && RUNNING.has(r.status)

function sortRuns(list) {
  return [...list].sort((a, b) => (b.started_at || '').localeCompare(a.started_at || ''))
}

function syncCurrent() {
  if (!currentRunId.value) return
  const r = runs.value.find(x => x.id === currentRunId.value)
  if (r) currentRun.value = r
}

function startPolling() {
  if (pollTimer) return
  pollTimer = setInterval(async () => {
    const res = await getTrainRuns().catch(() => null)
    if (res?.runs) {
      runs.value = sortRuns(res.runs)
      syncCurrent()
    }
    // 全部 run 结束：停轮询（latest/best 已随 /runs 返回）
    if (!runs.value.some(isRunning)) stopPolling()
  }, 3000)
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

// 选项列表始终包含当前已选中值：即使 runs 变动导致该值暂时不在列表里，
// 下拉框也不会被清空（轮询/刷新不清空选中）。
const withSelected = (values, selected) => {
  if (selected && !values.includes(selected)) values.push(selected)
  return values.map(v => ({ label: v, value: v }))
}
const modelOptions = computed(() => withSelected([...new Set(runs.value.map(r => r.model))], filters.value.model))
const datasetOptions = computed(() => withSelected([...new Set(runs.value.map(r => r.dataset))], filters.value.dataset))
const statusOptions = computed(() => withSelected([...new Set(runs.value.map(r => r.status))], filters.value.status))

const filteredRuns = computed(() => {
  let list = runs.value
  if (filters.value.model) list = list.filter(r => r.model === filters.value.model)
  if (filters.value.dataset) list = list.filter(r => r.dataset === filters.value.dataset)
  if (filters.value.status) list = list.filter(r => r.status === filters.value.status)
  return list
})

const curveTitle = computed(() => {
  const r = currentRun.value
  if (!r) return ''
  const tail = isRunning(r) ? ' · 训练中，实时刷新…' : ''
  return `${r.model} · ${r.dataset} · ${r.id}${tail}`
})

const curveMetric = ref('loss')
const metricOptions = [
  { label: 'loss', value: 'loss' },
  { label: 'PSNR(dB)', value: 'psnr' },
  { label: 'bpp', value: 'bpp' },
]

// 所有（筛选后）run 叠加显示同一指标；x=epoch（value 轴，不同 run 不同长度可对齐）
const curveOption = computed(() => {
  const metric = curveMetric.value
  const series = filteredRuns.value
    .filter(r => r.loss_series?.some(p => p[metric] != null))
    .map(r => ({
      name: r.id,
      type: 'line',
      smooth: true,
      showSymbol: false,
      data: r.loss_series.filter(p => p[metric] != null).map(p => [p.epoch, p[metric]]),
    }))
  if (!series.length) return null
  const yname = metric === 'psnr' ? 'PSNR(dB)' : metric
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: series.map(s => s.name), type: 'scroll', top: 0 },
    grid: { top: 40 },
    xAxis: { type: 'value', name: 'epoch', minInterval: 1 },
    yAxis: { type: 'value', name: yname, scale: true },
    series,
  }
})

const runColumns = computed(() => [
  { title: 'ID', key: 'id', width: 230, ellipsis: { tooltip: true } },
  { title: '模型', key: 'model', width: 80 },
  { title: '数据集', key: 'dataset', width: 120, ellipsis: { tooltip: true } },
  { title: 'epochs', key: 'epochs', width: 80, render: (r) => `${r.loss_series?.length || 0}/${r.epochs ?? '-'}` },
  { title: 'latest(test)', key: 'latest', width: 150, render: (r) => fmtLatest(r.latest) },
  { title: 'best(test)', key: 'best', width: 130, render: (r) => fmtBest(r.best) },
  { title: '状态', key: 'status', width: 80 },
  { title: '时间', key: 'started_at', width: 100, render: (r) => r.started_at?.split('T')[0] || '-' },
  {
    title: '操作', key: 'actions', width: 180, fixed: 'right',
    render: (r) => h('div', { style: 'display:flex;gap:6px' }, [
      h(NButton, { size: 'small', type: 'primary', secondary: true, onClick: () => router.push(`/training/results/runs/${r.id}`) }, { default: () => '详情' }),
      h(NButton, { size: 'small', secondary: true, disabled: !r.has_best, onClick: () => downloadCkpt(r, 'best') }, { default: () => 'best' }),
      h(NButton, { size: 'small', secondary: true, disabled: !r.has_latest, onClick: () => downloadCkpt(r, 'latest') }, { default: () => 'latest' }),
    ]),
  },
])

function fmt(v) { return (v == null || isNaN(v)) ? '-' : Number(v).toFixed(4) }
function fmtLatest(la) {
  if (!la?.test) return '-'
  return `ep${la.epoch} ${la.test.psnr != null ? Number(la.test.psnr).toFixed(2) + 'dB' : ''} ${la.test.bpp != null ? Number(la.test.bpp).toFixed(3) + 'bpp' : ''}`
}
function fmtBest(b) {
  if (!b?.test) return '-'
  return `ep${b.epoch} ${b.test.psnr != null ? Number(b.test.psnr).toFixed(2) + 'dB' : ''}`
}

function selectRun(r) {
  currentRunId.value = r?.id ?? null
  currentRun.value = r ?? null
  if (r && !r.loss_series?.length) message.info('该 run 暂无 loss_series（训练未开始或未记录）')
  if (isRunning(r)) startPolling()
}

function downloadCkpt(run, kind) {
  const rid = run.id
  const path = kind === 'best' ? `checkpoints/${rid}.best.pth` : `checkpoints/${rid}.pth`
  const a = document.createElement('a')
  a.href = getTrainOutputUrl(path)
  a.download = kind === 'best' ? `${rid}.best.pth` : `${rid}.pth`
  a.click()
}

async function load() {
  loading.value = true
  try {
    const runsRes = await getTrainRuns().catch(() => ({ runs: [] }))
    runs.value = sortRuns(runsRes?.runs || [])
    // 自动选中最新一条 run（开页面即可看到正在跑的曲线）
    if (!currentRunId.value && runs.value.length) {
      selectRun(runs.value[0])
    } else {
      syncCurrent()
    }
    if (runs.value.some(isRunning)) startPolling()
  } catch (e) { message.error('加载失败') }
  loading.value = false
}

onMounted(load)
onUnmounted(stopPolling)
</script>

<style scoped lang="scss">
.train-results { display: flex; flex-direction: column; }
.flex-between { display: flex; justify-content: space-between; align-items: center; }
.lbl { font-size: 13px; color: var(--color-text-secondary); }
.hint { font-size: 12px; color: var(--color-text-dim); }
.curve-card .curve-wrap { background: var(--color-elevated); border-radius: 8px; padding: 8px; }
.curve { height: 320px; width: 100%; }
.curve-placeholder { color: var(--color-text-dim); padding: 48px; text-align: center; font-size: 14px; }
</style>
