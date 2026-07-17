<template>
  <div class="page-container train-results">
    <!-- 筛选 -->
    <n-card size="small">
      <n-space align="center" size="small" wrap>
        <span class="lbl">模型</span>
        <n-select v-model:value="filters.model" :options="modelOptions" placeholder="全部" clearable size="small" style="width: 160px" @update:value="reload" />
        <span class="lbl">数据集</span>
        <n-select v-model:value="filters.dataset" :options="datasetOptions" placeholder="全部" clearable size="small" style="width: 140px" @update:value="reload" />
        <span class="lbl">状态</span>
        <n-select v-model:value="filters.status" :options="statusOptions" placeholder="全部" clearable size="small" style="width: 100px" @update:value="reload" />
        <n-button size="small" @click="reload">刷新</n-button>
      </n-space>
    </n-card>

    <!-- 常驻 loss 曲线区：当前页所有 run 叠加显示，可切指标 -->
    <n-card size="small" class="curve-card">
      <template #header>
        <div class="flex-between">
          <h3>训练曲线（当前页 run 叠加）</h3>
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

    <!-- 测试曲线（test_metrics 叠加） -->
    <n-card size="small" class="curve-card" style="margin-top: 12px">
      <template #header>
        <div class="flex-between">
          <h3>测试曲线（当前页 run 叠加）</h3>
          <n-space size="small" align="center">
            <span class="hint">指标</span>
            <n-select v-model:value="testMetric" :options="metricOptions" size="small" style="width: 110px" />
          </n-space>
        </div>
      </template>
      <div v-if="testCurveOption" class="curve-wrap">
        <v-chart class="curve" :option="testCurveOption" autoresize />
      </div>
      <div v-else class="curve-placeholder">暂无测试数据（eval 每 eval-every epoch 出 test_metrics）</div>
    </n-card>

    <!-- 训练结果列表（run + checkpoint 合并） -->
    <n-card size="small" title="训练结果列表" style="margin-top: 12px">
      <n-spin :show="loading">
        <n-data-table
          v-if="runs.length"
          :columns="runColumns"
          :data="runs"
          :bordered="false"
          size="small"
          striped
          :scroll-x="1100"
          remote
          :pagination="pagination"
          :row-key="(r) => r.id"
          @update:page="onPageChange"
        />
        <EmptyState v-else description="暂无训练 run。在「训练运行」页启动训练后，结果 + 曲线在此。" />
      </n-spin>
    </n-card>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed, h } from 'vue'
import { NCard, NSpin, NSpace, NSelect, NButton, NDataTable, useMessage } from 'naive-ui'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent } from 'echarts/components'
import EmptyState from '../../components/common/EmptyState.vue'
import { getTrainRuns, getTrainOutputUrl } from '../../api/training'
import { useRouter } from 'vue-router'

const router = useRouter()

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent])

const message = useMessage()
const loading = ref(false)
const runs = ref([])

const PAGE_SIZE = 50
const pagination = ref({
  page: 1,
  pageSize: PAGE_SIZE,
  itemCount: 0,
  showSizePicker: false,
  prefix: ({ itemCount }) => `共 ${itemCount} 条`,
})

const filters = ref({ model: null, dataset: null, status: null })

const RUNNING = new Set(['running', 'started'])
let pollTimer = null
const isRunning = (r) => !!r && RUNNING.has(r.status)

function sortRuns(list) {
  return [...list].sort((a, b) => (b.started_at || '').localeCompare(a.started_at || ''))
}

const withSelected = (values, selected) => {
  if (selected && !values.includes(selected)) values.push(selected)
  return values.map(v => ({ label: v, value: v }))
}
const modelOptions = computed(() => withSelected([...new Set(runs.value.map(r => r.model))], filters.value.model))
const datasetOptions = computed(() => withSelected([...new Set(runs.value.map(r => r.dataset))], filters.value.dataset))
const statusOptions = computed(() => withSelected([...new Set(runs.value.map(r => r.status))], filters.value.status))

const curveMetric = ref('loss')
const metricOptions = [
  { label: 'loss', value: 'loss' },
  { label: 'PSNR(dB)', value: 'psnr' },
  { label: 'bpp', value: 'bpp' },
]

const curveOption = computed(() => {
  const metric = curveMetric.value
  const series = runs.value
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
    grid: { top: 40, bottom: 50 },
    xAxis: { type: 'value', name: 'epoch', minInterval: 1 },
    yAxis: { type: 'value', name: yname, scale: true },
    dataZoom: [
      { type: 'slider', xAxisIndex: 0, bottom: 8 },
      { type: 'inside', xAxisIndex: 0 },
    ],
    series,
  }
})

const testMetric = ref('psnr')
const testCurveOption = computed(() => {
  const metric = testMetric.value
  const series = runs.value
    .filter(r => r.test_metrics?.some(p => p[metric] != null))
    .map(r => ({
      name: r.id,
      type: 'line',
      smooth: true,
      showSymbol: false,
      data: r.test_metrics.filter(p => p[metric] != null).map(p => [p.epoch, p[metric]]),
    }))
  if (!series.length) return null
  const yname = metric === 'psnr' ? 'PSNR(dB)' : metric
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: series.map(s => s.name), type: 'scroll', top: 0 },
    grid: { top: 40, bottom: 50 },
    xAxis: { type: 'value', name: 'epoch', minInterval: 1 },
    yAxis: { type: 'value', name: yname, scale: true },
    dataZoom: [
      { type: 'slider', xAxisIndex: 0, bottom: 8 },
      { type: 'inside', xAxisIndex: 0 },
    ],
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

function downloadCkpt(run, kind) {
  const rid = run.id
  const path = kind === 'best' ? `checkpoints/${rid}.best.pth` : `checkpoints/${rid}.pth`
  const a = document.createElement('a')
  a.href = getTrainOutputUrl(path)
  a.download = kind === 'best' ? `${rid}.best.pth` : `${rid}.pth`
  a.click()
}

async function fetchPage(page) {
  const offset = (page - 1) * PAGE_SIZE
  const params = { offset, limit: PAGE_SIZE }
  if (filters.value.model) params.model = filters.value.model
  if (filters.value.dataset) params.dataset = filters.value.dataset
  if (filters.value.status) params.status = filters.value.status
  return getTrainRuns(params)
}

async function reload() {
  loading.value = true
  try {
    pagination.value.page = 1
    const res = await fetchPage(1)
    runs.value = sortRuns(res?.runs || [])
    pagination.value.itemCount = res?.total ?? runs.value.length
    if (runs.value.some(isRunning)) startPolling()
    else stopPolling()
  } catch { message.error('加载失败') }
  loading.value = false
}

async function onPageChange(page) {
  loading.value = true
  try {
    pagination.value.page = page
    const res = await fetchPage(page)
    runs.value = sortRuns(res?.runs || [])
    pagination.value.itemCount = res?.total ?? runs.value.length
  } catch { message.error('加载失败') }
  loading.value = false
}

function startPolling() {
  if (pollTimer) return
  pollTimer = setInterval(async () => {
    const res = await fetchPage(pagination.value.page).catch(() => null)
    if (res?.runs) {
      runs.value = sortRuns(res.runs)
      pagination.value.itemCount = res.total ?? pagination.value.itemCount
    }
    if (!runs.value.some(isRunning)) stopPolling()
  }, 3000)
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

onMounted(reload)
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
