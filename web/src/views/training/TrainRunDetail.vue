<template>
  <div class="page-container train-run-detail">
    <n-spin :show="loading">
      <!-- run 概要 -->
      <n-card v-if="run" size="small">
        <template #header>
          <div class="flex-between">
            <div>
              <h2>{{ run.id }}</h2>
              <n-space size="small" class="meta-line">
                <n-tag size="small" :type="statusType">{{ run.status }}</n-tag>
                <span class="dim">{{ run.model }} · {{ run.dataset }} · q{{ run.quality }} · bs{{ run.batch }} · λ{{ run.lamb }}</span>
                <span v-if="isLive" class="live">· 训练中，实时刷新…</span>
              </n-space>
            </div>
            <n-button size="small" secondary @click="router.push('/training/results')">返回列表</n-button>
          </div>
        </template>
      </n-card>

      <!-- 训练曲线（loss_series） -->
      <n-card v-if="run" size="small" class="block" title="训练曲线（训练集 loss/PSNR/bpp vs epoch）">
        <div v-if="trainCurve" class="curve-wrap"><v-chart class="curve" :option="trainCurve" autoresize /></div>
        <EmptyState v-else description="暂无 loss_series（epoch 1 还没落盘）" />
      </n-card>

      <!-- test 指标曲线（held-out val） -->
      <n-card v-if="run" size="small" class="block" title="测试指标（held-out val PSNR/bpp vs epoch）">
        <div v-if="testCurve" class="curve-wrap"><v-chart class="curve" :option="testCurve" autoresize /></div>
        <EmptyState v-else description="暂无 test_metrics（首个 epoch eval 还没跑）" />
      </n-card>

      <!-- 重建可视化：每 epoch 一个 carousel（6 样本，原图|输入|重建三图），grid 排列、懒加载 -->
      <n-card v-if="run" size="small" class="block" title="重建可视化（每 epoch 一个 carousel：左原图 | 中输入边缘 | 右重建，‹ › 切样本）">
        <div v-if="epochGroups.length" class="viz-grid">
          <VizEpochCard
            v-for="g in epochGroups" :key="g.epoch"
            :epoch="g.epoch" :samples="g.samples"
            @preview="(src, title) => { previewSrc = src; previewTitle = title; previewVisible = true }"
          />
        </div>
        <EmptyState v-else description="暂无可视化（首个 epoch viz 还没生成）" />
      </n-card>

      <EmptyState v-else-if="!loading" description="未找到该训练 run" />
    </n-spin>

    <n-modal v-model:show="previewVisible" preset="card" style="width: min(90vw, 1200px)" :title="previewTitle" :bordered="false">
      <img :src="previewSrc" class="preview-img" />
    </n-modal>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NCard, NSpin, NSpace, NTag, NButton, NModal, useMessage } from 'naive-ui'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import EmptyState from '../../components/common/EmptyState.vue'
import VizEpochCard from './VizEpochCard.vue'
import { getTrainRunDetail, getTrainOutputUrl } from '../../api/training'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent])

const route = useRoute()
const router = useRouter()
const message = useMessage()
const runId = route.params.id

const loading = ref(false)
const run = ref(null)
const previewVisible = ref(false)
const previewSrc = ref('')
const previewTitle = ref('')

const RUNNING = new Set(['running', 'started'])
const isLive = computed(() => run.value && RUNNING.has(run.value.status))
// viz 按 epoch 分组：[{epoch, samples:[{sample,path},...]}, ...]（兼容旧格式单 path/epoch）
const epochGroups = computed(() => {
  const flat = run.value?.viz || []
  const map = new Map()
  for (const v of flat) {
    const e = v.epoch
    if (!map.has(e)) map.set(e, { epoch: e, samples: [] })
    map.get(e).samples.push(v.sample != null ? { sample: v.sample, path: v.path } : { path: v.path })
  }
  return [...map.values()].sort((a, b) => a.epoch - b.epoch)
})
const statusType = computed(() => {
  const s = run.value?.status
  return s === 'completed' ? 'success' : s === 'failed' ? 'error' : s === 'running' ? 'info' : 'default'
})

const trainCurve = computed(() => {
  const r = run.value
  if (!r || !r.loss_series?.length) return null
  const epochs = r.loss_series.map(p => p.epoch)
  const build = (k) => r.loss_series.map(p => p[k])
  const series = []
  if (r.loss_series.some(p => p.loss != null)) series.push({ name: 'loss', type: 'line', data: build('loss'), smooth: true })
  if (r.loss_series.some(p => p.psnr != null)) series.push({ name: 'PSNR(dB)', type: 'line', data: build('psnr'), smooth: true, yAxisIndex: 1 })
  if (r.loss_series.some(p => p.bpp != null)) series.push({ name: 'bpp', type: 'line', data: build('bpp'), smooth: true, yAxisIndex: 1 })
  if (!series.length) return null
  return { tooltip: { trigger: 'axis' }, legend: { data: series.map(s => s.name) },
    xAxis: { type: 'category', data: epochs, name: 'epoch' },
    yAxis: [{ type: 'value', name: 'loss' }, { type: 'value', name: 'PSNR/bpp' }], series }
})

const testCurve = computed(() => {
  const r = run.value
  if (!r || !r.test_metrics?.length) return null
  const epochs = r.test_metrics.map(p => p.epoch)
  const build = (k) => r.test_metrics.map(p => p[k])
  const series = []
  if (r.test_metrics.some(p => p.psnr != null)) series.push({ name: 'test PSNR(dB)', type: 'line', data: build('psnr'), smooth: true, yAxisIndex: 0 })
  if (r.test_metrics.some(p => p.bpp != null)) series.push({ name: 'test bpp', type: 'line', data: build('bpp'), smooth: true, yAxisIndex: 1 })
  if (r.test_metrics.some(p => p.loss != null)) series.push({ name: 'test loss', type: 'line', data: build('loss'), smooth: true, yAxisIndex: 1 })
  if (!series.length) return null
  return { tooltip: { trigger: 'axis' }, legend: { data: series.map(s => s.name) },
    xAxis: { type: 'category', data: epochs, name: 'epoch' },
    yAxis: [{ type: 'value', name: 'PSNR' }, { type: 'value', name: 'bpp/loss' }], series }
})

let pollTimer = null
function startPolling() {
  if (pollTimer) return
  pollTimer = setInterval(async () => {
    const r = await getTrainRunDetail(runId).catch(() => null)
    if (r) run.value = r
    if (!RUNNING.has(r?.status)) stopPolling()
  }, 3000)
}
function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null } }

async function load() {
  loading.value = true
  try {
    const r = await getTrainRunDetail(runId)
    if (!r || !r.id) throw new Error('not found')
    run.value = r
    if (RUNNING.has(r.status)) startPolling()
  } catch {
    message.error('训练 run 不存在')
    router.push('/training/results')
  }
  loading.value = false
}

onMounted(load)
onUnmounted(stopPolling)
</script>

<style scoped lang="scss">
.train-run-detail { display: flex; flex-direction: column; }
.flex-between { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.meta-line { align-items: center; margin-top: 4px; }
.dim { color: var(--color-text-dim); font-size: 12px; }
.live { color: var(--color-primary); font-size: 12px; }
.block { margin-top: 12px; }
.curve-wrap { background: var(--color-elevated); border-radius: 8px; padding: 8px; }
.curve { height: 320px; width: 100%; }
.viz-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px;
}
.preview-img { width: 100%; display: block; border-radius: 8px; }
</style>
