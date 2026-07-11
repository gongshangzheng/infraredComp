<template>
  <div class="page-container">
    <n-card title="轮廓视频压缩基准测试" :bordered="false">
      <template #header-extra>
        <n-space>
          <n-tag v-if="generatedAt" size="small" type="info">
            生成于 {{ generatedAt.slice(0, 19).replace('T', ' ') }}
          </n-tag>
          <n-button size="small" @click="load" :loading="loading">刷新</n-button>
        </n-space>
      </template>

      <!-- 统计卡 -->
      <n-grid v-if="stats" cols="1 s:2 m:4" responsive="screen" :x-gap="12" :y-gap="12">
        <n-gi>
          <n-card size="small"><div class="stat-label">最佳 PSNR</div>
            <div class="stat-value">{{ stats.bestPsnr.toFixed(2) }} dB</div>
            <div class="stat-sub">{{ stats.bestPsnrCodec }}</div></n-card>
        </n-gi>
        <n-gi>
          <n-card size="small"><div class="stat-label">最小 BPP</div>
            <div class="stat-value">{{ stats.minBpp.toFixed(3) }}</div>
            <div class="stat-sub">{{ stats.minBppCodec }}</div></n-card>
        </n-gi>
        <n-gi>
          <n-card size="small"><div class="stat-label">最快编码</div>
            <div class="stat-value">{{ stats.fastestEnc.toFixed(0) }} fps</div>
            <div class="stat-sub">{{ stats.fastestEncCodec }}</div></n-card>
        </n-gi>
        <n-gi>
          <n-card size="small"><div class="stat-label">最高压缩比</div>
            <div class="stat-value">{{ stats.bestRatio.toFixed(1) }}x</div>
            <div class="stat-sub">{{ stats.bestRatioCodec }}</div></n-card>
        </n-gi>
      </n-grid>
    </n-card>

    <!-- RD 曲线 -->
    <n-card v-if="runs.length" title="RD 曲线：PSNR vs Bitrate" :bordered="false" class="block">
      <canvas ref="chartRef" height="120"></canvas>
    </n-card>

    <!-- 结果表 -->
    <n-card v-if="runs.length" title="逐次运行结果" :bordered="false" class="block">
      <n-data-table :columns="columns" :data="runs" :bordered="false" size="small"
                    :pagination="{ pageSize: 15 }" />
    </n-card>

    <EmptyState v-else description="尚无评测结果"
      hint="运行：uv run python -m benchmark.video --input datasets/raw/xxx.mp4 --method canny --crfs 18,23,28,33" />
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed, h } from 'vue'
import { NCard, NGrid, NGi, NDataTable, NTag, NButton, NSpace } from 'naive-ui'
import Chart from 'chart.js/auto'
import EmptyState from '../../components/common/EmptyState.vue'
import { getResults } from '../../api/benchmark'

const CODEC_COLORS = {
  x264: '#ef4444', x265: '#3b82f6', svtav1: '#10b981', vp9: '#f59e0b',
}
const colorFor = (c) => CODEC_COLORS[c] || '#6b7280'

const runs = ref([])
const generatedAt = ref(null)
const loading = ref(false)
const chartRef = ref(null)
let chart = null

const stats = computed(() => {
  if (!runs.value.length) return null
  const r = runs.value
  const best = (fn) => r.reduce((a, b) => (fn(b) > fn(a) ? b : a))
  const bestPsnr = best(x => x.psnr)
  const minBpp = r.filter(x => x.bpp > 0).reduce((a, b) => (b.bpp < a.bpp ? b : a))
  const fastestEnc = best(x => x.enc_fps)
  const bestRatio = best(x => x.compression_ratio)
  return {
    bestPsnr: bestPsnr.psnr, bestPsnrCodec: bestPsnr.codec,
    minBpp: minBpp.bpp, minBppCodec: minBpp.codec,
    fastestEnc: fastestEnc.enc_fps, fastestEncEncCodec: fastestEnc.codec, fastestEncCodec: fastestEnc.codec,
    bestRatio: bestRatio.compression_ratio, bestRatioCodec: bestRatio.codec,
  }
})

const columns = [
  { title: 'Codec', key: 'codec', render: (r) => r.codec },
  { title: 'CRF', key: 'crf' },
  { title: 'Sequence', key: 'sequence_name' },
  { title: 'PSNR(dB)', key: 'psnr', render: (r) => r.psnr.toFixed(2) },
  { title: 'SSIM', key: 'ssim', render: (r) => r.ssim.toFixed(4) },
  { title: 'BPP', key: 'bpp', render: (r) => r.bpp.toFixed(3) },
  { title: 'Bitrate(kbps)', key: 'bitrate_kbps', render: (r) => r.bitrate_kbps.toFixed(1) },
  { title: 'Ratio', key: 'compression_ratio', render: (r) => r.compression_ratio.toFixed(1) + 'x' },
  { title: 'Enc fps', key: 'enc_fps', render: (r) => r.enc_fps.toFixed(1) },
  { title: 'Dec fps', key: 'dec_fps', render: (r) => r.dec_fps.toFixed(1) },
  { title: 'Temporal', key: 'temporal_metric', render: (r) => r.temporal_metric.toFixed(2) },
]

function drawChart() {
  if (!chartRef.value || !runs.value.length) return
  // group by codec, sort by bitrate
  const byCodec = {}
  for (const r of runs.value) {
    (byCodec[r.codec] ||= []).push({ x: r.bitrate_kbps, y: r.psnr })
  }
  const datasets = Object.entries(byCodec).map(([codec, pts]) => ({
    label: codec,
    data: pts.sort((a, b) => a.x - b.x),
    borderColor: colorFor(codec),
    backgroundColor: colorFor(codec),
    pointRadius: 5, tension: 0.2,
  }))
  if (chart) chart.destroy()
  chart = new Chart(chartRef.value, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      scales: {
        x: { type: 'linear', title: { display: true, text: 'Bitrate (kbps)' } },
        y: { title: { display: true, text: 'PSNR (dB)' } },
      },
      plugins: { legend: { position: 'top' } },
    },
  })
}

async function load() {
  loading.value = true
  try {
    const data = await getResults()
    runs.value = data.runs || []
    generatedAt.value = data.generated_at
    setTimeout(drawChart, 0)
  } finally {
    loading.value = false
  }
}

onMounted(load)
onUnmounted(() => chart?.destroy())
</script>

<style scoped lang="scss">
.block { margin-top: 16px; }
.stat-label { font-size: 12px; color: #6b7280; }
.stat-value { font-size: 22px; font-weight: 700; color: #4f46e5; }
.stat-sub { font-size: 12px; color: #9ca3af; }
</style>
