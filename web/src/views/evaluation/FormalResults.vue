<template>
  <div class="page-container">
    <!-- 筛选:数据集/方法 -->
    <n-card size="small">
      <n-space align="center" size="small" wrap>
        <span class="lbl">数据集</span>
        <n-select v-model:value="filters.dataset" :options="datasetOptions" placeholder="全部" clearable size="small" style="width:150px" />
        <span class="lbl">方法</span>
        <n-select v-model:value="filters.method" :options="methodOptions" placeholder="全部" clearable size="small" style="width:130px" />
        <n-button size="small" @click="load">刷新</n-button>
      </n-space>
    </n-card>

    <!-- 演示视频(2-3 小窗口,不自动开播;preload=none + 用户点 play 才加载) -->
    <n-card size="small">
      <template #header><h3>演示视频</h3></template>
      <n-spin :show="loading">
        <div class="demo-grid">
          <div v-for="r in demoRuns" :key="r.id" class="demo-cell">
            <div class="demo-title">{{ r.codec }} · {{ r.sequence_name }} · crf{{ r.crf }}</div>
            <video v-if="r.output_video" :src="getOutputUrl(r.output_video)" preload="none" controls playsinline class="demo-video" />
            <div v-else class="no-video">无码流</div>
          </div>
          <div v-if="demoRuns.length === 0" class="no-demo">无演示视频(先跑 formal 全量 baseline)</div>
        </div>
      </n-spin>
    </n-card>

    <!-- 平均指标表(per-(codec,crf) 16 行,跨所有序列平均) -->
    <n-card size="small">
      <template #header><h3>平均指标(per codec × crf,跨所有序列)</h3></template>
      <n-spin :show="loading">
        <n-data-table :columns="aggColumns" :data="aggregated" size="small" :bordered="false" />
      </n-spin>
    </n-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { NCard, NSpace, NSelect, NButton, NSpin, NDataTable } from 'naive-ui'
import { getEvalResults, getMethods, getAggregatedResults, getOutputUrl } from '../../api/evaluation'

const loading = ref(false)
const results = ref([])
const methods = ref([])
const aggregated = ref([])
const filters = ref({ dataset: null, method: null })

const methodOptions = computed(() => methods.value.map(m => ({ label: m, value: m })))
const datasetOptions = computed(() => [...new Set(results.value.map(r => r.dataset_name).filter(Boolean))].map(d => ({ label: d, value: d })))

// 演示视频:每 codec 选 1 个代表 run(crf 23 + 第一个序列),2-3 个小窗口
const demoRuns = computed(() => {
  const byCodec = {}
  for (const r of results.value) {
    if (filters.value.dataset && r.dataset_name !== filters.value.dataset) continue
    if (filters.value.method && r.method !== filters.value.method) continue
    if (!byCodec[r.codec] && r.crf === 23 && r.output_video) byCodec[r.codec] = r
  }
  return Object.values(byCodec).slice(0, 3)
})

const aggColumns = computed(() => [
  { title: 'codec', key: 'codec', width: 80 },
  { title: 'CRF', key: 'crf', width: 60 },
  { title: '序列数', key: 'count', width: 70 },
  { title: 'PSNR', key: 'psnr', width: 80, render: (r) => fmt(r.psnr) },
  { title: 'SSIM', key: 'ssim', width: 70, render: (r) => fmt(r.ssim) },
  { title: '码率(kb/s)', key: 'bitrate_kbps', width: 110, render: (r) => fmt(r.bitrate_kbps) },
  { title: 'BPP', key: 'bpp', width: 70, render: (r) => fmt(r.bpp) },
  { title: '压缩比', key: 'ratio', width: 80, render: (r) => fmt(r.ratio) },
  { title: '编码fps', key: 'enc_fps', width: 90, render: (r) => fmt(r.enc_fps) },
  { title: '解码fps', key: 'dec_fps', width: 90, render: (r) => fmt(r.dec_fps) },
])

function fmt(v) {
  if (v == null) return '-'
  return typeof v === 'number' ? v.toFixed(2) : v
}

async function load() {
  loading.value = true
  try {
    const [res, meth] = await Promise.all([getEvalResults(), getMethods()])
    results.value = res
    methods.value = meth.methods || []
    const params = {}
    if (filters.value.dataset) params.dataset = filters.value.dataset
    if (filters.value.method) params.method = filters.value.method
    aggregated.value = await getAggregatedResults(params)
  } finally {
    loading.value = false
  }
}
onMounted(load)
</script>

<style scoped lang="scss">
.demo-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
.demo-cell { border: 1px solid var(--color-border); border-radius: 6px; padding: 8px; background: var(--color-card); }
.demo-title { font-size: 12px; color: var(--color-text-secondary); margin-bottom: 4px; }
.demo-video { width: 100%; max-height: 180px; background: #000; display: block; }
.no-video, .no-demo { color: var(--color-text-dim); padding: 20px; text-align: center; }
</style>
