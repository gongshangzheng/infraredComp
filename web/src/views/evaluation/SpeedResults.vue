<template>
  <div class="page-container">
    <!-- 筛选:数据集/方法/序列/codec/crf -->
    <n-card size="small">
      <n-space align="center" size="small" wrap>
        <span class="lbl">数据集</span>
        <n-select v-model:value="filters.dataset" :options="datasetOptions" placeholder="全部" clearable size="small" style="width:150px" />
        <span class="lbl">方法</span>
        <n-select v-model:value="filters.method" :options="methodOptions" placeholder="全部" clearable size="small" style="width:130px" />
        <span class="lbl">序列</span>
        <n-select v-model:value="filters.sequence" :options="sequenceOptions" placeholder="全部" clearable size="small" style="width:140px" />
        <span class="lbl">codec</span>
        <n-select v-model:value="filters.codec" :options="codecOptions" placeholder="全部" clearable size="small" style="width:120px" />
        <span class="lbl">CRF</span>
        <n-select v-model:value="filters.crf" :options="crfOptions" placeholder="全部" clearable size="small" style="width:100px" />
        <n-button size="small" @click="load">刷新</n-button>
      </n-space>
    </n-card>

    <n-spin :show="loading">
      <div v-if="filteredResults.length === 0" class="empty">
        无结果。先在"评测运行"跑一次 speed run,或调整筛选。
      </div>
      <div v-else>
        <div v-for="codec in codecGroups" :key="codec" class="codec-row">
          <div class="codec-label">{{ codec }}</div>
          <div class="grid">
            <div v-for="r in runsByCodec[codec]" :key="r.id" class="cell">
              <div class="cell-title">{{ r.sequence_name }} · crf{{ r.crf }}</div>
              <video v-if="r.output_video" :src="getOutputUrl(r.output_video)" preload="none" controls playsinline class="cell-video" />
              <div v-else class="no-video">无码流</div>
              <div class="cell-meta">PSNR {{ fmt(r.psnr) }} · {{ fmt(r.bitrate_kbps) }} kb/s</div>
            </div>
          </div>
        </div>
      </div>
    </n-spin>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { NCard, NSpace, NSelect, NButton, NSpin } from 'naive-ui'
import { getEvalResults, getMethods, getOutputUrl } from '../../api/evaluation'

const loading = ref(false)
const results = ref([])
const methods = ref([])
const filters = ref({ dataset: null, method: null, sequence: null, codec: null, crf: null })

const methodOptions = computed(() => methods.value.map(m => ({ label: m, value: m })))
const sequenceOptions = computed(() => [...new Set(results.value.map(r => r.sequence_name))].map(s => ({ label: s, value: s })))
const codecOptions = computed(() => [...new Set(results.value.map(r => r.codec))].map(c => ({ label: c, value: c })))
const crfOptions = computed(() => [...new Set(results.value.map(r => r.crf))].sort((a, b) => a - b).map(c => ({ label: 'crf' + c, value: c })))
const datasetOptions = computed(() => [...new Set(results.value.map(r => r.dataset_name).filter(Boolean))].map(d => ({ label: d, value: d })))

const filteredResults = computed(() => {
  let list = results.value
  if (filters.value.dataset) list = list.filter(r => r.dataset_name === filters.value.dataset)
  if (filters.value.method) list = list.filter(r => r.method === filters.value.method)
  if (filters.value.sequence) list = list.filter(r => r.sequence_name === filters.value.sequence)
  if (filters.value.codec) list = list.filter(r => r.codec === filters.value.codec)
  if (filters.value.crf !== null && filters.value.crf !== undefined) list = list.filter(r => r.crf === filters.value.crf)
  return list
})

const codecGroups = computed(() => [...new Set(filteredResults.value.map(r => r.codec))].sort())
const runsByCodec = computed(() => {
  const m = {}
  for (const r of filteredResults.value) {
    (m[r.codec] = m[r.codec] || []).push(r)
  }
  return m
})

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
  } finally {
    loading.value = false
  }
}
onMounted(load)
</script>

<style scoped lang="scss">
.codec-row { margin-bottom: 20px; }
.codec-label { font-weight: 700; margin-bottom: 8px; color: var(--color-text-primary); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
.cell { border: 1px solid var(--color-border); border-radius: 6px; padding: 8px; background: var(--color-card); }
.cell-title { font-size: 12px; color: var(--color-text-secondary); margin-bottom: 4px; }
.cell-video { width: 100%; max-height: 150px; background: #000; display: block; }
.no-video { height: 100px; display: flex; align-items: center; justify-content: center; color: var(--color-text-dim); background: var(--color-bg); }
.cell-meta { font-size: 11px; color: var(--color-text-dim); margin-top: 4px; }
.empty { padding: 40px; text-align: center; color: var(--color-text-dim); }
</style>
