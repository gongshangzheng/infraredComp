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
            <div v-for="r in runsByCodec[codec]" :key="r.id" class="cell"
                 :class="{ 'cell-expanded': isExpanded(r.id) }">
              <n-button class="expand-btn" size="tiny" quaternary
                        @click="toggleExpand(r.id)">
                <template #icon>
                  <n-icon :component="isExpanded(r.id) ? ContractOutline : ExpandOutline" />
                </template>
              </n-button>
              <div class="cell-title">{{ r.sequence_name }} · crf{{ r.crf }}</div>
              <div class="videos">
                <div class="v">
                  <span class="vlabel">原始</span>
                  <video v-if="r.original_video" :src="getOutputUrl(r.original_video)" preload="none" controls playsinline class="cell-video" />
                  <div v-else class="no-video">无</div>
                </div>
                <div class="v">
                  <span class="vlabel">边缘</span>
                  <video v-if="r.contour_video" :src="getOutputUrl(r.contour_video)" preload="none" controls playsinline class="cell-video" />
                  <div v-else class="no-video">无</div>
                </div>
                <div class="v">
                  <span class="vlabel">重建</span>
                  <video v-if="r.output_video" :src="getOutputUrl(r.output_video)" preload="none" controls playsinline class="cell-video" />
                  <div v-else class="no-video">无码流</div>
                </div>
              </div>
              <div class="cell-meta">
                PSNR {{ fmt(r.psnr) }} · SSIM {{ fmt(r.ssim) }} · bpp {{ fmt(r.bpp) }}
                · enc {{ fmt(r.enc_fps) }}fps · dec {{ fmt(r.dec_fps) }}fps
              </div>
            </div>
          </div>
        </div>
      </div>
    </n-spin>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { NCard, NSpace, NSelect, NButton, NSpin, NIcon } from 'naive-ui'
import { ExpandOutline, ContractOutline } from '@vicons/ionicons5'
import { useRoute, useRouter } from 'vue-router'
import { getEvalResults, getMethods, getOutputUrl } from '../../api/evaluation'

const route = useRoute()
const router = useRouter()

const loading = ref(false)
const results = ref([])
const methods = ref([])
const expanded = ref(new Set())
// 筛选器初始值从 URL query 恢复（F5 刷新不再清空）；crf 是数字需 Number()。
const numQ = (v) => (v == null || v === '' ? null : Number(v))
const filters = ref({
  dataset: route.query.dataset || null,
  method: route.query.method || null,
  sequence: route.query.sequence || null,
  codec: route.query.codec || null,
  crf: numQ(route.query.crf),
})

// 当前筛选写回 URL query（replace 不堆历史）；客户端筛选是响应式的，watch 即时同步。
function syncQuery() {
  const q = {}
  for (const k of ['dataset', 'method', 'sequence', 'codec']) {
    if (filters.value[k] != null && filters.value[k] !== '') q[k] = String(filters.value[k])
  }
  if (filters.value.crf != null) q.crf = String(filters.value.crf)
  router.replace({ query: q })
}
watch(filters, syncQuery, { deep: true })

function isExpanded(id) { return expanded.value.has(id) }
function toggleExpand(id) {
  const s = new Set(expanded.value)
  if (s.has(id)) s.delete(id); else s.add(id)
  expanded.value = s
}

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

let defaultsApplied = false   // 默认值只在首次挂载填一次，之后刷新/切换都保留用户选择
async function load() {
  loading.value = true
  try {
    const [res, meth] = await Promise.all([getEvalResults({ mode: 'speed' }), getMethods()])
    results.value = res
    methods.value = meth.methods || []
    // 仅首次进入填默认值（首个数据集 + canny）；用户手动改过或清空过都不覆盖。
    // sequence/codec/crf 仍默认“全部”（多值，全选更合理）。客户端筛选响应式，切换自动重过滤。
    if (!defaultsApplied) {
      if (!filters.value.dataset) {
        const datasets = [...new Set(results.value.map(r => r.dataset_name).filter(Boolean))]
        if (datasets.length) filters.value.dataset = datasets[0]
      }
      if (!filters.value.method) {
        filters.value.method = methods.value.includes('canny') ? 'canny' : (methods.value[0] || null)
      }
      defaultsApplied = true
    } else if (filters.value.method && !methods.value.includes(filters.value.method)) {
      filters.value.method = methods.value[0] || null
    }
  } finally {
    loading.value = false
  }
}
onMounted(load)
</script>

<style scoped lang="scss">
.codec-row { margin-bottom: 20px; }
.codec-label { font-weight: 700; margin-bottom: 8px; color: var(--color-text-primary); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }
.cell { position: relative; border: 1px solid var(--color-border); border-radius: 6px; padding: 8px; background: var(--color-card); }
.expand-btn { position: absolute; top: 4px; right: 4px; z-index: 2; opacity: 0.7; }
.expand-btn:hover { opacity: 1; }
.cell-expanded { grid-column: 1 / -1; }
.cell-expanded .cell-video { max-height: 320px; }
.cell-expanded .videos { gap: 10px; }
.cell-title { font-size: 12px; color: var(--color-text-secondary); margin-bottom: 6px; padding-right: 48px; }
.videos { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.v { display: flex; flex-direction: column; gap: 2px; }
.vlabel { font-size: 10px; color: var(--color-text-dim); }
.cell-video { width: 100%; max-height: 120px; background: #000; display: block; object-fit: contain; }
.no-video { height: 80px; display: flex; align-items: center; justify-content: center; color: var(--color-text-dim); background: var(--color-bg); font-size: 11px; }
.cell-meta { font-size: 11px; color: var(--color-text-dim); margin-top: 6px; }
.empty { padding: 40px; text-align: center; color: var(--color-text-dim); }
</style>
