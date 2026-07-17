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
        <n-button size="small" @click="reload">刷新</n-button>
      </n-space>
    </n-card>

    <n-spin :show="loading">
      <div v-if="allFiltered.length === 0 && !loading" class="empty">
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
              <div v-if="isExpanded(r.id)" class="videos">
                <div class="v">
                  <span class="vlabel">原始</span>
                  <video v-if="r.original_video && !isImage(r.original_video)" :src="getOutputUrl(r.original_video)" preload="none" controls playsinline class="cell-video" />
                  <img v-else-if="r.original_video" :src="getUrl(r.original_video, r.dataset_name)" class="cell-img" />
                  <div v-else class="no-video">无</div>
                </div>
                <div class="v">
                  <span class="vlabel">边缘</span>
                  <video v-if="r.contour_video && !isImage(r.contour_video)" :src="getOutputUrl(r.contour_video)" preload="none" controls playsinline class="cell-video" />
                  <img v-else-if="r.contour_video" :src="getUrl(r.contour_video, r.dataset_name)" class="cell-img" />
                  <div v-else class="no-video">无</div>
                </div>
                <div class="v">
                  <span class="vlabel">重建</span>
                  <video v-if="r.output_video && !isImage(r.output_video)" :src="getOutputUrl(r.output_video)" preload="none" controls playsinline class="cell-video" />
                  <img v-else-if="r.output_video" :src="getUrl(r.output_video, r.dataset_name)" class="cell-img" />
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

    <!-- 加载更多 -->
    <div v-if="hasMore" class="load-more">
      <n-button size="small" :loading="loadingMore" @click="loadMore">加载更多（{{ allFiltered.length }} / {{ total }}）</n-button>
    </div>
    <div v-else-if="total > 0" class="load-more-hint">已加载全部 {{ total }} 条结果</div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { NCard, NSpace, NSelect, NButton, NSpin, NIcon } from 'naive-ui'
import { ExpandOutline, ContractOutline } from '@vicons/ionicons5'
import { useRoute, useRouter } from 'vue-router'
import { getEvalResults, getMethods, getOutputUrl, getDatasetMediaUrl } from '../../api/evaluation'

const route = useRoute()
const router = useRouter()

const loading = ref(false)
const loadingMore = ref(false)
const results = ref([])
const methods = ref([])
const total = ref(0)
const expanded = ref(new Set())
const PAGE_SIZE = 24

const numQ = (v) => (v == null || v === '' ? null : Number(v))
const filters = ref({
  dataset: route.query.dataset || null,
  method: route.query.method || null,
  sequence: route.query.sequence || null,
  codec: route.query.codec || null,
  crf: numQ(route.query.crf),
})

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

const isImage = (path) => /\.(png|jpg|jpeg|bmp|tif|tiff)$/i.test(path || '')
const getUrl = (path, dataset) => {
  if (!path) return ''
  if (path.startsWith('contour/bsds_') || path.startsWith('contour/imagenet_')) return getDatasetMediaUrl(dataset, path)
  return getOutputUrl(path)
}
const methodOptions = computed(() => {
  const ds = filters.value.dataset
  const rs = ds ? results.value.filter(r => r.dataset_name === ds) : results.value
  return [...new Set(rs.map(r => r.method).filter(Boolean))].map(m => ({ label: m, value: m }))
})
const sequenceOptions = computed(() => [...new Set(results.value.map(r => r.sequence_name))].map(s => ({ label: s, value: s })))
const codecOptions = computed(() => [...new Set(results.value.map(r => r.codec))].map(c => ({ label: c, value: c })))
const crfOptions = computed(() => [...new Set(results.value.map(r => r.crf))].sort((a, b) => a - b).map(c => ({ label: 'crf' + c, value: c })))
const datasetOptions = computed(() => [...new Set(results.value.map(r => r.dataset_name).filter(Boolean))].map(d => ({ label: d, value: d })))

const allFiltered = computed(() => {
  let list = results.value
  if (filters.value.dataset) list = list.filter(r => r.dataset_name === filters.value.dataset)
  if (filters.value.method) list = list.filter(r => r.method === filters.value.method)
  if (filters.value.sequence) list = list.filter(r => r.sequence_name === filters.value.sequence)
  if (filters.value.codec) list = list.filter(r => r.codec === filters.value.codec)
  if (filters.value.crf !== null && filters.value.crf !== undefined) list = list.filter(r => r.crf === filters.value.crf)
  return list
})

const hasMore = computed(() => allFiltered.value.length < total.value)

const codecGroups = computed(() => [...new Set(allFiltered.value.map(r => r.codec))].sort())
const runsByCodec = computed(() => {
  const m = {}
  for (const r of allFiltered.value) {
    (m[r.codec] = m[r.codec] || []).push(r)
  }
  return m
})

function fmt(v) {
  if (v == null) return '-'
  return typeof v === 'number' ? v.toFixed(2) : v
}

let defaultsApplied = false

async function fetchPage(offset, limit) {
  return getEvalResults({ mode: 'speed', offset, limit })
}

async function reload() {
  loading.value = true
  try {
    const [res, meth] = await Promise.all([
      fetchPage(0, PAGE_SIZE),
      getMethods()
    ])
    const data = res.runs !== undefined ? res : { runs: res, total: Array.isArray(res) ? res.length : 0 }
    results.value = data.runs || []
    total.value = data.total ?? results.value.length
    methods.value = meth.methods || []
    if (!defaultsApplied) {
      if (!filters.value.dataset) {
        const datasets = [...new Set(results.value.map(r => r.dataset_name).filter(Boolean))]
        if (datasets.length) filters.value.dataset = datasets[0]
      }
      if (!filters.value.method) {
        const dsMethods = methodOptions.value
        filters.value.method = dsMethods.length ? dsMethods[0].value : (methods.value[0] || null)
      }
      defaultsApplied = true
    } else if (filters.value.method && !methods.value.includes(filters.value.method)) {
      filters.value.method = methods.value[0] || null
    }
  } finally {
    loading.value = false
  }
}

async function loadMore() {
  loadingMore.value = true
  try {
    const res = await fetchPage(results.value.length, PAGE_SIZE)
    if (res.runs !== undefined) {
      results.value.push(...res.runs)
      total.value = res.total ?? total.value
    } else if (Array.isArray(res)) {
      results.value.push(...res)
    }
  } finally {
    loadingMore.value = false
  }
}

onMounted(reload)
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
.load-more { text-align: center; padding: 16px; }
.load-more-hint { text-align: center; padding: 12px; font-size: 12px; color: var(--color-text-dim); }
</style>
