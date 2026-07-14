<template>
  <div class="page-container">
    <!-- 筛选:数据集/方法 -->
    <n-card size="small">
      <n-space align="center" size="small" wrap>
        <span class="lbl">数据集</span>
        <n-select v-model:value="filters.dataset" :options="datasetOptions" placeholder="全部" clearable size="small" style="width:150px" @update:value="load" />
        <span class="lbl">方法</span>
        <n-select v-model:value="filters.method" :options="methodOptions" placeholder="全部" clearable size="small" style="width:130px" @update:value="load" />
        <n-button size="small" @click="load">刷新</n-button>
      </n-space>
    </n-card>

    <!-- RD 曲线:每 codec 一条线,直观展示压缩效果 -->
    <n-card size="small">
      <template #header><h3>R-D 曲线(每 codec 一条线)</h3></template>
      <n-spin :show="loading">
        <div v-if="aggregated.length" class="charts">
          <v-chart class="chart" :option="rdOption('psnr')" autoresize />
          <v-chart class="chart" :option="rdOption('ssim')" autoresize />
        </div>
        <EmptyState v-else description="无 formal 评测数据" />
      </n-spin>
    </n-card>

    <!-- 平均指标表(原生 table:第一列 codec 真 rowspan 合并居中;每行右侧直接三拼 video) -->
    <n-card size="small">
      <template #header><h3>平均指标(per codec × crf,跨所有序列)</h3></template>
      <n-spin :show="loading">
        <div class="table-wrap">
          <table class="agg-table">
            <thead>
              <tr>
                <th class="c-codec">codec</th>
                <th class="c-c">CRF</th>
                <th class="c-c">序列数</th>
                <th class="c-r">PSNR</th>
                <th class="c-r">SSIM</th>
                <th class="c-r">码率(kb/s)</th>
                <th class="c-r">BPP</th>
                <th class="c-r">压缩比</th>
                <th class="c-r">编码fps</th>
                <th class="c-r">解码fps</th>
                <th class="c-demo">演示(原始|轮廓|重建)</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="grp in groupedAgg" :key="grp.codec">
                <tr v-for="(row, idx) in grp.rows" :key="row.codec + '|' + row.crf"
                    :class="{ 'grp-start': idx === 0 }">
                  <td v-if="idx === 0" :rowspan="grp.rows.length" class="c-codec codec-cell">
                    {{ grp.codec }}
                  </td>
                  <td class="c-c">{{ row.crf }}</td>
                  <td class="c-c">{{ row.count }}</td>
                  <td class="c-r">{{ fmt(row.psnr) }}</td>
                  <td class="c-r">{{ fmt(row.ssim) }}</td>
                  <td class="c-r">{{ fmt(row.bitrate_kbps) }}</td>
                  <td class="c-r">{{ fmt(row.bpp) }}</td>
                  <td class="c-r">{{ fmt(row.ratio) }}</td>
                  <td class="c-r">{{ fmt(row.enc_fps) }}</td>
                  <td class="c-r">{{ fmt(row.dec_fps) }}</td>
                  <td class="c-demo">
                    <video
                      :src="rowDemoUrl(row)"
                      preload="none" controls playsinline
                      class="demo-video" />
                  </td>
                </tr>
              </template>
              <tr v-if="!aggregated.length">
                <td colspan="11" class="empty">无 formal 评测结果(先跑一次 formal 全量 baseline)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </n-spin>
    </n-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { NCard, NSpace, NSelect, NButton, NSpin } from 'naive-ui'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { useRoute, useRouter } from 'vue-router'
import EmptyState from '../../components/common/EmptyState.vue'
import { getEvalResults, getMethods, getAggregatedResults } from '../../api/evaluation'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent])

const route = useRoute()
const router = useRouter()

const loading = ref(false)
const results = ref([])
const methods = ref([])
const aggregated = ref([])
// 筛选器初始值从 URL query 恢复（F5 刷新不再清空）；首次无 query 时由 load() 填默认。
const filters = ref({
  dataset: route.query.dataset || null,
  method: route.query.method || null,
})

const methodOptions = computed(() => methods.value.map(m => ({ label: m, value: m })))
const datasetOptions = computed(() => [...new Set(results.value.map(r => r.dataset_name).filter(Boolean))].map(d => ({ label: d, value: d })))

// 按连续同 codec 分组(供第一列 rowspan)
const groupedAgg = computed(() => {
  const out = []
  for (const r of aggregated.value) {
    const last = out[out.length - 1]
    if (last && last.codec === r.codec) last.rows.push(r)
    else out.push({ codec: r.codec, rows: [r] })
  }
  return out
})

// RD 曲线 option:x=bpp, y=metric(psnr/ssim),每 codec 一条线
function rdOption(metric) {
  const byCodec = {}
  for (const r of aggregated.value) {
    (byCodec[r.codec] ||= []).push(r)
  }
  const yName = metric === 'ssim' ? 'SSIM' : 'PSNR (dB)'
  const series = Object.entries(byCodec).map(([codec, rows]) => {
    const pts = rows
      .filter(r => r.bpp != null && r[metric] != null)
      .sort((a, b) => a.bpp - b.bpp)
      .map(r => [Number(r.bpp), Number(r[metric])])
    return { name: codec, type: 'line', smooth: true, showSymbol: true, symbolSize: 7, data: pts }
  })
  return {
    legend: { type: 'scroll', bottom: 0, textStyle: { fontSize: 11 } },
    grid: { left: 58, right: 24, top: 16, bottom: 48 },
    tooltip: { trigger: 'axis', confine: true, valueFormatter: (v) => (Array.isArray(v) ? `bpp=${v[0]} ${yName.split(' ')[0]}=${v[1]}` : v) },
    xAxis: { type: 'value', name: 'bpp', nameLocation: 'middle', nameGap: 28, scale: true },
    yAxis: { type: 'value', name: yName, scale: true },
    series,
  }
}

// 三拼演示 mp4 的流式 URL(指向 /results/row_demo,点 play 才后端生成+流)
function rowDemoUrl(row) {
  const p = new URLSearchParams({ codec: row.codec, crf: String(row.crf), mode: 'formal' })
  if (filters.value.method) p.set('method', filters.value.method)
  if (filters.value.dataset) p.set('dataset', filters.value.dataset)
  return `/api/evaluation/results/row_demo?${p.toString()}`
}

function fmt(v) {
  if (v == null) return '-'
  return typeof v === 'number' ? v.toFixed(2) : v
}

let defaultsApplied = false   // 默认值只在首次挂载填一次，之后刷新/切换都保留用户选择
// 把当前筛选写回 URL query（replace 不堆历史），F5 可从 URL 恢复选择。
function syncQuery() {
  const q = {}
  for (const k of ['dataset', 'method']) {
    if (filters.value[k] != null && filters.value[k] !== '') q[k] = String(filters.value[k])
  }
  router.replace({ query: q })
}
async function load() {
  loading.value = true
  try {
    const [res, meth] = await Promise.all([getEvalResults({ mode: 'formal' }), getMethods()])
    results.value = res
    methods.value = meth.methods || []
    // 仅首次进入给筛选器填默认值（首个数据集 + canny）；用户手动改过或清空过都不覆盖。
    if (!defaultsApplied) {
      if (!filters.value.dataset) {
        const datasets = [...new Set(results.value.map(r => r.dataset_name).filter(Boolean))]
        if (datasets.length) filters.value.dataset = datasets[0]
      }
      if (!filters.value.method) {
        const m = methods.value
        filters.value.method = m.includes('canny') ? 'canny' : (m[0] || null)
      }
      defaultsApplied = true
    } else if (filters.value.method && !methods.value.includes(filters.value.method)) {
      // 当前方法已不在可选列表（如权重缺失被下线）→ 回退到首个可用
      filters.value.method = methods.value[0] || null
    }
    const params = { mode: 'formal' }
    if (filters.value.dataset) params.dataset = filters.value.dataset
    if (filters.value.method) params.method = filters.value.method
    aggregated.value = await getAggregatedResults(params)
    syncQuery()
  } finally {
    loading.value = false
  }
}
onMounted(load)
</script>

<style scoped lang="scss">
.charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.chart { height: 360px; }
.table-wrap { overflow-x: auto; }
.agg-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  th, td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--color-border-light);
    white-space: nowrap;
  }
  thead th {
    position: sticky; top: 0;
    background: var(--color-card);
    font-weight: 600;
    text-align: right;
    border-bottom: 2px solid var(--color-border);
  }
  .c-codec { text-align: left; }
  .c-c { text-align: center; }
  .c-r { text-align: right; font-variant-numeric: tabular-nums; }
  .c-demo { text-align: center; }
  /* codec 组之间加粗分隔线 */
  .grp-start > td { border-top: 2px solid var(--color-border); }
  .codec-cell {
    font-weight: 700;
    text-align: center;
    vertical-align: middle;   /* 第一列 codec 跨行垂直居中 */
    background: var(--color-elevated);
  }
  .demo-video {
    width: 220px;
    max-height: 80px;
    background: #000;
    display: block;
    margin: 0 auto;
    border-radius: 4px;
  }
  .empty { color: var(--color-text-dim); padding: 30px; text-align: center; }
}
</style>
