<template>
  <div class="page-container">
    <n-card size="small">
      <template #header>
        <div class="flex-between">
          <h3>方法对比 — 不同轮廓提取方法下的 baseline</h3>
          <n-space align="center" size="small">
            <n-select v-model:value="filters.sequence" :options="sequenceOptions" placeholder="全部序列" clearable size="small" style="width: 140px" />
            <n-select v-model:value="filters.codec" :options="codecOptions" placeholder="全部 codec" clearable size="small" style="width: 120px" />
            <n-button size="small" @click="load">刷新</n-button>
          </n-space>
        </div>
      </template>
      <n-spin :show="loading">
        <template v-if="rows.length && methods.length">
          <p class="hint">每行一个操作点（序列 × codec × CRF），每列一种轮廓提取方法。点「播放」按需查看该方法的输出视频。</p>
          <table class="compare-matrix">
            <thead>
              <tr>
                <th>序列</th><th>codec</th><th>CRF</th>
                <th v-for="m in methods" :key="m">{{ m }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in rows" :key="r.key">
                <td>{{ r.sequence }}</td><td>{{ r.codec }}</td><td>{{ r.crf }}</td>
                <td v-for="m in methods" :key="m" class="cell">
                  <template v-if="r.byMethod[m]">
                    <div class="metrics">
                      <span class="m" title="PSNR (dB)">P {{ fmt(r.byMethod[m].psnr) }}</span>
                      <span class="m" title="SSIM">S {{ fmt(r.byMethod[m].ssim) }}</span>
                      <span class="m dim" title="码率 kbps">{{ fmt(r.byMethod[m].bitrate_kbps) }} kb/s</span>
                    </div>
                    <n-button size="tiny" type="primary" secondary @click="play(r.byMethod[m])">播放</n-button>
                  </template>
                  <span v-else class="dim">—</span>
                </td>
              </tr>
            </tbody>
          </table>
        </template>
        <EmptyState v-else description="暂无可对比数据。跑多个轮廓提取方法（canny/sobel）的 baseline 后，这里按方法对比 PSNR/码率 + 按需看输出视频。" />
      </n-spin>
    </n-card>

    <VideoModal v-model:show="videoShow" :src="videoSrc" :title="videoTitle" />
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { NCard, NSpin, NSpace, NSelect, NButton, useMessage } from 'naive-ui'
import EmptyState from '../../components/common/EmptyState.vue'
import VideoModal from '../../components/common/VideoModal.vue'
import { getEvalResults, getOutputUrl } from '../../api/evaluation'

const message = useMessage()
const loading = ref(false)
const results = ref([])
const filters = ref({ sequence: null, codec: null })
const videoShow = ref(false)
const videoSrc = ref('')
const videoTitle = ref('')

const sequenceOptions = computed(() => [...new Set(results.value.map(r => r.sequence_name))].map(s => ({ label: s, value: s })))
const codecOptions = computed(() => [...new Set(results.value.map(r => r.codec))].map(c => ({ label: c, value: c })))
const methods = computed(() => [...new Set(results.value.map(r => r.method))])

const rows = computed(() => {
  let list = results.value
  if (filters.value.sequence) list = list.filter(r => r.sequence_name === filters.value.sequence)
  if (filters.value.codec) list = list.filter(r => r.codec === filters.value.codec)
  // group by (sequence, codec, crf) -> {method -> run}
  const map = new Map()
  for (const r of list) {
    const key = `${r.sequence_name}|${r.codec}|${r.crf}`
    if (!map.has(key)) map.set(key, { key, sequence: r.sequence_name, codec: r.codec, crf: r.crf, byMethod: {} })
    map.get(key).byMethod[r.method] = r
  }
  return [...map.values()].sort((a, b) =>
    a.sequence.localeCompare(b.sequence) || a.codec.localeCompare(b.codec) || a.crf - b.crf)
})

function fmt(v) {
  if (v == null || isNaN(v)) return '-'
  return Number(v).toFixed(2)
}

function play(run) {
  // 按需：仅点击时赋值 src，VideoModal 内 <video v-if=show preload=none> 才请求字节
  if (!run.output_video) { message.warning('该方法/操作点暂无输出视频'); return }
  videoSrc.value = getOutputUrl(run.output_video)
  videoTitle.value = `${run.sequence_name} · ${run.codec} · crf${run.crf} · ${run.method}`
  videoShow.value = true
}

async function load() {
  loading.value = true
  try { results.value = (await getEvalResults()) || [] } catch (e) { message.error('加载结果失败'); results.value = [] }
  loading.value = false
}

onMounted(load)
</script>

<style scoped lang="scss">
.flex-between { display: flex; justify-content: space-between; align-items: center; }
.hint { font-size: 12px; color: var(--color-text-dim); margin-bottom: 12px; }
.compare-matrix { width: 100%; border-collapse: collapse; font-size: 13px; }
.compare-matrix th, .compare-matrix td { border: 1px solid var(--color-border-light); padding: 8px; text-align: center; }
.compare-matrix th { background: var(--color-elevated); }
.compare-matrix td.cell .metrics { display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; font-size: 12px; }
.compare-matrix .m { color: var(--color-text-secondary); }
.compare-matrix .dim { color: var(--color-text-dim); }
</style>
