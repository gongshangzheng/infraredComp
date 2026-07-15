<template>
  <div class="page-container">
    <n-spin :show="loading">
      <n-card v-if="dataset" size="small" class="dataset-header">
        <template #header>
          <div class="flex-between">
            <div>
              <h2>{{ dataset.name }}</h2>
              <n-space v-if="dataset.source" size="small" class="meta-line">
                <n-tag size="small" type="info">{{ dataset.kind === 'contour' ? '轮廓产物' : '原始数据集' }}</n-tag>
                <a v-if="dataset.source" :href="dataset.source" target="_blank" class="source-link">来源 ↗</a>
              </n-space>
            </div>
            <n-button v-if="showDownload" type="primary" :loading="downloading" @click="handleDownload">
              下载 OSU Thermal
            </n-button>
          </div>
        </template>

        <n-descriptions :column="3" size="small" label-placement="left" bordered
          v-if="dataset.citation || dataset.license || dataset.format"
        >
          <n-descriptions-item v-if="dataset.format" label="格式">{{ dataset.format }}</n-descriptions-item>
          <n-descriptions-item v-if="dataset.license" label="许可证">{{ dataset.license }}</n-descriptions-item>
          <n-descriptions-item v-if="dataset.citation" label="引用">{{ dataset.citation }}</n-descriptions-item>
        </n-descriptions>

        <p v-if="dataset.description" class="desc">{{ dataset.description }}</p>
      </n-card>

      <!-- 序列列表 -->
      <n-card v-if="dataset?.sequences?.length" title="序列" size="small" style="margin-top: 16px">
        <n-collapse accordion>
          <n-collapse-item v-for="seq in dataset.sequences" :key="seq.id" :name="seq.id">
            <template #header>
              <div class="seq-header">
                <span class="seq-name">{{ seq.name || seq.id }}</span>
                <n-space size="small">
                  <n-tag v-if="seq.missing" size="small" type="error">文件缺失</n-tag>
                  <n-tag v-else size="small" type="success">可用</n-tag>
                  <span class="dim">{{ seq.width }}×{{ seq.height }} · {{ seq.frame_count }} 帧 · {{ fmtFps(seq.fps) }} fps</span>
                </n-space>
              </div>
            </template>

            <!-- 序列左右对比：左原始 / 右轮廓（都是可播放视频，不要帧预览） -->
            <div class="seq-compare">
              <div class="compare-cell">
                <div class="cell-head">
                  <span class="vlabel">原始视频</span>
                  <span class="dim small">{{ seq.frame_count }} 帧 · {{ seq.width }}×{{ seq.height }}</span>
                </div>
                <video v-if="seq.view_source" :src="outputUrl(seq.view_source)"
                  controls preload="none" playsinline class="cmp-video" />
                <div v-else class="cmp-empty">
                  原始视频不可播放{{ seq.missing ? '（文件缺失）' : '' }}
                </div>
              </div>

              <div class="compare-cell">
                <div class="cell-head"><span class="vlabel">轮廓视频</span></div>
                <n-tabs v-if="contourCount(seq) >= 1" type="line" size="small" animated>
                  <n-tab-pane v-for="[method, info] in contourEntries(seq)" :key="method"
                    :name="method" :tab="`${method} · ${info.frame_count}帧`">
                    <video v-if="info.view_video" :src="outputUrl(info.view_video)"
                      controls preload="none" playsinline class="cmp-video" />
                    <div v-else class="cmp-empty">轮廓视频不可用</div>
                  </n-tab-pane>
                </n-tabs>
                <div v-else class="cmp-empty">无轮廓产物</div>
              </div>
            </div>
          </n-collapse-item>
        </n-collapse>
      </n-card>

      <!-- imagenet 在线提边缘预览（parquet 实时采样，不落地） -->
      <n-card v-if="dataset?.kind === 'image'" title="在线提边缘预览（从 parquet 实时采样）" size="small" style="margin-top: 16px">
        <template #header-extra>
          <n-space size="small" align="center">
            <n-select v-model:value="previewMethod" :options="methodOptions" size="small" style="width: 110px" />
            <n-button size="small" :loading="previewLoading" @click="loadPreview">生成预览</n-button>
          </n-space>
        </template>
        <n-spin :show="previewLoading">
          <div v-if="imagePreviews.length" class="preview-grid">
            <div v-for="(p, i) in imagePreviews" :key="i" class="preview-pair">
              <div class="preview-cell">
                <span class="vlabel">原始</span>
                <img :src="p.original" class="gallery-img" @click="previewSrc = p.original; previewTitle = `原始 #${i+1}`; previewVisible = true" />
              </div>
              <div class="preview-cell">
                <span class="vlabel">边缘 · {{ previewMethod }}</span>
                <img :src="p.edge" class="gallery-img" @click="previewSrc = p.edge; previewTitle = `${previewMethod} #${i+1}`; previewVisible = true" />
              </div>
            </div>
          </div>
          <EmptyState v-else description="点「生成预览」从 parquet 采样并实时提边缘" />
        </n-spin>
        <p class="hint">评测时只采样 {{ dataset.sample_images }} 张图（{{ dataset.usage === 'speed' ? 'speed run 检验' : 'formal 测试' }}），在线提 {{ previewMethod }} 边缘，不落地。</p>
      </n-card>

      <EmptyState v-else-if="!loading && dataset?.kind !== 'image'" description="该数据集暂无序列" />
    </n-spin>

    <!-- 图片预览 -->
    <n-modal v-model:show="previewVisible" preset="card" style="width: min(90vw, 1200px)" :title="previewTitle"
      :bordered="false" :segmented="{ content: true }"
    >
      <img :src="previewSrc" class="preview-img" />
    </n-modal>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  NCard, NSpin, NTag, NSpace, NButton, NSelect, NDescriptions, NDescriptionsItem,
  NCollapse, NCollapseItem, NTabs, NTabPane, NModal, useMessage,
} from 'naive-ui'
import EmptyState from '../../components/common/EmptyState.vue'
import {
  getDatasetDetail, getDatasetPreview, getOutputUrl, downloadDataset,
} from '../../api/evaluation'

const route = useRoute()
const router = useRouter()
const message = useMessage()

const loading = ref(false)
const dataset = ref(null)
const downloading = ref(false)
const previewVisible = ref(false)
const previewSrc = ref('')
const previewTitle = ref('')
const previewMethod = ref('canny')
const previewLoading = ref(false)
const imagePreviews = ref([])
const methodOptions = [{ label: 'canny', value: 'canny' }, { label: 'sobel', value: 'sobel' }]

const datasetId = route.params.id

const showDownload = computed(() => dataset.value?.id === 'osu_color_thermal' && hasMissingSequences.value)
const hasMissingSequences = computed(() => (dataset.value?.sequences || []).some(s => s.missing))

function outputUrl(path) {
  return getOutputUrl(path)
}

function contourEntries(seq) {
  return Object.entries(seq.contour || {})
}

function contourCount(seq) {
  return Object.keys(seq.contour || {}).length
}

function fmtFps(v) {
  return v && !isNaN(v) ? Number(v).toFixed(2) : '-'
}

async function handleDownload() {
  downloading.value = true
  try {
    const res = await downloadDataset(datasetId)
    message.success(res?.note || '下载已启动')
    // 轮询刷新：每 5 秒刷新一次，最多 6 次（30 秒）
    let checks = 0
    const timer = setInterval(async () => {
      checks++
      await load()
      if (!hasMissingSequences.value || checks >= 6) {
        clearInterval(timer)
        if (!hasMissingSequences.value) message.success('OSU Thermal 下载完成')
        downloading.value = false
      }
    }, 5000)
  } catch (e) {
    message.error(e.message || '下载失败')
    downloading.value = false
  }
}

async function load() {
  loading.value = true
  try {
    dataset.value = await getDatasetDetail(datasetId)
    if (dataset.value?.kind === 'image') loadPreview()
  } catch {
    message.error('数据集不存在')
    router.push('/evaluation/datasets')
  }
  loading.value = false
}

async function loadPreview() {
  previewLoading.value = true
  try {
    const res = await getDatasetPreview(datasetId, { method: previewMethod.value, n: 8 })
    imagePreviews.value = res?.previews || []
    if (!imagePreviews.value.length) message.warning('未生成预览（parquet 读取失败？）')
  } catch (e) {
    message.error(e.message || '预览生成失败')
    imagePreviews.value = []
  }
  previewLoading.value = false
}

onMounted(load)
</script>

<style scoped lang="scss">
.dataset-header h2 {
  margin: 0 0 6px;
  font-size: 18px;
}
.meta-line { align-items: center; }
.source-link {
  font-size: 12px;
  color: var(--color-primary);
  text-decoration: none;
  &:hover { text-decoration: underline; }
}
.desc {
  margin: 12px 0 0;
  font-size: 13px;
  color: var(--color-text-secondary);
  line-height: 1.5;
}
.flex-between {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.seq-header {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.seq-name {
  font-weight: 600;
  font-size: 14px;
}

.seq-compare {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin: 8px 0 4px;
}
.compare-cell {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;  /* let video shrink in grid */
}
.cell-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.cmp-video {
  width: 100%;
  max-height: 360px;
  background: #000;
  border-radius: 8px;
  object-fit: contain;
}
.cmp-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
  border: 1px dashed var(--color-border-light);
  border-radius: 8px;
  color: var(--color-text-dim);
  font-size: 13px;
}

.preview-img {
  width: 100%;
  display: block;
  border-radius: 8px;
}
.gallery-img {
  flex: 0 0 auto;
  width: 140px;
  height: 100px;
  object-fit: cover;
  border-radius: 6px;
  cursor: pointer;
  background: var(--color-elevated);
  border: 1px solid var(--color-border-light);
  transition: transform 0.15s;
  &:hover { transform: scale(1.05); }
}
.preview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 12px;
}
.preview-pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  padding: 8px;
  border: 1px solid var(--color-border-light);
  border-radius: 8px;
  background: var(--color-elevated);
}
.preview-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: center;
}
.preview-cell .gallery-img {
  width: 100%;
  height: auto;
  max-height: 140px;
}
.vlabel {
  font-size: 11px;
  color: var(--color-text-dim);
}
.dim { color: var(--color-text-dim); }
.small { font-size: 12px; }
</style>
