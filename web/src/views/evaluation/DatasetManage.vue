<template>
  <div class="page-container">
    <n-card title="数据集" size="small">
      <n-spin :show="loading">
        <n-grid v-if="datasets.length" :cols="3" :x-gap="16" :y-gap="16">
          <n-gi v-for="d in datasets" :key="d.id">
            <n-card size="small" hoverable class="dataset-card" @click="goDetail(d)">
              <template #header>
                <div class="flex-between">
                  <h4>{{ d.name }}</h4>
                  <n-tag v-if="d.kind === 'contour'" size="small" type="warning">轮廓</n-tag>
                  <n-tag v-else-if="d.kind === 'image'" size="small" type="success">图像</n-tag>
                  <n-tag v-else size="small" type="info">原始</n-tag>
                </div>
              </template>
              <n-descriptions :column="1" size="small">
                <n-descriptions-item v-if="d.kind === 'image'" label="用途">{{ d.usage === 'speed' ? 'speed run（检验）' : 'formal（测试）' }}</n-descriptions-item>
                <n-descriptions-item v-if="d.kind === 'image'" label="采样图数">{{ d.sample_images }} 张</n-descriptions-item>
                <template v-else>
                  <n-descriptions-item label="序列数">{{ d.sequences?.length ?? 0 }}</n-descriptions-item>
                  <n-descriptions-item label="轮廓方法">
                    <span v-if="d.contour_methods?.length">{{ d.contour_methods.join(', ') }}</span>
                    <span v-else class="dim">—</span>
                  </n-descriptions-item>
                  <n-descriptions-item label="状态">
                    <n-tag v-if="hasMissing(d)" size="small" type="error">文件缺失</n-tag>
                    <n-tag v-else size="small" type="success">可用</n-tag>
                  </n-descriptions-item>
                </template>
              </n-descriptions>
              <p v-if="d.description" class="desc">{{ d.description }}</p>
            </n-card>
          </n-gi>
        </n-grid>
        <EmptyState v-else description="暂无数据集数据" />
      </n-spin>
    </n-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { NCard, NSpin, NGrid, NGi, NTag, NDescriptions, NDescriptionsItem } from 'naive-ui'
import EmptyState from '../../components/common/EmptyState.vue'
import { getDatasets } from '../../api/evaluation'

const router = useRouter()
const loading = ref(false)
const datasets = ref([])

function hasMissing(d) {
  return (d.sequences || []).some(s => s.missing)
}

function goDetail(d) {
  router.push(`/evaluation/datasets/${d.id}`)
}

onMounted(async () => {
  loading.value = true
  try { datasets.value = await getDatasets() } catch { datasets.value = [] }
  loading.value = false
})
</script>

<style scoped lang="scss">
.dataset-card {
  cursor: pointer;
  transition: transform 0.15s, box-shadow 0.15s;
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  }
}
.flex-between {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}
h4 {
  margin: 0;
  font-size: 15px;
}
.desc {
  margin: 10px 0 0;
  font-size: 12px;
  color: var(--color-text-dim);
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.dim { color: var(--color-text-dim); }
</style>
