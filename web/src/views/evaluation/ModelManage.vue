<template>
  <div class="page-container">
    <n-card title="Codec" size="small">
      <n-spin :show="loading">
        <n-grid v-if="codecs.length" :cols="3" :x-gap="16" :y-gap="16">
          <n-gi v-for="c in codecs" :key="c.id">
            <n-card size="small" hoverable>
              <template #header>
                <div class="flex-between">
                  <h4>{{ c.id }}</h4>
                  <n-tag size="small" :type="c.kind === 'codec' ? 'success' : 'warning'">{{ c.kind }}</n-tag>
                </div>
              </template>
              <n-descriptions :column="1" size="small">
                <n-descriptions-item label="名称">{{ c.name || '-' }}</n-descriptions-item>
                <n-descriptions-item label="family">{{ c.family || '-' }}</n-descriptions-item>
                <n-descriptions-item label="qualities">{{ (c.qualities || []).join(', ') || '-' }}</n-descriptions-item>
                <n-descriptions-item label="ext">{{ c.ext || '-' }}</n-descriptions-item>
                <n-descriptions-item label="说明">{{ c.description || '-' }}</n-descriptions-item>
              </n-descriptions>
            </n-card>
          </n-gi>
        </n-grid>
        <EmptyState v-else description="暂无 codec（benchmark/video/codecs 注册表为空）" />
      </n-spin>
    </n-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { NCard, NSpin, NGrid, NGi, NTag, NDescriptions, NDescriptionsItem } from 'naive-ui'
import EmptyState from '../../components/common/EmptyState.vue'
import { getCodecs } from '../../api/evaluation'

const loading = ref(false)
const codecs = ref([])

onMounted(async () => {
  loading.value = true
  try { codecs.value = await getCodecs() } catch { codecs.value = [] }
  loading.value = false
})
</script>
