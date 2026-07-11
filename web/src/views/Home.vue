<template>
  <div class="page-container home-page">
    <!-- 欢迎区 -->
    <n-card class="welcome-card">
      <div class="welcome-inner">
        <h1>红外图像压缩论文库</h1>
        <p class="welcome-desc">收集整理图像压缩领域的前沿论文，涵盖学习式压缩、生成式压缩、联合信源信道编码、视频压缩、红外图像压缩等方向</p>
        <n-button type="primary" @click="router.push('/papers/list')">浏览论文库 →</n-button>
      </div>
    </n-card>

    <!-- 统计卡片 -->
    <n-grid :cols="4" :x-gap="16" :y-gap="16" style="margin-top: 16px">
      <n-gi>
        <n-card class="stat-card" hoverable>
          <div class="stat-icon total">📚</div>
          <div class="stat-info">
            <span class="stat-num">{{ stats.total }}</span>
            <span class="stat-label">论文总数</span>
          </div>
        </n-card>
      </n-gi>
      <n-gi>
        <n-card class="stat-card" hoverable>
          <div class="stat-icon daily">📰</div>
          <div class="stat-info">
            <span class="stat-num">{{ stats.by_source?.daily_paper || 0 }}</span>
            <span class="stat-label">arXiv 日报</span>
          </div>
        </n-card>
      </n-gi>
      <n-gi>
        <n-card class="stat-card" hoverable>
          <div class="stat-icon blog">📝</div>
          <div class="stat-info">
            <span class="stat-num">{{ stats.by_source?.blog || 0 }}</span>
            <span class="stat-label">深度精读</span>
          </div>
        </n-card>
      </n-gi>
      <n-gi>
        <n-card class="stat-card" hoverable>
          <div class="stat-icon cat">🗂️</div>
          <div class="stat-info">
            <span class="stat-num">{{ Object.keys(stats.by_category || {}).length }}</span>
            <span class="stat-label">研究方向</span>
          </div>
        </n-card>
      </n-gi>
    </n-grid>

    <!-- 分类分布 + 技术路线 -->
    <n-grid :cols="2" :x-gap="16" :y-gap="16" style="margin-top: 16px">
      <n-gi>
        <n-card title="研究方向分布" size="small">
          <div class="category-list">
            <div v-for="(count, cat) in (stats.by_category || {})" :key="cat" class="category-item" @click="router.push(`/papers/list?category=${cat}`)">
              <span class="cat-name">{{ categoryLabel(cat) }}</span>
              <n-progress
                type="line"
                :percentage="Math.round(count / stats.total * 100)"
                :show-indicator="false"
                :height="8"
                :border-radius="4"
                style="flex: 1; margin: 0 12px;"
              />
              <span class="cat-count">{{ count }}</span>
            </div>
          </div>
        </n-card>
      </n-gi>
      <n-gi>
        <n-card title="技术路线" size="small">
          <n-space direction="vertical" :size="8">
            <n-tag v-for="route in techRoutes" :key="route" type="info" size="small" round>
              {{ route }}
            </n-tag>
          </n-space>
        </n-card>
      </n-gi>
    </n-grid>

    <!-- 最近论文 -->
    <n-card title="最近添加" size="small" style="margin-top: 16px">
      <n-spin :show="loading">
        <div v-if="recentPapers.length" class="recent-list">
          <div v-for="p in recentPapers" :key="p.id" class="recent-item" @click="router.push(`/papers/${p.id}`)">
            <div class="recent-info">
              <span class="recent-title">{{ p.title_zh || p.title }}</span>
              <span class="recent-meta">{{ p.published_at_str }} · {{ p.venue }}</span>
            </div>
            <n-tag size="tiny" :type="categoryColor(p.categories?.[0])" round>{{ categoryLabel(p.categories?.[0]) }}</n-tag>
          </div>
        </div>
        <EmptyState v-else description="暂无论文数据" />
      </n-spin>
    </n-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  NCard, NGrid, NGi, NSpin, NTag, NSpace, NButton, NProgress,
} from 'naive-ui'
import EmptyState from '../components/common/EmptyState.vue'
import { getPaperStats, getPaperList } from '../api/papers'

const router = useRouter()

const stats = ref({ total: 0, by_category: {}, by_source: {} })
const recentPapers = ref([])
const loading = ref(false)

const techRoutes = [
  '学习式图像压缩 (LIC)', '生成式压缩 (HiFiC/CDC)', '扩散模型压缩',
  '联合信源信道编码 (JSCC)', '视频压缩 (LVC)', '红外图像压缩',
  '视觉标记化 (Tokenization)', '知识蒸馏', '小波变换',
]

const categoryLabels = {
  learned_compression: '学习式压缩',
  generative_compression: '生成式压缩',
  jscc: '联合信源信道编码',
  video_compression: '视频压缩',
  infrared_compression: '红外图像压缩',
  tokenization: '视觉标记化',
  distillation: '知识蒸馏',
}

function categoryLabel(cat) {
  return categoryLabels[cat] || cat
}

function categoryColor(cat) {
  const map = {
    learned_compression: 'info',
    generative_compression: 'success',
    jscc: 'warning',
    video_compression: 'error',
    infrared_compression: 'info',
    tokenization: 'success',
    distillation: 'warning',
  }
  return map[cat] || 'default'
}

onMounted(async () => {
  loading.value = true
  try {
    stats.value = await getPaperStats()
    const data = await getPaperList({ limit: 500 })
    // 按日期降序取最近10篇
    recentPapers.value = [...(data.papers || [])]
      .sort((a, b) => new Date(b.published_at || 0) - new Date(a.published_at || 0))
      .slice(0, 10)
  } catch {}
  loading.value = false
})
</script>

<style scoped lang="scss">
.welcome-card {
  background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
  border: none;
  :deep(.n-card__content) { padding: 32px; }
}
.welcome-inner {
  color: #fff;
  h1 { font-size: 28px; margin-bottom: 12px; }
  .welcome-desc { font-size: 14px; opacity: 0.9; margin-bottom: 20px; line-height: 1.6; }
}

.stat-card {
  :deep(.n-card__content) { display: flex; align-items: center; gap: 16px; }
}
.stat-icon {
  width: 48px; height: 48px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 24px; flex-shrink: 0;
  &.total { background: #eef2ff; }
  &.daily { background: #fef3c7; }
  &.blog { background: #dcfce7; }
  &.cat { background: #fce7f3; }
}
.stat-info { display: flex; flex-direction: column; }
.stat-num { font-size: 24px; font-weight: 700; color: var(--color-text-heading); }
.stat-label { font-size: 12px; color: var(--color-text-secondary); }

.category-list { display: flex; flex-direction: column; gap: 10px; }
.category-item {
  display: flex; align-items: center; cursor: pointer;
  padding: 4px 8px; border-radius: 6px; transition: background 0.15s;
  &:hover { background: var(--color-elevated); }
  .cat-name { font-size: 13px; width: 120px; flex-shrink: 0; color: var(--color-text-secondary); }
  .cat-count { font-size: 13px; font-weight: 600; color: var(--color-primary); width: 30px; text-align: right; }
}

.recent-list { display: flex; flex-direction: column; gap: 8px; }
.recent-item {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 12px; border-radius: 8px; cursor: pointer; transition: all 0.15s;
  border: 1px solid transparent;
  &:hover { background: var(--color-elevated); border-color: #e2e8f0; }
}
.recent-info { display: flex; flex-direction: column; gap: 2px; min-width: 0; flex: 1; }
.recent-title {
  font-size: 14px; color: var(--color-text-heading); font-weight: 500;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.recent-meta { font-size: 12px; color: var(--color-text-dim); }
</style>
