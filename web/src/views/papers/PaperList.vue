<template>
  <div class="page-container">
    <n-card size="small">
      <template #header>
        <div class="flex-between">
          <div class="header-left">
            <h3>论文列表</h3>
            <n-tag v-if="stats.total" size="small" type="info" round>共 {{ stats.total }} 篇</n-tag>
          </div>
        </div>
      </template>

      <!-- 统计栏 -->
      <div v-if="stats.total" class="stats-bar">
        <div class="stat-item">
          <span class="stat-num">{{ stats.total }}</span>
          <span class="stat-label">论文总数</span>
        </div>
        <div class="stat-divider" />
        <template v-for="(count, source) in (stats.by_source || {})" :key="source">
          <div class="stat-item">
            <span class="stat-num">{{ count }}</span>
            <span class="stat-label">{{ source === 'daily_paper' ? 'arXiv日报' : '深度精读' }}</span>
          </div>
          <div class="stat-divider" />
        </template>
        <div class="stat-item">
          <span class="stat-num">{{ starCount }}</span>
          <span class="stat-label">收藏</span>
        </div>
        <div class="stat-divider" />
        <div class="stat-item">
          <span class="stat-num">{{ pinCount }}</span>
          <span class="stat-label">置顶</span>
        </div>
      </div>

      <!-- 筛选工具栏 -->
      <div class="filter-toolbar">
        <n-space align="center">
          <n-input v-model:value="searchQuery" placeholder="搜索标题/摘要/标签" clearable size="small" style="width: 220px" @update:value="onSearchChange" />
          <n-select v-model:value="selectedCategory" :options="categoryOptions" placeholder="研究方向" clearable size="small" style="width: 180px" @update:value="onFilterChange" />
          <n-select v-model:value="selectedSource" :options="sourceOptions" placeholder="来源" clearable size="small" style="width: 120px" @update:value="onFilterChange" />
        </n-space>
        <n-space align="center">
          <n-button
            size="small"
            :type="starredOnly ? 'warning' : 'default'"
            :secondary="!starredOnly"
            @click="starredOnly = !starredOnly; currentPage = 1"
          >收藏</n-button>
          <n-button-group size="small">
            <n-button :type="sortBy === 'newest' ? 'primary' : 'default'" @click="sortBy = 'newest'" title="最新优先">↓</n-button>
            <n-button :type="sortBy === 'oldest' ? 'primary' : 'default'" @click="sortBy = 'oldest'" title="最早优先">↑</n-button>
            <n-button :type="sortBy === 'title' ? 'primary' : 'default'" @click="sortBy = 'title'" title="标题排序">Aa</n-button>
          </n-button-group>
        </n-space>
      </div>

      <n-spin :show="loading">
        <div v-if="filteredPapers.length" class="papers-grid">
          <div v-for="p in paginatedPapers" :key="p.id" class="paper-card" :class="{ pinned: p.pinned }" @click="openDetail(p.id)">
            <!-- 缩略图 -->
            <div class="paper-thumb">
              <img
                v-if="p.arxiv_id && !failedThumbs.has(p.arxiv_id)"
                :src="thumbnailUrl(p.arxiv_id)"
                :alt="p.title"
                @error="onThumbError(p.arxiv_id)"
                loading="lazy"
              />
              <div v-else class="placeholder">
                <span class="icon">📄</span>
                <span>{{ p.source === 'blog' ? '精读' : '论文' }}</span>
              </div>
            </div>
            <!-- 内容 -->
            <div class="paper-content">
              <h4 class="paper-title">{{ p.title_zh || p.title }}</h4>
              <div v-if="p.title_zh" class="paper-title-en">{{ p.title }}</div>
              <div class="paper-meta">
                <n-tag v-for="cat in (p.categories || [])" :key="cat" size="tiny" :type="categoryColor(cat)" round>{{ categoryLabel(cat) }}</n-tag>
                <n-tag v-if="p.source === 'blog'" size="tiny" type="error" round>精读</n-tag>
                <n-tag v-if="p.venue" size="tiny" type="default" round>{{ p.venue }}</n-tag>
                <span v-if="p.published_at_str" class="paper-date">{{ p.published_at_str }}</span>
              </div>
              <p class="paper-authors">{{ formatAuthors(p.authors) }}</p>
              <p class="paper-abstract">{{ p.abstract_zh || p.abstract }}</p>
              <div class="paper-tags" v-if="p.tags && p.tags.length">
                <span v-for="tag in p.tags.slice(0, 5)" :key="tag" class="paper-tag">{{ tag }}</span>
              </div>
              <div class="paper-footer">
                <div class="paper-links">
                  <a v-if="p.url" :href="p.url" target="_blank" class="paper-link" @click.stop>arXiv</a>
                  <a v-if="p.pdf_url" :href="p.pdf_url" target="_blank" class="paper-link" @click.stop>PDF</a>
                  <a v-if="p.github_url" :href="p.github_url" target="_blank" class="paper-link code" @click.stop>Code</a>
                </div>
                <div class="paper-actions">
                  <span v-if="p.summary_zh" class="badge-icon" title="有AI概述">✨</span>
                  <span v-if="p.has_note" class="badge-icon" title="有笔记">📝</span>
                  <button class="action-btn" :class="{ active: p.pinned }" @click.stop="handlePin(p)" title="置顶">📌</button>
                  <button class="action-btn" :class="{ active: p.starred }" @click.stop="handleStar(p)" title="收藏">⭐</button>
                </div>
              </div>
            </div>
          </div>
        </div>
        <EmptyState v-else description="暂无论文数据" />
      </n-spin>

      <!-- 分页 -->
      <div v-if="filteredPapers.length > pageSize" class="pagination-wrap">
        <n-pagination
          v-model:page="currentPage"
          :page-count="Math.ceil(filteredPapers.length / pageSize)"
          :page-size="pageSize"
        />
      </div>
    </n-card>

    <!-- 论文详情浮窗 -->
    <n-modal v-model:show="showDetailModal" :mask-closable="true" @update:show="onDetailModalUpdate" class="detail-modal-wrap">
      <div class="detail-modal-body">
        <n-spin :show="detailLoading">
          <div v-if="detailPaper" class="detail-inner">
            <!-- 头部 -->
            <div class="dm-header">
              <div class="dm-header-info">
                <h2>{{ detailPaper.title_zh || detailPaper.title }}
                  <span v-if="detailPaper.summary_zh" class="dm-ai-badge" title="已生成 AI 概述">✨</span>
                </h2>
                <p v-if="detailPaper.title_zh" class="dm-title-en">{{ detailPaper.title }}</p>
                <p class="dm-authors">{{ formatAllAuthors(detailPaper.authors) }}</p>
                <div class="dm-tags">
                  <n-tag v-for="cat in (detailPaper.categories || [])" :key="cat" size="tiny" :type="categoryColor(cat)" round>{{ categoryLabel(cat) }}</n-tag>
                  <n-tag v-if="detailPaper.source === 'blog'" size="tiny" type="error" round>精读</n-tag>
                  <n-tag v-if="detailPaper.venue" size="tiny" type="default" round>{{ detailPaper.venue }}</n-tag>
                  <span v-if="detailPaper.published_at_str" class="dm-date">{{ detailPaper.published_at_str }}</span>
                </div>
                <div v-if="detailPaper.tags && detailPaper.tags.length" class="dm-tags-list">
                  <span v-for="tag in detailPaper.tags" :key="tag" class="dm-tag">{{ tag }}</span>
                </div>
              </div>
              <div class="dm-actions">
                <n-button :type="detailPaper.starred ? 'warning' : 'default'" :secondary="!detailPaper.starred" @click="handleDetailStar">
                  {{ detailPaper.starred ? '已收藏' : '收藏' }}
                </n-button>
                <n-button :type="detailPaper.pinned ? 'info' : 'default'" :secondary="!detailPaper.pinned" @click="handleDetailPin">
                  {{ detailPaper.pinned ? '已置顶' : '置顶' }}
                </n-button>
                <n-button v-if="detailPaper.url" tag="a" :href="detailPaper.url" target="_blank" secondary>arXiv</n-button>
                <n-button v-if="detailPaper.pdf_url" tag="a" :href="detailPaper.pdf_url" target="_blank" secondary>PDF</n-button>
                <n-button v-if="detailPaper.github_url" tag="a" :href="detailPaper.github_url" target="_blank" type="success" secondary>Code</n-button>
              </div>
            </div>
            <!-- 摘要 + 缩略图 -->
            <div class="dm-body">
              <div v-if="detailPaper.arxiv_id" class="dm-thumb">
                <img :src="thumbnailUrl(detailPaper.arxiv_id)" :alt="detailPaper.title" @error="e => e.target.style.display = 'none'" />
              </div>
              <div class="dm-text">
                <h3>摘要</h3>
                <p v-if="detailPaper.abstract_zh" class="dm-abstract-zh">{{ detailPaper.abstract_zh }}</p>
                <p class="dm-abstract-en">{{ detailPaper.abstract }}</p>
                <template v-if="detailPaper.summary_zh">
                  <n-divider />
                  <h3>AI 概述</h3>
                  <p class="dm-summary">{{ detailPaper.summary_zh }}</p>
                </template>
              </div>
            </div>
            <!-- 笔记 -->
            <div class="dm-notes">
              <div class="dm-notes-header">
                <h3>精读笔记</h3>
                <n-space>
                  <n-button v-if="!editingNote" secondary @click="startEditNote">编辑笔记</n-button>
                  <template v-else>
                    <n-button secondary @click="cancelEditNote">取消</n-button>
                    <n-button type="primary" :loading="savingNote" @click="saveNote">保存</n-button>
                  </template>
                </n-space>
              </div>
              <MarkdownRenderer v-if="!editingNote && detailNote" :content="detailNote" />
              <n-input v-else-if="editingNote" v-model:value="noteDraft" type="textarea" :rows="8" placeholder="支持 Markdown 格式..." />
              <p v-else class="dm-no-note">暂无笔记，点击「编辑」开始记录</p>
            </div>
          </div>
        </n-spin>
      </div>
    </n-modal>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  NCard, NSpin, NSpace, NInput, NSelect, NTag, NButton, NButtonGroup, NDivider,
  NPagination, NModal, useMessage,
} from 'naive-ui'
import EmptyState from '../../components/common/EmptyState.vue'
import MarkdownRenderer from '../../components/common/MarkdownRenderer.vue'
import {
  getPaperList, getPaperStats, getThumbnailUrl,
  starPaper, pinPaper, getPaperDetail, getPaperNote, savePaperNote,
} from '../../api/papers'

const route = useRoute()
const router = useRouter()
const message = useMessage()

const loading = ref(false)
const papers = ref([])
const stats = ref({ total: 0, by_category: {}, by_source: {} })
const searchQuery = ref('')
const selectedCategory = ref(null)
const selectedSource = ref(null)
const sortBy = ref('newest')
const starredOnly = ref(false)
const currentPage = ref(1)
const pageSize = 12
const failedThumbs = ref(new Set())

// 详情浮窗状态
const showDetailModal = ref(false)
const detailPaper = ref(null)
const detailLoading = ref(false)
const detailNote = ref('')
const editingNote = ref(false)
const noteDraft = ref('')
const savingNote = ref(false)
const detailPaperId = ref(null)

const starCount = computed(() => papers.value.filter(p => p.starred).length)
const pinCount = computed(() => papers.value.filter(p => p.pinned).length)

const categoryLabels = {
  learned_compression: '学习式压缩',
  generative_compression: '生成式压缩',
  jscc: '联合信源信道编码',
  video_compression: '视频压缩',
  infrared_compression: '红外图像压缩',
  tokenization: '视觉标记化',
  distillation: '知识蒸馏',
}

const sourceOptions = [
  { label: 'arXiv日报', value: 'daily_paper' },
  { label: '深度精读', value: 'blog' },
]

const categoryOptions = computed(() => {
  const cats = stats.value.by_category || {}
  return Object.entries(cats).map(([k, v]) => ({ label: `${categoryLabels[k] || k} (${v})`, value: k }))
})

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

const filteredPapers = computed(() => {
  let result = papers.value
  if (searchQuery.value) {
    const q = searchQuery.value.toLowerCase()
    result = result.filter(p =>
      (p.title || '').toLowerCase().includes(q) ||
      (p.title_zh || '').toLowerCase().includes(q) ||
      (p.abstract || '').toLowerCase().includes(q) ||
      (p.abstract_zh || '').toLowerCase().includes(q) ||
      (p.tags || []).some(t => t.toLowerCase().includes(q))
    )
  }
  if (selectedCategory.value) {
    result = result.filter(p => (p.categories || []).includes(selectedCategory.value))
  }
  if (selectedSource.value) {
    result = result.filter(p => p.source === selectedSource.value)
  }
  if (starredOnly.value) result = result.filter(p => p.starred)
  return [...result].sort((a, b) => {
    if (a.pinned && !b.pinned) return -1
    if (!a.pinned && b.pinned) return 1
    if (sortBy.value === 'newest') return new Date(b.published_at || 0) - new Date(a.published_at || 0)
    if (sortBy.value === 'oldest') return new Date(a.published_at || 0) - new Date(b.published_at || 0)
    if (sortBy.value === 'title') return (a.title_zh || a.title || '').localeCompare(b.title_zh || b.title || '')
    return 0
  })
})

const paginatedPapers = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return filteredPapers.value.slice(start, start + pageSize)
})

function thumbnailUrl(arxivId) {
  return getThumbnailUrl(arxivId)
}

function onThumbError(arxivId) {
  failedThumbs.value.add(arxivId)
  failedThumbs.value = new Set(failedThumbs.value)
}

function formatAuthors(authors) {
  if (!authors || !authors.length) return ''
  const list = Array.isArray(authors) ? authors : [authors]
  return list.slice(0, 3).join(', ') + (list.length > 3 ? ` 等 ${list.length} 人` : '')
}

function formatAllAuthors(authors) {
  if (!authors || !authors.length) return ''
  const list = Array.isArray(authors) ? authors : [authors]
  return list.join(', ')
}

function onSearchChange() {
  currentPage.value = 1
}

function onFilterChange() {
  currentPage.value = 1
}

async function loadPapers() {
  loading.value = true
  try {
    const data = getPaperList()
    papers.value = data.papers || []
  } catch { papers.value = [] }
  loading.value = false
}

async function loadStats() {
  try { stats.value = getPaperStats() } catch {}
}

async function handleStar(p) {
  const newVal = !p.starred
  p.starred = newVal
  try {
    starPaper(p.id, newVal)
    message.success(newVal ? '已收藏' : '已取消收藏')
  } catch {
    p.starred = !newVal
    message.error('操作失败')
  }
}

async function handlePin(p) {
  const newVal = !p.pinned
  p.pinned = newVal
  try {
    pinPaper(p.id, newVal)
    message.success(newVal ? '已置顶' : '已取消置顶')
  } catch {
    p.pinned = !newVal
    message.error('操作失败')
  }
}

onMounted(() => {
  loadPapers()
  loadStats()
  // 从 URL 读取 category 参数
  if (route.query.category) {
    selectedCategory.value = route.query.category
  }
  // 恢复浮窗状态
  const hash = window.location.hash
  if (hash.startsWith('#paper=')) {
    const id = hash.substring(7)
    if (id) {
      detailPaperId.value = id
      showDetailModal.value = true
      loadDetail(id)
    }
  }
})

// ---- 详情浮窗功能 ----
function openDetail(id) {
  detailPaperId.value = id
  showDetailModal.value = true
  loadDetail(id)
  history.replaceState(null, '', `#paper=${id}`)
}

function onDetailModalUpdate(show) {
  if (!show) {
    detailPaper.value = null
    detailNote.value = ''
    editingNote.value = false
    detailPaperId.value = null
    history.replaceState(null, '', window.location.pathname + window.location.search)
  }
}

async function loadDetail(id) {
  detailLoading.value = true
  try {
    const p = getPaperDetail(id)
    const n = getPaperNote(id)
    detailPaper.value = p
    detailNote.value = n?.content || ''
  } catch {}
  detailLoading.value = false
}

function startEditNote() { noteDraft.value = detailNote.value; editingNote.value = true }
function cancelEditNote() { editingNote.value = false }

async function saveNote() {
  savingNote.value = true
  try {
    savePaperNote(detailPaperId.value, noteDraft.value)
    detailNote.value = noteDraft.value
    editingNote.value = false
    // 更新列表中的 has_note
    const p = papers.value.find(p => p.id === detailPaperId.value)
    if (p) p.has_note = true
    message.success('笔记已保存')
  } catch { message.error('保存失败') }
  savingNote.value = false
}

async function handleDetailStar() {
  const newVal = !detailPaper.value.starred
  detailPaper.value.starred = newVal
  const p = papers.value.find(p => p.id === detailPaperId.value)
  if (p) p.starred = newVal
  try {
    starPaper(detailPaperId.value, newVal)
    message.success(newVal ? '已收藏' : '已取消收藏')
  } catch {
    detailPaper.value.starred = !newVal
    if (p) p.starred = !newVal
    message.error('操作失败')
  }
}

async function handleDetailPin() {
  const newVal = !detailPaper.value.pinned
  detailPaper.value.pinned = newVal
  const p = papers.value.find(p => p.id === detailPaperId.value)
  if (p) p.pinned = newVal
  try {
    pinPaper(detailPaperId.value, newVal)
    message.success(newVal ? '已置顶' : '已取消置顶')
  } catch {
    detailPaper.value.pinned = !newVal
    if (p) p.pinned = !newVal
    message.error('操作失败')
  }
}
</script>

<style scoped lang="scss">
.flex-between { flex-wrap: wrap; gap: 8px; }
.header-left { display: flex; align-items: center; gap: 10px; }

.filter-toolbar {
  display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;
  flex-wrap: wrap; gap: 10px;
}

.stats-bar {
  display: flex; gap: 16px; align-items: center; margin-bottom: 14px; padding: 10px 16px;
  background: #f8fafc; border-radius: 8px; flex-wrap: wrap;
}
.stat-item { display: flex; align-items: baseline; gap: 5px; }
.stat-num { font-size: 20px; font-weight: bold; color: #4f46e5; }
.stat-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-divider { width: 1px; height: 20px; background: #e2e8f0; }

.papers-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 14px; }

.paper-card {
  display: flex; flex-direction: column; padding: 14px; background: #fff;
  border: 1px solid #e2e8f0; border-radius: 10px; transition: all 0.2s; cursor: pointer;
  &:hover { border-color: #4f46e544; box-shadow: 0 4px 16px rgba(0,0,0,0.08); transform: translateY(-2px); }
  &.pinned { border-left: 3px solid #4f46e5; background: #fafaff; }
}

.paper-thumb {
  border-radius: 8px; overflow: hidden; background: #f1f5f9; margin-bottom: 10px;
  display: flex; align-items: center; justify-content: center; min-height: 100px; max-height: 140px;
  img { width: 100%; height: 100%; display: block; object-fit: cover; }
  .placeholder { color: #94a3b8; font-size: 12px; text-align: center; padding: 20px; .icon { font-size: 28px; display: block; margin-bottom: 6px; opacity: 0.5; } }
}

.paper-content { display: flex; flex-direction: column; min-width: 0; flex: 1; }
.paper-title { font-size: 14px; font-weight: 600; line-height: 1.5; margin-bottom: 4px; color: #1e293b; }
.paper-title-en { font-size: 12px; color: #94a3b8; margin-bottom: 6px; font-style: italic; }
.paper-meta { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px; align-items: center; }
.paper-date { font-size: 11px; color: #94a3b8; margin-left: 4px; }
.paper-authors { font-size: 12px; color: #64748b; margin-bottom: 8px; }
.paper-abstract {
  font-size: 13px; color: #475569; line-height: 1.6; margin-bottom: 8px;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.paper-tags { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 8px; }
.paper-tag {
  font-size: 11px; padding: 1px 8px; border-radius: 999px;
  background: #f1f5f9; color: #64748b;
}
.paper-footer { display: flex; justify-content: space-between; align-items: center; margin-top: auto; padding-top: 8px; }
.paper-links { display: flex; gap: 6px; }
.paper-link {
  display: inline-flex; padding: 3px 10px; border-radius: 999px;
  border: 1px solid #e2e8f0; font-size: 12px; color: #4f46e5;
  text-decoration: none; transition: all 0.15s;
  &:hover { background: #4f46e5; color: #fff; border-color: #4f46e5; }
  &.code { color: #10b981; &:hover { background: #10b981; border-color: #10b981; } }
}
.paper-actions { display: flex; gap: 4px; align-items: center; }
.badge-icon { font-size: 14px; opacity: 0.7; }
.action-btn {
  background: none; border: none; cursor: pointer; font-size: 16px;
  padding: 2px 4px; border-radius: 4px; opacity: 0.25;
  transition: all 0.15s; line-height: 1;
  &:hover { opacity: 0.6; background: #f1f5f9; }
  &.active { opacity: 1; }
}

.pagination-wrap { display: flex; justify-content: center; margin-top: 16px; }

// ---- 详情浮窗 ----
.detail-modal-body {
  background: #fff; border-radius: 12px; width: 80vw; max-width: 800px;
  max-height: 85vh; overflow-y: auto; padding: 24px;
}
.detail-inner { display: flex; flex-direction: column; gap: 16px; }
.dm-header {
  display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  h2 { font-size: 18px; line-height: 1.4; margin-bottom: 4px; }
}
.dm-header-info { flex: 1; min-width: 0; }
.dm-title-en { font-size: 13px; color: #94a3b8; font-style: italic; margin-bottom: 4px; }
.dm-authors { font-size: 13px; color: #64748b; margin-bottom: 8px; }
.dm-tags { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }
.dm-tags-list { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 6px; }
.dm-tag {
  font-size: 11px; padding: 2px 8px; border-radius: 999px;
  background: #f1f5f9; color: #64748b;
}
.dm-date { font-size: 12px; color: #94a3b8; margin-left: 6px; }
.dm-actions { display: flex; gap: 6px; flex-shrink: 0; flex-wrap: wrap; }

.dm-body { display: flex; gap: 20px; flex-wrap: wrap; }
.dm-thumb { width: 200px; flex-shrink: 0; img { width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; } }
.dm-text { flex: 1; min-width: 200px; h3 { font-size: 15px; margin-bottom: 8px; } }
.dm-abstract-zh { font-size: 14px; color: #1e293b; line-height: 1.7; margin-bottom: 12px; }
.dm-abstract-en { font-size: 13px; color: #64748b; line-height: 1.6; }
.dm-summary { font-size: 14px; color: #334155; line-height: 1.7; background: #f8fafc; padding: 12px; border-radius: 8px; }

.dm-notes { border-top: 1px solid #e2e8f0; padding-top: 16px; }
.dm-notes-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; h3 { font-size: 15px; } }
.dm-no-note { color: #94a3b8; font-size: 13px; }

.dm-ai-badge { font-size: 16px; cursor: default; vertical-align: middle; margin-left: 6px; }
</style>
