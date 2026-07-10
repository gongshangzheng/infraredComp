<template>
  <div class="page-container">
    <n-spin :show="loading">
      <div v-if="paper" class="paper-detail">
        <!-- 顶部信息区 -->
        <n-card>
          <template #header>
            <div class="detail-header">
              <div class="header-info">
                <h2>{{ paper.title_zh || paper.title }}</h2>
                <p v-if="paper.title_zh" class="title-en">{{ paper.title }}</p>
                <p class="authors">{{ formatAuthors(paper.authors) }}</p>
                <div class="meta-tags">
                  <n-tag v-for="cat in (paper.categories || [])" :key="cat" size="small" :type="categoryColor(cat)" round>{{ categoryLabel(cat) }}</n-tag>
                  <n-tag v-if="paper.source === 'blog'" size="small" type="error" round>精读</n-tag>
                  <n-tag v-if="paper.venue" size="small" type="default" round>{{ paper.venue }}</n-tag>
                  <span v-if="paper.published_at_str" class="date">{{ paper.published_at_str }}</span>
                </div>
                <div v-if="paper.tags && paper.tags.length" class="tags-list">
                  <span v-for="tag in paper.tags" :key="tag" class="detail-tag">{{ tag }}</span>
                </div>
              </div>
              <div class="header-actions">
                <n-button size="small" :type="paper.starred ? 'warning' : 'default'" @click="handleStar">
                  {{ paper.starred ? '已收藏' : '收藏' }}
                </n-button>
                <n-button size="small" :type="paper.pinned ? 'info' : 'default'" @click="handlePin">
                  {{ paper.pinned ? '已置顶' : '置顶' }}
                </n-button>
                <n-button tag="a" :href="paper.url" target="_blank" size="small" v-if="paper.url">arXiv</n-button>
                <n-button tag="a" :href="paper.pdf_url" target="_blank" size="small" v-if="paper.pdf_url">PDF</n-button>
                <n-button tag="a" :href="paper.github_url" target="_blank" size="small" v-if="paper.github_url" type="success">Code</n-button>
                <n-button @click="router.back()" size="small">返回</n-button>
              </div>
            </div>
          </template>

          <div class="detail-body">
            <!-- 缩略图 -->
            <div v-if="paper.arxiv_id" class="detail-thumb">
              <img :src="thumbnailUrl(paper.arxiv_id)" :alt="paper.title" @error="onThumbError" />
            </div>

            <!-- 摘要 -->
            <div class="detail-content">
              <h3>摘要</h3>
              <p v-if="paper.abstract_zh" class="abstract-zh">{{ paper.abstract_zh }}</p>
              <p class="abstract-en">{{ paper.abstract }}</p>

              <!-- AI 概述 -->
              <div v-if="paper.summary_zh" class="summary-section">
                <n-divider />
                <h3>AI 概述</h3>
                <p class="summary-text">{{ paper.summary_zh }}</p>
              </div>
            </div>
          </div>
        </n-card>

        <!-- 笔记区 -->
        <n-card title="精读笔记" size="small" style="margin-top: 16px">
          <template #header-extra>
            <n-space>
              <n-button v-if="!editingNote" size="small" @click="startEditNote">编辑笔记</n-button>
              <template v-else>
                <n-button size="small" @click="cancelEditNote">取消</n-button>
                <n-button size="small" type="primary" :loading="savingNote" @click="saveNote">保存</n-button>
              </template>
            </n-space>
          </template>
          <MarkdownRenderer v-if="!editingNote && noteContent" :content="noteContent" />
          <n-input v-else-if="editingNote" v-model:value="noteDraft" type="textarea" :rows="12" placeholder="支持 Markdown 格式..." />
          <EmptyState v-else description="暂无笔记，点击「编辑笔记」开始记录" />
        </n-card>
      </div>
      <EmptyState v-else description="未找到该论文" />
    </n-spin>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NCard, NSpin, NButton, NTag, NSpace, NDivider, NInput, useMessage } from 'naive-ui'
import MarkdownRenderer from '../../components/common/MarkdownRenderer.vue'
import EmptyState from '../../components/common/EmptyState.vue'
import {
  getPaperDetail, getPaperNote, savePaperNote,
  getThumbnailUrl, starPaper, pinPaper,
} from '../../api/papers'

const route = useRoute()
const router = useRouter()
const message = useMessage()

const loading = ref(false)
const paper = ref(null)
const noteContent = ref('')
const editingNote = ref(false)
const noteDraft = ref('')
const savingNote = ref(false)

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

function thumbnailUrl(arxivId) { return getThumbnailUrl(arxivId) }
function onThumbError(e) { e.target.style.display = 'none' }
function formatAuthors(authors) {
  if (!authors || !authors.length) return ''
  const list = Array.isArray(authors) ? authors : [authors]
  return list.join(', ')
}

function startEditNote() { noteDraft.value = noteContent.value; editingNote.value = true }
function cancelEditNote() { editingNote.value = false }

async function handleStar() {
  const newVal = !paper.value.starred
  paper.value.starred = newVal
  try {
    starPaper(route.params.id, newVal)
    message.success(newVal ? '已收藏' : '已取消收藏')
  } catch {
    paper.value.starred = !newVal
    message.error('操作失败')
  }
}

async function handlePin() {
  const newVal = !paper.value.pinned
  paper.value.pinned = newVal
  try {
    pinPaper(route.params.id, newVal)
    message.success(newVal ? '已置顶' : '已取消置顶')
  } catch {
    paper.value.pinned = !newVal
    message.error('操作失败')
  }
}

async function saveNote() {
  savingNote.value = true
  try {
    savePaperNote(route.params.id, noteDraft.value)
    noteContent.value = noteDraft.value
    editingNote.value = false
    message.success('笔记已保存')
  } catch { message.error('保存失败') }
  savingNote.value = false
}

onMounted(async () => {
  loading.value = true
  try {
    const p = getPaperDetail(route.params.id)
    const n = getPaperNote(route.params.id)
    paper.value = p
    noteContent.value = n?.content || ''
  } catch {}
  loading.value = false
})
</script>

<style scoped lang="scss">
.detail-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.header-info { flex: 1; min-width: 0; h2 { margin-bottom: 4px; } }
.title-en { font-size: 13px; color: #94a3b8; font-style: italic; margin-bottom: 4px; }
.authors { font-size: 13px; color: #6b7280; margin-bottom: 8px; }
.meta-tags { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; .date { font-size: 12px; color: #94a3b8; margin-left: 6px; } }
.tags-list { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 6px; }
.detail-tag {
  font-size: 11px; padding: 2px 8px; border-radius: 999px;
  background: #f1f5f9; color: #64748b;
}
.header-actions { display: flex; gap: 6px; flex-shrink: 0; flex-wrap: wrap; }

.detail-body { display: flex; gap: 20px; }
.detail-thumb { width: 200px; flex-shrink: 0; img { width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; } }
.detail-content { flex: 1; min-width: 0; h3 { font-size: 15px; margin-bottom: 8px; } }
.abstract-zh { font-size: 14px; color: #1e293b; line-height: 1.7; margin-bottom: 12px; }
.abstract-en { font-size: 13px; color: #6b7280; line-height: 1.6; }
.summary-section { margin-top: 16px; }
.summary-text { font-size: 14px; color: #334155; line-height: 1.7; background: #f8fafc; padding: 12px; border-radius: 8px; }
</style>
