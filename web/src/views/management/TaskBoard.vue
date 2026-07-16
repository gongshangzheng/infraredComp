<template>
  <div class="page-container">
    <n-spin :show="loading">
      <!-- 项目切换：看板从选中项目的 tasks.json 派生 -->
      <div class="project-bar">
        <span class="lbl">项目</span>
        <n-select
          v-model:value="currentSlug"
          :options="projectOptions"
          size="small"
          style="width: 220px"
          placeholder="选择项目"
          @update:value="onProjectChange"
        />
      </div>

      <n-grid :cols="3" :x-gap="16" :y-gap="16" style="margin-top: 12px">
        <n-gi>
          <n-card title="待开始" size="small">
            <template #header-extra><n-tag size="small" round>{{ tasks.pending.length }}</n-tag></template>
            <div class="task-list">
              <div v-for="t in tasks.pending" :key="t.id" class="task-card" :class="t.priority ? `pri-${t.priority.toLowerCase()}` : ''">
                <div class="task-name">
                  <span v-if="t.priority" class="priority-badge" :class="`priority-${t.priority.toLowerCase()}`">{{ t.priority }}</span>
                  {{ t.name }}
                </div>
                <div class="task-meta">
                  <span>{{ t.owner }}</span>
                  <span v-if="t.end_date">{{ t.end_date }}</span>
                </div>
              </div>
              <p v-if="!tasks.pending.length" class="text-light">暂无</p>
            </div>
          </n-card>
        </n-gi>
        <n-gi>
          <n-card title="进行中" size="small">
            <template #header-extra><n-tag size="small" type="info" round>{{ tasks.in_progress.length }}</n-tag></template>
            <div class="task-list">
              <div v-for="t in tasks.in_progress" :key="t.id" class="task-card" :class="t.priority ? `pri-${t.priority.toLowerCase()}` : ''">
                <div class="task-name">
                  <span v-if="t.priority" class="priority-badge" :class="`priority-${t.priority.toLowerCase()}`">{{ t.priority }}</span>
                  {{ t.name }}
                </div>
                <div class="task-meta">
                  <StatusBadge :status="t.status" />
                  <span>{{ t.owner }}</span>
                </div>
                <div v-if="t.note" class="task-note">{{ t.note }}</div>
              </div>
              <p v-if="!tasks.in_progress.length" class="text-light">暂无</p>
            </div>
          </n-card>
        </n-gi>
        <n-gi>
          <n-card title="已完成" size="small">
            <template #header-extra><n-tag size="small" type="success" round>{{ tasks.completed.length }}</n-tag></template>
            <div class="task-list">
              <div v-for="t in tasks.completed" :key="t.id" class="task-card" :class="t.priority ? `pri-${t.priority.toLowerCase()}` : ''">
                <div class="task-name">
                  <span v-if="t.priority" class="priority-badge" :class="`priority-${t.priority.toLowerCase()}`">{{ t.priority }}</span>
                  {{ t.name }}
                </div>
                <div class="task-meta">
                  <span>{{ t.owner }}</span>
                  <span v-if="t.end_date">{{ t.end_date }}</span>
                </div>
              </div>
              <p v-if="!tasks.completed.length" class="text-light">暂无</p>
            </div>
          </n-card>
        </n-gi>
      </n-grid>
    </n-spin>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { NGrid, NGi, NCard, NTag, NSpin, NSelect } from 'naive-ui'
import StatusBadge from '../../components/common/StatusBadge.vue'
import { getProjects, getTasks } from '../../api/management'

const loading = ref(false)
const projects = ref([])
const currentSlug = ref(null)
const tasks = ref({ pending: [], in_progress: [], completed: [] })

const projectOptions = computed(() => projects.value.map(p => ({ label: p.title || p.slug, value: p.slug })))

async function loadTasks(slug) {
  if (!slug) { tasks.value = { pending: [], in_progress: [], completed: [] }; return }
  try {
    tasks.value = await getTasks(slug)
  } catch {
    tasks.value = { pending: [], in_progress: [], completed: [] }
  }
}

function onProjectChange(slug) {
  currentSlug.value = slug
  loadTasks(slug)
}

onMounted(async () => {
  loading.value = true
  try {
    projects.value = await getProjects()
    if (projects.value.length) {
      currentSlug.value = projects.value[0].slug
      await loadTasks(currentSlug.value)
    }
  } catch {}
  loading.value = false
})
</script>

<style scoped lang="scss">
.project-bar { display: flex; align-items: center; gap: 8px; }
.lbl { font-size: 13px; color: var(--color-text-secondary); }
.task-list { display: flex; flex-direction: column; gap: 8px; }
.task-card {
  padding: 12px;
  background: #f9fafb;
  border-radius: 8px;
  border-left: 3px solid #e5e7eb;
  &.pri-p1 { border-left-color: #ef4444; }
  &.pri-p2 { border-left-color: #f59e0b; }
  &.pri-p3 { border-left-color: #a1a1aa; }

  .task-name { font-weight: 500; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }
  .task-meta { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #6b7280; }
  .task-note { font-size: 12px; color: #9ca3af; margin-top: 4px; }
}
.priority-badge {
  flex-shrink: 0; font-size: 9px; font-weight: 700; letter-spacing: 0.03em;
  padding: 1px 5px; border-radius: 3px;
  &.priority-p1 { background: rgba(239,68,68,0.15); color: #ef4444; }
  &.priority-p2 { background: rgba(245,158,11,0.15); color: #f59e0b; }
  &.priority-p3 { background: rgba(161,161,170,0.15); color: #71717a; }
}
</style>
