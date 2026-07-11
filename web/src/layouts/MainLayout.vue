<template>
  <n-layout has-sider style="height: 100vh">
    <!-- 侧边栏 -->
    <n-layout-sider
      bordered
      collapse-mode="width"
      :collapsed-width="64"
      :width="240"
      :collapsed="collapsed"
      show-trigger
      @collapse="collapsed = true"
      @expand="collapsed = false"
    >
      <div class="sidebar-logo">
        <span v-if="!collapsed" class="logo-text">红外图像压缩</span>
        <span v-else class="logo-icon">IR</span>
      </div>
      <n-menu
        :collapsed="collapsed"
        :collapsed-width="64"
        :collapsed-icon-size="22"
        :options="menuOptions"
        :value="activeKey"
        @update:value="handleMenuSelect"
      />
    </n-layout-sider>

    <!-- 主内容区 -->
    <n-layout>
      <!-- 顶部 -->
      <n-layout-header bordered class="app-header">
        <div class="header-left">
          <n-breadcrumb>
            <n-breadcrumb-item
              v-for="item in breadcrumbs"
              :key="item.path || item.title"
              :to="item.path || undefined"
              :clickable="!!item.path"
              @click="onBreadcrumbClick(item)"
            >
              {{ item.title }}
            </n-breadcrumb-item>
          </n-breadcrumb>
        </div>
        <div class="header-right">
          <n-button quaternary circle class="theme-toggle" @click="themeStore.toggle">
            <template #icon>
              <n-icon size="18">
                <sunny-outline v-if="themeStore.isDark" />
                <moon-outline v-else />
              </n-icon>
            </template>
          </n-button>
          <span class="header-date">{{ today }}</span>
        </div>
      </n-layout-header>

      <!-- 内容区 -->
      <n-layout-content class="app-content" :native-scrollbar="false">
        <router-view />
      </n-layout-content>
    </n-layout>
  </n-layout>
</template>

<script setup>
import { ref, computed, h } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  NLayout, NLayoutSider, NLayoutHeader, NLayoutContent,
  NMenu, NBreadcrumb, NBreadcrumbItem, NButton, NIcon,
} from 'naive-ui'
import {
  HomeOutline, DocumentTextOutline, LibraryOutline,
  PeopleOutline, GridOutline, CalendarOutline,
  CheckboxOutline, FlagOutline, ChatbubblesOutline, BarChartOutline,
  FlaskOutline, FilmOutline, CubeOutline, LayersOutline, SettingsOutline,
  SunnyOutline, MoonOutline,
} from '@vicons/ionicons5'
import { useThemeStore } from '../stores/theme'

const route = useRoute()
const router = useRouter()
const collapsed = ref(false)
const themeStore = useThemeStore()

function renderIcon(icon) {
  return () => h(NIcon, null, { default: () => h(icon) })
}

const menuOptions = [
  {
    label: '首页',
    key: '/',
    icon: renderIcon(HomeOutline),
  },
  {
    label: '项目管理',
    key: 'management',
    icon: renderIcon(GridOutline),
    children: [
      { label: '项目树', key: '/management/projects', icon: renderIcon(GridOutline) },
      { label: '团队成员', key: '/management/team', icon: renderIcon(PeopleOutline) },
      { label: '日报', key: '/management/daily', icon: renderIcon(CalendarOutline) },
      { label: '周报', key: '/management/weekly', icon: renderIcon(CalendarOutline) },
      { label: '月报', key: '/management/monthly', icon: renderIcon(CalendarOutline) },
      { label: '任务看板', key: '/management/tasks', icon: renderIcon(CheckboxOutline) },
      { label: '里程碑', key: '/management/milestones', icon: renderIcon(FlagOutline) },
      { label: '会议纪要', key: '/management/meetings', icon: renderIcon(ChatbubblesOutline) },
    ],
  },
  {
    label: '论文库',
    key: 'papers',
    icon: renderIcon(LibraryOutline),
    children: [
      { label: '论文列表', key: '/papers/list', icon: renderIcon(DocumentTextOutline) },
    ],
  },
  {
    label: '评测体系',
    key: 'evaluation',
    icon: renderIcon(FlaskOutline),
    children: [
      { label: '评测运行', key: '/evaluation/run', icon: renderIcon(FlaskOutline) },
      { label: '评测结果', key: '/evaluation/results', icon: renderIcon(BarChartOutline) },
      { label: '方法对比', key: '/evaluation/compare', icon: renderIcon(LayersOutline) },
      { label: '查看输出', key: '/evaluation/outputs', icon: renderIcon(FilmOutline) },
      { label: '模型(codec)配置', key: '/evaluation/models', icon: renderIcon(CubeOutline) },
      { label: '数据集配置', key: '/evaluation/datasets', icon: renderIcon(LayersOutline) },
      { label: '评测配置', key: '/evaluation/configs', icon: renderIcon(SettingsOutline) },
    ],
  },
]

const activeKey = computed(() => {
  const path = route.path
  const allKeys = [
    '/', '/papers/list',
    '/management/projects', '/management/team', '/management/daily',
    '/management/weekly', '/management/monthly', '/management/tasks',
    '/management/milestones', '/management/meetings',
    '/evaluation/run', '/evaluation/results', '/evaluation/compare', '/evaluation/outputs', '/evaluation/models', '/evaluation/datasets', '/evaluation/configs',
  ]
  let best = '/'
  for (const k of allKeys) {
    if (path.startsWith(k) && k.length > best.length) best = k
  }
  return best
})

function handleMenuSelect(key) {
  router.push(key)
}

function onBreadcrumbClick(item) {
  if (item.path && item.path !== route.path) router.push(item.path)
}

// 模块 -> 模块首页路径（让 breadcrumb 的模块项可点回退）
const MODULE_PATH = {
  management: '/management/projects',
  papers: '/papers/list',
  evaluation: '/evaluation/results',
}
const MODULE_LABEL = {
  papers: '论文库',
  management: '项目管理',
  evaluation: '评测体系',
}

const breadcrumbs = computed(() => {
  const items = [{ title: '首页', path: '/' }]
  if (route.meta.module && MODULE_LABEL[route.meta.module]) {
    items.push({ title: MODULE_LABEL[route.meta.module], path: MODULE_PATH[route.meta.module] || '' })
  }
  // 当前页放最后、不可点击（已在当前页）
  if (route.meta.title && route.meta.title !== '首页') {
    items.push({ title: route.meta.title, path: '' })
  }
  return items
})

const today = computed(() => {
  const d = new Date()
  const weekdays = ['日', '一', '二', '三', '四', '五', '六']
  return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日 星期${weekdays[d.getDay()]}`
})
</script>

<style scoped lang="scss">
.sidebar-logo {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-bottom: 1px solid var(--color-border);
  color: #fff;
  font-weight: 700;

  .logo-text {
    font-size: 16px;
    white-space: nowrap;
  }
  .logo-icon {
    font-size: 14px;
  }
}

.app-header {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: var(--color-card);
  border-bottom: 1px solid var(--color-border);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.theme-toggle {
  color: var(--color-text-secondary);
}

.header-date {
  font-size: 13px;
  color: var(--color-text-dim);
}

.app-content {
  height: calc(100vh - 56px);
  padding: 0;
  background: var(--color-bg);
}
</style>
