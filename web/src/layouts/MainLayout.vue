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
            <n-breadcrumb-item v-for="item in breadcrumbs" :key="item.path">
              {{ item.title }}
            </n-breadcrumb-item>
          </n-breadcrumb>
        </div>
        <div class="header-right">
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
  NMenu, NBreadcrumb, NBreadcrumbItem, NIcon,
} from 'naive-ui'
import {
  HomeOutline, DocumentTextOutline, LibraryOutline,
} from '@vicons/ionicons5'

const route = useRoute()
const router = useRouter()
const collapsed = ref(false)

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
    label: '论文库',
    key: 'papers',
    icon: renderIcon(LibraryOutline),
    children: [
      { label: '论文列表', key: '/papers/list', icon: renderIcon(DocumentTextOutline) },
    ],
  },
]

const activeKey = computed(() => {
  const path = route.path
  const allKeys = ['/', '/papers/list']
  let best = '/'
  for (const k of allKeys) {
    if (path.startsWith(k) && k.length > best.length) best = k
  }
  return best
})

function handleMenuSelect(key) {
  router.push(key)
}

const breadcrumbs = computed(() => {
  const items = [{ title: '首页', path: '/' }]
  const moduleMap = {
    papers: '论文库',
  }
  if (route.meta.module && moduleMap[route.meta.module]) {
    items.push({ title: moduleMap[route.meta.module], path: '' })
  }
  if (route.meta.title && route.meta.title !== '首页') {
    items.push({ title: route.meta.title, path: route.path })
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
  border-bottom: 1px solid #2a2a3e;
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
  background: #fff;
}

.header-date {
  font-size: 13px;
  color: #9ca3af;
}

.app-content {
  height: calc(100vh - 56px);
  padding: 0;
}
</style>
