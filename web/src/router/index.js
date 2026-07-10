import { createRouter, createWebHistory } from 'vue-router'
import MainLayout from '../layouts/MainLayout.vue'

const routes = [
  {
    path: '/',
    component: MainLayout,
    children: [
      {
        path: '',
        name: 'Home',
        component: () => import('../views/Home.vue'),
        meta: { title: '首页' },
      },
      {
        path: 'papers',
        redirect: '/papers/list',
      },
      {
        path: 'papers/list',
        name: 'PaperList',
        component: () => import('../views/papers/PaperList.vue'),
        meta: { title: '论文列表', module: 'papers' },
      },
      {
        path: 'papers/:id',
        name: 'PaperDetail',
        component: () => import('../views/papers/PaperDetail.vue'),
        meta: { title: '论文详情', module: 'papers' },
      },
    ],
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, from, next) => {
  document.title = to.meta.title
    ? `${to.meta.title} - 红外图像压缩论文库`
    : '红外图像压缩论文库'
  next()
})

export default router
