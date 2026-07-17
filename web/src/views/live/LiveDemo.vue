<template>
  <div class="live-demo">
    <div class="toolbar">
      <span class="title">实时红外画面</span>
      <n-button-group size="small">
        <n-button
          v-for="p in palettes"
          :key="p.key"
          :type="palette === p.key ? 'primary' : 'default'"
          @click="palette = p.key"
        >{{ p.label }}</n-button>
      </n-button-group>
      <span class="status" :class="{ ok: live, bad: !live }">{{ live ? '● 在线' : '○ 离线' }}</span>
      <span class="hint">mode={{ palette }}</span>
    </div>
    <div class="frame">
      <img v-if="live" :src="streamUrl" :key="palette" alt="IR" crossorigin="anonymous" />
      <div v-else class="offline">MJPEG 服务离线，请运行 ir_mjpeg_server.py 或检查相机 USB</div>
    </div>
  </div>
</template>
<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { NButton, NButtonGroup } from 'naive-ui'
const palettes = [
  { key: 'ironbow', label: '描红' },
  { key: 'whitehot', label: '白热' },
  { key: 'blackhot', label: '黑热' },
  { key: 'rainbow', label: '彩虹' },
  { key: 'canny', label: 'Canny' },
  { key: 'sobel', label: 'Sobel' },
]
const palette = ref('ironbow')
const BASE = 'http://127.0.0.1:8080'
const streamUrl = computed(() => BASE + '/stream?palette=' + palette.value)
const live = ref(false)
let t
function check() {
  fetch(BASE + '/', { mode: 'no-cors' }).then(() => (live.value = true)).catch(() => (live.value = false))
}
onMounted(() => { check(); t = setInterval(check, 3000) })
onUnmounted(() => clearInterval(t))
</script>
<style scoped>
.live-demo { padding: 16px; height: 100%; display: flex; flex-direction: column; gap: 12px; }
.toolbar { display: flex; align-items: center; gap: 12px; }
.toolbar .title { font-weight: 600; }
.status.ok { color: #18a058; }
.status.bad { color: #d03050; }
.frame { flex: 1; background: #000; border-radius: 8px; display: flex; align-items: center; justify-content: center; }
.frame img { max-width: 100%; max-height: 100%; object-fit: contain; }
.offline { color: #ff8888; font-size: 14px; }
</style>
