<template>
  <div class="viz-epoch-card">
    <div class="vec-head">
      <span class="vec-epoch">epoch {{ epoch }}</span>
      <span v-if="samples.length > 1" class="vec-nav">
        <n-button size="tiny" quaternary :disabled="cur <= 0" @click="cur = Math.max(0, cur - 1)">‹</n-button>
        <span class="vec-count">{{ cur + 1 }}/{{ samples.length }}</span>
        <n-button size="tiny" quaternary :disabled="cur >= samples.length - 1" @click="cur = Math.min(samples.length - 1, cur + 1)">›</n-button>
      </span>
    </div>
    <img
      v-if="curItem"
      :src="getTrainOutputUrl(curItem.path)"
      loading="lazy"
      class="vec-img"
      @click="$emit('preview', getTrainOutputUrl(curItem.path), `epoch ${epoch} · sample ${(curItem.sample ?? cur) + 1}`)"
    />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { NButton } from 'naive-ui'
import { getTrainOutputUrl } from '../../api/training'

const props = defineProps({
  epoch: { type: [Number, String], required: true },
  samples: { type: Array, default: () => [] },  // [{sample, path}] 或 [{path}]（旧格式）
})
defineEmits(['preview'])

const cur = ref(0)
const curItem = computed(() => {
  if (!props.samples.length) return null
  return props.samples[Math.min(cur.value, props.samples.length - 1)] || null
})
</script>

<style scoped lang="scss">
.viz-epoch-card {
  display: flex; flex-direction: column; gap: 4px;
  padding: 8px; border: 1px solid var(--color-border-light); border-radius: 8px; background: var(--color-elevated);
}
.vec-head { display: flex; justify-content: space-between; align-items: center; }
.vec-epoch { font-size: 11px; color: var(--color-text-dim); }
.vec-nav { display: flex; align-items: center; gap: 2px; }
.vec-count { font-size: 11px; color: var(--color-text-dim); min-width: 28px; text-align: center; }
.vec-img {
  width: 100%; height: auto; display: block; border-radius: 6px; cursor: pointer; background: #000;
  /* 不降分辨率：按原图宽高比、宽度撑满 cell；高度由内容决定 */
}
</style>
