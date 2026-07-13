<template>
  <div class="page-container">
    <n-card title="评测配置与运行" size="small">
      <n-spin :show="loading">
        <n-form ref="formRef" :model="form" label-placement="left" :label-width="100" style="max-width: 600px">
          <n-form-item label="选择模型" required>
            <n-select v-model:value="form.model_id" :options="modelOptions" placeholder="请选择模型" />
          </n-form-item>
          <n-form-item v-if="selectedModel?.kind === 'dl' || selectedModel?.kind === 'learned-video'" label="checkpoint">
            <n-select v-model:value="form.checkpoint_id" :options="checkpointOptions" :placeholder="checkpointOptions.length ? '选择 trained checkpoint（不选=pretrained）' : '暂无 trained checkpoint（将用 pretrained）'" clearable :disabled="!selectedModel?.trainable" />
            <span class="hint">{{ selectedModel?.trainable === false ? '该模型推理专用（无训练），用 pretrained；如 ssf2020 可训练则可选 trained。' : '选 trained checkpoint → 用训练产出评测；不选=pretrained。' }}</span>
          </n-form-item>
          <n-form-item v-if="selectedModel?.kind === 'dl' || selectedModel?.kind === 'learned-video'" label="quality" required>
            <n-select v-model:value="form.quality" :options="qualityOptions" placeholder="quality 级" />
          </n-form-item>
          <n-form-item label="提取方法" required>
            <n-select v-model:value="form.method" :options="methodOptions" placeholder="选择轮廓提取方法 (canny/sobel)" />
          </n-form-item>
          <n-form-item label="选择数据集" required>
            <n-select v-model:value="form.dataset_id" :options="datasetOptions" placeholder="请选择数据集" />
          </n-form-item>
          <n-form-item label="评测配置">
            <n-select v-model:value="form.config_id" :options="configOptions" placeholder="选择配置（可选）" clearable />
          </n-form-item>
          <n-form-item label="Batch Size">
            <n-input-number v-model:value="form.batch_size" :min="1" :max="256" />
          </n-form-item>
          <n-form-item label="设备">
            <n-radio-group v-model:value="form.device">
              <n-radio value="cuda">GPU (CUDA)</n-radio>
              <n-radio value="cpu">CPU</n-radio>
            </n-radio-group>
          </n-form-item>
          <n-form-item label="额外参数">
            <n-input v-model:value="form.extra_args" type="textarea" :rows="2" placeholder="--clip_len 16 --num_workers 4" />
          </n-form-item>
          <n-form-item label=" ">
            <n-button type="primary" :loading="running" @click="handleRun">启动评测</n-button>
          </n-form-item>
        </n-form>
      </n-spin>
    </n-card>

    <n-card v-if="runResult" title="运行结果" size="small" style="margin-top: 16px">
      <n-alert :type="runResult.success ? 'success' : 'error'" :title="runResult.success ? '评测完成' : '评测失败'">
        <pre>{{ runResult.output }}</pre>
      </n-alert>
      <!-- 运行后若返回 output_video，直接按需看输出视频 + 指标 -->
      <div v-if="runResult.videoSrc" style="margin-top: 12px">
        <h4 style="margin-bottom: 8px">输出视频</h4>
        <video :src="runResult.videoSrc" controls preload="none" playsinline style="width: 100%; max-height: 360px; background: #000" />
        <n-descriptions v-if="runResult.metrics" :column="3" size="small" label-placement="left" bordered style="margin-top: 12px">
          <n-descriptions-item v-for="(v, k) in runResult.metrics" :key="k" :label="k">
            {{ typeof v === 'number' ? (v.toFixed?.(3) ?? v) : v }}
          </n-descriptions-item>
        </n-descriptions>
      </div>
    </n-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { NCard, NSpin, NForm, NFormItem, NSelect, NInputNumber, NRadioGroup, NRadio, NInput, NButton, NAlert, NDescriptions, NDescriptionsItem, useMessage } from 'naive-ui'
import { getModels, getDatasets, getEvalConfigs, getMethods, runEvaluation, getOutputUrl } from '../../api/evaluation'

const message = useMessage()
const loading = ref(false)
const running = ref(false)
const models = ref([])
const datasets = ref([])
const configs = ref([])
const methods = ref([])
const runResult = ref(null)
const form = ref({ model_id: null, method: 'canny', dataset_id: null, config_id: null, batch_size: 8, device: 'cuda', extra_args: '', checkpoint_id: null, quality: null })

const selectedModel = computed(() => models.value.find(m => m.id === form.value.model_id) || null)
const modelOptions = computed(() => models.value.map(m => ({ label: `${m.name} (${m.type}${m.kind === 'dl' ? ' · DL' : m.kind === 'learned-video' ? ' · 学习视频' : ''})`, value: m.id })))
const checkpointOptions = computed(() => (selectedModel.value?.checkpoint || []).map(p => ({ label: p, value: p })))
const qualityOptions = computed(() => (selectedModel.value?.qualities || []).map(q => ({ label: `q${q}`, value: q })))
const methodOptions = computed(() => methods.value.map(m => ({ label: m, value: m })))
const datasetOptions = computed(() => datasets.value.map(d => ({ label: d.name, value: d.id })))
const configOptions = computed(() => configs.value.map(c => ({ label: c.name || c.id, value: c.id })))

async function handleRun() {
  if (!form.value.model_id || !form.value.dataset_id) {
    message.warning('请选择模型和数据集')
    return
  }
  if (!form.value.method) {
    message.warning('请选择轮廓提取方法')
    return
  }
  running.value = true
  runResult.value = null
  try {
    const res = await runEvaluation(form.value)
    // 若后端返回 output_video，按需拼 URL（<video preload=none> 仅渲染时才请求字节）
    const vid = res?.output_video ? getOutputUrl(res.output_video) : ''
    runResult.value = {
      success: true,
      output: JSON.stringify(res, null, 2),
      videoSrc: vid,
      metrics: res?.metrics || null,
    }
    message.success('评测已完成')
  } catch (e) {
    runResult.value = { success: false, output: e.message || '评测执行失败' }
    message.error('评测执行失败')
  }
  running.value = false
}

onMounted(async () => {
  loading.value = true
  try {
    const [m, d, c, meth] = await Promise.all([
      getModels().catch(() => []),
      getDatasets().catch(() => []),
      getEvalConfigs().catch(() => []),
      getMethods().catch(() => ({ methods: [] })),
    ])
    models.value = m || []
    datasets.value = d || []
    configs.value = c || []
    methods.value = meth?.methods || []
    if (!form.value.method && methods.value.length) form.value.method = methods.value[0]
  } catch {}
  loading.value = false
})
</script>
