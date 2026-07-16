<template>
  <div class="page-container">
    <n-card title="评测配置与运行" size="small">
      <n-spin :show="loading">
        <n-form ref="formRef" :model="form" label-placement="left" :label-width="100" style="max-width: 600px">
          <n-form-item label="评测模式">
            <n-radio-group v-model:value="form.mode">
              <n-radio value="speed">speed run(少量视频,看主观)</n-radio>
              <n-radio value="formal">formal test(全量,看平均指标)</n-radio>
            </n-radio-group>
          </n-form-item>
          <n-form-item label="codec" required>
            <n-select v-model:value="form.codecs" :options="codecOptions" multiple
              placeholder="选择 codec（传统 + 学习式；空=默认传统 4 个）" />
            <span class="hint">学习式 codec（ssf2020/img-*/dcvc_rt）用各自 quality 级；CRF 仅对传统 codec 生效。</span>
          </n-form-item>
          <n-form-item label="提取方法" required>
            <n-select v-model:value="form.method" :options="methodOptions" placeholder="选择轮廓提取方法 (canny/sobel)" />
          </n-form-item>
          <n-form-item label="训练 Checkpoint">
            <n-select v-model:value="form.checkpoint" :options="checkpointOptions" clearable placeholder="选 run checkpoint → 自动填 codec/quality（可选）" />
            <span class="hint">选择后自动填入 codec 和 quality，也可手动选 codec</span>
          </n-form-item>
          <n-form-item label="CRFs">
            <n-select v-model:value="form.crfs" :options="crfOptions" multiple placeholder="默认 18,23,28,33（仅传统 codec）" />
          </n-form-item>
          <n-form-item v-if="form.mode === 'speed'" label="序列子集">
            <n-input v-model:value="form.sequences" placeholder="逗号分隔 stem,如 akiyo_cif,bus_cif(空=全量)" />
          </n-form-item>
          <n-form-item label="选择数据集" required>
            <n-select v-model:value="form.dataset_id" :options="datasetOptions" placeholder="请选择数据集" />
          </n-form-item>
          <n-form-item label="评测配置">
            <n-select v-model:value="form.config_id" :options="configOptions" placeholder="选择配置（可选）" clearable />
          </n-form-item>
          <n-form-item label=" ">
            <n-button type="primary" :loading="running" @click="handleRun">启动评测</n-button>
          </n-form-item>
        </n-form>
      </n-spin>
    </n-card>

    <n-card v-if="runResult" title="运行结果" size="small" style="margin-top: 16px">
      <n-alert :type="runResult.success ? 'success' : 'error'" :title="runResult.success ? '评测已启动' : '评测失败'">
        <pre>{{ runResult.output }}</pre>
      </n-alert>
    </n-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { NCard, NSpin, NForm, NFormItem, NSelect, NButton, NAlert, useMessage } from 'naive-ui'
import { getCodecs, getDatasets, getEvalConfigs, getMethods, runEvaluation } from '../../api/evaluation'
import { listCheckpoints, getTrainRuns } from '../../api/training'

const message = useMessage()
const router = useRouter()
const loading = ref(false)
const running = ref(false)
const codecs = ref([])
const datasets = ref([])
const configs = ref([])
const methods = ref([])
const checkpoints = ref([])
const trainRuns = ref([])
const runResult = ref(null)
const form = ref({ mode: 'speed', method: 'canny', codecs: [], crfs: [], sequences: '', dataset_id: null, config_id: null, checkpoint: null })

const LEARNED_KEYWORDS = ['ssf', 'elic', 'img-', 'dcvc', 'nevc']
const isLearnedCodec = computed(() =>
  (form.value.codecs || []).some(id => LEARNED_KEYWORDS.some(kw => id.includes(kw)))
)

const codecOptions = computed(() => codecs.value.map(c => ({ label: `${c.id}（${c.name} · ${c.kind}）`, value: c.id })))
const methodOptions = computed(() => methods.value.map(m => ({ label: m, value: m })))
const datasetOptions = computed(() => datasets.value.map(d => ({ label: `${d.name}${(d.sequences || []).some(s => s.missing) ? ' (文件缺失)' : ''}`, value: d.id })))
const configOptions = computed(() => configs.value.map(c => ({ label: c.name || c.id, value: c.id })))
const crfOptions = [18, 23, 28, 33].map(c => ({ label: 'crf' + c, value: c }))
const checkpointOptions = computed(() => {
  const opts = []
  for (const run of trainRuns.value) {
    const mid = run.model_id || run.model || ''
    const q = run.quality ?? ''
    if (run.has_best && run.best) {
      opts.push({
        label: `${mid} q${q} · best (PSNR=${run.best.test?.psnr?.toFixed(2) ?? '?'})`,
        value: { path: run.best.path, model_id: mid, quality: q }
      })
    }
    if (run.has_latest && run.latest) {
      opts.push({
        label: `${mid} q${q} · latest (epoch=${run.latest.epoch ?? '?'})`,
        value: { path: run.latest.path, model_id: mid, quality: q }
      })
    }
  }
  return opts
})

watch(() => form.value.checkpoint, (ckpt) => {
  if (!ckpt || typeof ckpt !== 'object') return
  const mid = (ckpt.model_id || '').toLowerCase()
  const codecId = mid === 'ssf2020' ? 'ssf2020' : `img-${mid}`
  const exists = codecs.value.some(c => c.id === codecId)
  if (exists) form.value.codecs = [codecId]
  form.value.crfs = []
})

async function handleRun() {
  if (!form.value.codecs.length) {
    message.warning('请选择至少一个 codec')
    return
  }
  if (!form.value.dataset_id) {
    message.warning('请选择数据集')
    return
  }
  if (!form.value.method) {
    message.warning('请选择轮廓提取方法')
    return
  }
  running.value = true
  runResult.value = null
  try {
    const payload = { ...form.value, checkpoint: form.value.checkpoint?.path ?? form.value.checkpoint ?? null }
    const res = await runEvaluation(payload)
    runResult.value = {
      success: true,
      output: JSON.stringify(res, null, 2),
    }
    message.success('评测已启动,跳转结果页')
    router.push(form.value.mode === 'speed' ? '/evaluation/speed' : '/evaluation/formal')
  } catch (e) {
    runResult.value = { success: false, output: e.message || '评测执行失败' }
    message.error('评测执行失败')
  }
  running.value = false
}

onMounted(async () => {
  loading.value = true
  try {
    const [c, d, cfg, meth, ckpts, runs] = await Promise.all([
      getCodecs().catch(() => []),
      getDatasets().catch(() => []),
      getEvalConfigs().catch(() => []),
      getMethods().catch(() => ({ methods: [] })),
      listCheckpoints().catch(() => []),
      getTrainRuns().catch(() => ({ runs: [] })),
    ])
    codecs.value = c || []
    datasets.value = d || []
    configs.value = cfg || []
    methods.value = meth?.methods || []
    checkpoints.value = Array.isArray(ckpts) ? ckpts : (ckpts?.checkpoints || [])
    trainRuns.value = runs?.runs || []
    if (!form.value.method && methods.value.length) form.value.method = methods.value[0]
  } catch {}
  loading.value = false
})
</script>

<style scoped lang="scss">
.hint { display: block; font-size: 11px; color: var(--color-text-dim); margin-top: 4px; }
</style>

