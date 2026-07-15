import request from './request'

// codec 列表（传统 + 学习式，统一来自 benchmark 注册表；模型即 codec）
export const getCodecs = () => request.get('/evaluation/codecs')
export const getCodecDetail = (id) => request.get(`/evaluation/codecs/${id}`)
// 兼容旧名（部分页面可能仍引用）
export const getModels = getCodecs
export const getModelDetail = getCodecDetail

// 轮廓提取方法（canny/sobel/...）
export const getMethods = () => request.get('/evaluation/methods')

// 数据集
export const getDatasets = () => request.get('/evaluation/datasets')
export const getDatasetDetail = (id) => request.get(`/evaluation/datasets/${id}`)
export const getDatasetPreview = (id, params) => request.get(`/evaluation/datasets/${id}/preview`, { params })
export const downloadDataset = (id) => request.post(`/evaluation/datasets/${id}/download`)
export const getDatasetMediaUrl = (datasetId, path) => `/api/evaluation/datasets/${datasetId}/media/${path}`

// 评测配置
export const getEvalConfigs = () => request.get('/evaluation/configs')
export const getEvalConfig = (id) => request.get(`/evaluation/configs/${id}`)

// 评测运行
export const runEvaluation = (data) => request.post('/evaluation/run', data)

// 评测结果
export const getEvalResults = (params) => request.get('/evaluation/results', { params })
export const getEvalResultDetail = (id) => request.get(`/evaluation/results/${id}`)
export const compareResults = (params) => request.get('/evaluation/results/compare', { params })
export const getAggregatedResults = (params) => request.get('/evaluation/results/aggregate', { params })
export const getRowDemo = (params) => request.get('/evaluation/results/row_demo', { params })

// 输出视频/码流（按需服务）
export const listOutputs = () => request.get('/evaluation/outputs')
// 拼接按需播放 URL（<video preload="none"> 仅在挂载时才请求字节）
export const getOutputUrl = (path) => `/api/evaluation/outputs/${path}`
