import request from './request'

export const getResults = (params) => request.get('/benchmark/results', { params })
export const compareResults = (params) => request.get('/benchmark/results/compare', { params })
export const listRuns = () => request.get('/benchmark/runs')
export const runBenchmark = () => request.post('/benchmark/run')
