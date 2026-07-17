import axios from 'axios'

const request = axios.create({
  baseURL: '/api',
  timeout: 15000,
})

// In-memory GET cache: url → { data, expiry }
const _cache = new Map()
const _TTL = 30_000 // 30 s

request.interceptors.request.use(
  (config) => {
    if (config.method === 'get' || config.method === undefined) {
      const key = config.url + (config.params ? JSON.stringify(config.params) : '')
      const hit = _cache.get(key)
      if (hit && Date.now() < hit.expiry) {
        // Return a resolved promise that bypasses the actual request
        config._cacheKey = null
        config.adapter = () => Promise.resolve({ data: hit.data, status: 200, headers: {}, config })
      } else {
        config._cacheKey = key
      }
    }
    return config
  },
  (error) => Promise.reject(error)
)

request.interceptors.response.use(
  (response) => {
    const key = response.config._cacheKey
    if (key) {
      _cache.set(key, { data: response.data, expiry: Date.now() + _TTL })
    }
    return response.data
  },
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

export default request
