import papersData from '../data/papers.json'

// 从静态 JSON 读取论文数据（无需后端）
const STORAGE_KEY_STARRED = 'ir-comp:starred'
const STORAGE_KEY_PINNED = 'ir-comp:pinned'
const STORAGE_KEY_NOTES = 'ir-comp:notes'

function getStarred() {
  try { return new Set(JSON.parse(localStorage.getItem(STORAGE_KEY_STARRED) || '[]')) }
  catch { return new Set() }
}

function getPinned() {
  try { return new Set(JSON.parse(localStorage.getItem(STORAGE_KEY_PINNED) || '[]')) }
  catch { return new Set() }
}

function getNotes() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY_NOTES) || '{}') }
  catch { return {} }
}

function decorate(paper) {
  const starred = getStarred()
  const pinned = getPinned()
  const notes = getNotes()
  return {
    ...paper,
    starred: starred.has(paper.id),
    pinned: pinned.has(paper.id),
    has_note: !!notes[paper.id],
  }
}

export function getPaperList(params = {}) {
  let papers = papersData.papers.map(decorate)

  // 分类筛选
  if (params.category) {
    papers = papers.filter(p => (p.categories || []).includes(params.category))
  }

  // 搜索
  if (params.search) {
    const q = params.search.toLowerCase()
    papers = papers.filter(p =>
      (p.title || '').toLowerCase().includes(q) ||
      (p.title_zh || '').toLowerCase().includes(q) ||
      (p.abstract || '').toLowerCase().includes(q) ||
      (p.abstract_zh || '').toLowerCase().includes(q) ||
      (p.tags || []).some(t => t.toLowerCase().includes(q))
    )
  }

  // 来源筛选
  if (params.source) {
    papers = papers.filter(p => p.source === params.source)
  }

  return { papers, total: papers.length }
}

export function getPaperStats() {
  const papers = papersData.papers
  const by_category = {}
  const by_source = {}
  papers.forEach(p => {
    ;(p.categories || []).forEach(c => { by_category[c] = (by_category[c] || 0) + 1 })
    by_source[p.source] = (by_source[p.source] || 0) + 1
  })
  return { total: papers.length, by_category, by_source }
}

export function getPaperDetail(id) {
  const paper = papersData.papers.find(p => p.id === id)
  if (!paper) return null
  return decorate(paper)
}

export function getPaperNote(id) {
  const notes = getNotes()
  return { content: notes[id] || '' }
}

export function savePaperNote(id, content) {
  const notes = getNotes()
  notes[id] = content
  localStorage.setItem(STORAGE_KEY_NOTES, JSON.stringify(notes))
  return { success: true }
}

export function starPaper(id, starred) {
  const set = getStarred()
  if (starred) set.add(id); else set.delete(id)
  localStorage.setItem(STORAGE_KEY_STARRED, JSON.stringify([...set]))
  return { success: true }
}

export function pinPaper(id, pinned) {
  const set = getPinned()
  if (pinned) set.add(id); else set.delete(id)
  localStorage.setItem(STORAGE_KEY_PINNED, JSON.stringify([...set]))
  return { success: true }
}

export function getThumbnailUrl(arxivId) {
  if (!arxivId) return ''
  return `https://arxiv.org/html/${arxivId}v1/x1.png`
}

export function getCategories() {
  return [
    { label: '学习式压缩', value: 'learned_compression' },
    { label: '生成式压缩', value: 'generative_compression' },
    { label: '联合信源信道编码', value: 'jscc' },
    { label: '视频压缩', value: 'video_compression' },
    { label: '红外图像压缩', value: 'infrared_compression' },
    { label: '视觉标记化', value: 'tokenization' },
    { label: '知识蒸馏', value: 'distillation' },
  ]
}
