const BASE = '/api'

async function get(path, params = {}) {
  const url = new URL(BASE + path, window.location.origin)
  Object.entries(params).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== '') url.searchParams.set(k, v)
  })
  const res = await fetch(url)
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export const api = {
  runs: ()           => get('/runs'),
  runSummary: (id)   => get(`/runs/${id}`),
  breakdown: ()      => get('/scores/breakdown'),
  heatmap: ()        => get('/scores/heatmap'),
  costQuality: ()    => get('/scores/cost-quality'),
  results: (params)  => get('/results', params),
}
