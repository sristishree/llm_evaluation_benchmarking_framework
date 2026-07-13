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

async function post(path, body) {
  const res = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

async function del(path) {
  const res = await fetch(BASE + path, { method: 'DELETE' })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export const api = {
  catalog:      ()           => get('/catalog'),
  taskBank:     (refresh)    => get('/task-bank', refresh ? { refresh: true } : {}),
  runs:         ()           => get('/runs'),
  runSummary:   (id)         => get(`/runs/${id}`),
  runStatus:    (id)         => get(`/runs/${id}/status`),
  startRun:     (params)     => post('/runs', params),
  deleteRun:    (id)         => del(`/runs/${id}`),
  breakdown:    (dryRuns)    => get('/scores/breakdown',   dryRuns ? { include_dry_runs: true } : {}),
  heatmap:      (dryRuns)    => get('/scores/heatmap',      dryRuns ? { include_dry_runs: true } : {}),
  costQuality:  (dryRuns)    => get('/scores/cost-quality', dryRuns ? { include_dry_runs: true } : {}),
  results:      (params)     => get('/results', params),
  distribution: (dryRuns)   => get('/scores/distribution', dryRuns ? { include_dry_runs: true } : {}),
}
