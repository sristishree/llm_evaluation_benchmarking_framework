import { useState } from 'react'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, EmptyState, MetricTile, SectionHeader, Table } from './ui'

const RUN_COLS = [
  { key: 'run_id',     label: 'Run ID',    render: r => <code className="text-xs bg-gray-100 px-2 py-0.5 rounded">{r.run_id.slice(0, 8)}…</code> },
  { key: 'provider',   label: 'Provider' },
  { key: 'model',      label: 'Model' },
  { key: 'task_count', label: 'Tasks' },
  { key: 'created_at', label: 'Created', render: r => new Date(r.created_at).toLocaleString() },
  { key: 'notes',      label: 'Notes', render: r => r.notes || <span className="text-gray-400">—</span> },
]

function fmt(n, digits = 0) {
  if (n == null) return '—'
  return typeof n === 'number' ? n.toLocaleString(undefined, { maximumFractionDigits: digits }) : n
}

export default function RunHistory() {
  const { data: runs, loading, error } = useApi(() => api.runs())
  const [selectedId, setSelectedId] = useState(null)
  const { data: summary, loading: sumLoading } = useApi(
    () => selectedId ? api.runSummary(selectedId) : Promise.resolve(null),
    [selectedId],
  )

  if (loading) return <Spinner />
  if (error)   return <ErrorCard message={error} />
  if (!runs?.length) return <EmptyState message="No benchmark runs found. Run a benchmark first." />

  return (
    <div className="space-y-8">
      <div>
        <SectionHeader>All Runs</SectionHeader>
        <Table
          columns={RUN_COLS}
          rows={runs}
          keyFn={r => r.run_id}
        />
      </div>

      <div>
        <SectionHeader>Inspect a Run</SectionHeader>
        <div className="mb-4">
          <select
            value={selectedId || ''}
            onChange={e => setSelectedId(e.target.value || null)}
            className="border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 min-w-80"
          >
            <option value="">— select a run —</option>
            {runs.map(r => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id.slice(0, 8)}… · {r.provider} / {r.model} · {r.task_count} tasks
              </option>
            ))}
          </select>
        </div>

        {sumLoading && <Spinner />}

        {summary && Object.keys(summary).length > 0 && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 mb-4">
              <MetricTile label="Tasks"       value={fmt(summary.task_count)} />
              <MetricTile label="Tokens"      value={fmt(summary.total_tokens)} />
              <MetricTile label="Cost (USD)"  value={summary.total_cost_usd != null ? `$${summary.total_cost_usd.toFixed(4)}` : '—'} />
              <MetricTile label="Avg Latency" value={summary.avg_latency_ms != null ? `${Math.round(summary.avg_latency_ms)} ms` : '—'} />
              <MetricTile label="LLM Judge"   value={summary.avg_llm_judge != null ? summary.avg_llm_judge.toFixed(3) : '—'} />
            </div>
            {summary.started_at && (
              <p className="text-xs text-gray-400">
                Started: {new Date(summary.started_at).toLocaleString()} ·
                Finished: {new Date(summary.finished_at).toLocaleString()}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  )
}
