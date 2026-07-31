import { Fragment, useState } from 'react'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, EmptyState, MetricTile, SectionHeader } from './ui'

function fmt(n, digits = 0) {
  if (n == null) return '—'
  return typeof n === 'number' ? n.toLocaleString(undefined, { maximumFractionDigits: digits }) : n
}

function TrashIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
  )
}

function ChevronIcon({ expanded }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className={`w-3.5 h-3.5 flex-shrink-0 transition-transform text-gray-400 ${expanded ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
    </svg>
  )
}

const DIM_LABEL = {
  judge_faithfulness:   'Faithfulness',
  judge_coverage:       'Coverage',
  judge_conciseness:    'Conciseness',
  judge_completeness:   'Completeness',
  judge_precision:      'Precision',
  judge_accuracy:       'Accuracy',
  judge_justifiability: 'Justifiability',
}

function ScorePill({ value, size = 'sm' }) {
  if (value == null) return null
  const pct = value * 100
  const color = pct >= 70 ? 'bg-green-100 text-green-700'
              : pct >= 40 ? 'bg-yellow-100 text-yellow-700'
              :              'bg-red-100 text-red-700'
  const cls = size === 'lg'
    ? `inline-flex items-center text-sm font-bold px-2.5 py-1 rounded-full ${color}`
    : `inline-flex items-center text-xs font-semibold px-2 py-0.5 rounded-full ${color}`
  return <span className={cls}>{(value * 100).toFixed(0)}%</span>
}

function DimPill({ dimKey, value }) {
  if (value == null) return null
  const pct = value * 100
  const color = pct >= 70 ? 'bg-purple-100 text-purple-700'
              : pct >= 40 ? 'bg-yellow-100 text-yellow-700'
              :              'bg-red-100 text-red-700'
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full ${color}`}>
      {DIM_LABEL[dimKey] ?? dimKey.replace('judge_', '')}: {pct.toFixed(0)}%
    </span>
  )
}

function OutputBlock({ label, value }) {
  if (value == null) return null
  let display = value
  if (typeof value !== 'string') {
    try { display = JSON.stringify(value, null, 2) } catch { display = String(value) }
  }
  return (
    <div>
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">{label}</p>
      <pre className="text-xs text-gray-700 bg-gray-50 rounded-lg p-3 whitespace-pre-wrap font-mono max-h-36 overflow-y-auto leading-relaxed">
        {display}
      </pre>
    </div>
  )
}

function TaskRow({ row }) {
  const [open, setOpen] = useState(false)
  const dims = row.judge_dimensions
    ? Object.entries(row.judge_dimensions).filter(([, v]) => v != null)
    : []
  const hasJudge = row.llm_judge_score != null

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      {/* Header — always visible */}
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-3 px-4 py-3 bg-white hover:bg-gray-50 transition-colors text-left"
      >
        <ChevronIcon expanded={open} />

        {/* Task ID */}
        <code className="text-xs font-mono text-gray-700 flex-1 truncate">{row.task_id}</code>

        {/* Badges */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">{row.task_type}</span>
          {row.difficulty && (
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              row.difficulty === 'easy'   ? 'bg-green-100 text-green-700'
            : row.difficulty === 'medium' ? 'bg-yellow-100 text-yellow-700'
            :                              'bg-red-100 text-red-700'
            }`}>{row.difficulty}</span>
          )}
          {hasJudge && <ScorePill value={row.llm_judge_score} />}
          {row.rubric_overridden === 1 && (
            <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full font-medium">custom rubric</span>
          )}
        </div>
      </button>

      {/* Expanded detail */}
      {open && (
        <div className="border-t border-gray-100 bg-gray-50 px-4 py-4 space-y-4">
          {/* Judge overall + dimensions */}
          {hasJudge && (
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">LLM Judge</p>
                <ScorePill value={row.llm_judge_score} size="lg" />
              </div>
              {dims.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {dims.map(([k, v]) => <DimPill key={k} dimKey={k} value={v} />)}
                </div>
              )}
            </div>
          )}

          {/* Reasoning */}
          {row.judge_reasoning && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Judge Reasoning</p>
              <p className="text-xs text-gray-700 bg-white border border-gray-200 rounded-lg p-3 leading-relaxed">
                {row.judge_reasoning}
              </p>
            </div>
          )}

          {/* Input / Expected / Prediction */}
          <div className="grid grid-cols-1 gap-3">
            <OutputBlock label="Expected" value={row.expected} />
            <OutputBlock label="Model Output" value={row.parsed_output} />
          </div>

          {row.parse_error && (
            <p className="text-xs text-red-600 bg-red-50 rounded-lg px-3 py-2">
              <span className="font-semibold">Parse error:</span> {row.parse_error}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function RunDetail({ runId }) {
  const { data: summary, loading: summaryLoading } = useApi(() => api.runSummary(runId), [runId])
  const { data: tasks,   loading: tasksLoading }   = useApi(() => api.results({ run_id: runId, limit: 1000 }), [runId])

  if (summaryLoading || tasksLoading) return (
    <div className="flex items-center gap-2 text-sm text-gray-500 py-1">
      <div className="w-4 h-4 border-2 border-blue-200 border-t-blue-500 rounded-full animate-spin" />
      Loading…
    </div>
  )

  return (
    <div className="space-y-5">
      {/* Summary tiles */}
      {summary && Object.keys(summary).length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          <MetricTile label="Tasks"       value={fmt(summary.task_count)} />
          <MetricTile label="Tokens"      value={fmt(summary.total_tokens)} />
          <MetricTile label="Cost (USD)"  value={summary.total_cost_usd != null ? `$${summary.total_cost_usd.toFixed(4)}` : '—'} />
          <MetricTile label="Avg Latency" value={summary.avg_latency_ms != null ? `${Math.round(summary.avg_latency_ms)} ms` : '—'} />
          <MetricTile label="LLM Judge"   value={summary.avg_llm_judge  != null ? summary.avg_llm_judge.toFixed(3) : '—'} />
        </div>
      )}

      {/* Per-task results */}
      {tasks?.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Task Results ({tasks.length})
          </p>
          <div className="space-y-1.5 max-h-[60vh] overflow-y-auto pr-1">
            {tasks.map(row => <TaskRow key={row.id ?? row.task_id} row={row} />)}
          </div>
        </div>
      )}

      {summary?.started_at && (
        <p className="text-xs text-gray-400">
          Started: {new Date(summary.started_at).toLocaleString()} ·
          Finished: {new Date(summary.finished_at).toLocaleString()}
        </p>
      )}
    </div>
  )
}

export default function RunHistory({ includeDryRuns = false }) {
  const [refreshKey,  setRefreshKey]  = useState(0)
  const [deletingId,  setDeletingId]  = useState(null)
  const [deleteError, setDeleteError] = useState(null)
  const [expandedId,  setExpandedId]  = useState(null)

  const { data: allRuns, loading, error } = useApi(() => api.runs(), [refreshKey])
  const runs = allRuns ? allRuns.filter(r => includeDryRuns || !r.is_dry_run) : null

  function toggleExpand(id) {
    setExpandedId(prev => prev === id ? null : id)
  }

  async function handleDelete(e, run) {
    e.stopPropagation()
    const confirmed = window.confirm(
      `Delete run ${run.run_id.slice(0, 8)}…?\n\n` +
      `Provider: ${run.provider} / ${run.model}\n` +
      `Tasks: ${run.task_count}\n\n` +
      `This will permanently remove the run and all its results.`
    )
    if (!confirmed) return

    setDeletingId(run.run_id)
    setDeleteError(null)
    try {
      await api.deleteRun(run.run_id)
      if (expandedId === run.run_id) setExpandedId(null)
      setRefreshKey(k => k + 1)
    } catch (err) {
      setDeleteError(`Failed to delete: ${err.message}`)
    } finally {
      setDeletingId(null)
    }
  }

  if (loading) return <Spinner />
  if (error)   return <ErrorCard message={error} />
  if (!runs?.length) return <EmptyState message="No benchmark runs found. Run a benchmark first." />

  return (
    <div className="space-y-4">
      <SectionHeader>All Runs</SectionHeader>

      {deleteError && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-700">
          {deleteError}
        </div>
      )}

      <div className="rounded-xl border border-gray-200 overflow-hidden">
        <table className="w-full text-sm text-left">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200 text-gray-600">
              <th className="px-4 py-3 font-semibold">Run ID</th>
              <th className="px-4 py-3 font-semibold">Provider / Model</th>
              <th className="px-4 py-3 font-semibold">Tasks</th>
              <th className="px-4 py-3 font-semibold whitespace-nowrap">Created</th>
              <th className="px-4 py-3 font-semibold">Notes</th>
              <th className="px-4 py-3" />
            </tr>
          </thead>
          <tbody>
            {runs.map(run => {
              const expanded = expandedId === run.run_id
              return (
                <Fragment key={run.run_id}>
                  <tr
                    onClick={() => toggleExpand(run.run_id)}
                    className={`border-b border-gray-100 cursor-pointer select-none transition-colors ${expanded ? 'bg-blue-50 border-blue-100' : 'hover:bg-gray-50'}`}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <ChevronIcon expanded={expanded} />
                        <code className="text-xs font-mono bg-gray-100 px-2 py-0.5 rounded tracking-wide" title={run.run_id}>
                          {run.run_id.slice(0, 8)}…
                        </code>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="font-medium text-gray-800">{run.provider}</div>
                      <div className="text-xs text-gray-400 mt-0.5">{run.model}</div>
                    </td>
                    <td className="px-4 py-3 text-gray-700">{run.task_count}</td>
                    <td className="px-4 py-3 text-gray-500 whitespace-nowrap">
                      <div>{new Date(run.created_at).toLocaleDateString()}</div>
                      <div className="text-xs text-gray-400 mt-0.5">{new Date(run.created_at).toLocaleTimeString()}</div>
                      {run.is_dry_run === 1 && (
                        <span className="inline-block mt-1 text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full font-medium">dry run</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-gray-400 max-w-xs truncate">{run.notes || '—'}</td>
                    <td className="px-4 py-3">
                      <button
                        onClick={e => handleDelete(e, run)}
                        disabled={deletingId === run.run_id}
                        title="Delete run"
                        className="p-1.5 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        {deletingId === run.run_id ? <span className="text-xs">…</span> : <TrashIcon />}
                      </button>
                    </td>
                  </tr>

                  {expanded && (
                    <tr>
                      <td colSpan={6} className="px-6 py-5 bg-blue-50 border-b border-blue-100">
                        <RunDetail runId={run.run_id} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
