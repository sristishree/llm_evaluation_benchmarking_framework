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

function RunDetail({ runId }) {
  const { data: summary, loading } = useApi(() => api.runSummary(runId), [runId])

  if (loading) return (
    <div className="flex items-center gap-2 text-sm text-gray-500 py-1">
      <div className="w-4 h-4 border-2 border-blue-200 border-t-blue-500 rounded-full animate-spin" />
      Loading…
    </div>
  )

  if (!summary || !Object.keys(summary).length) return (
    <p className="text-sm text-gray-400">No summary available.</p>
  )

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <MetricTile label="Tasks"       value={fmt(summary.task_count)} />
        <MetricTile label="Tokens"      value={fmt(summary.total_tokens)} />
        <MetricTile label="Cost (USD)"  value={summary.total_cost_usd  != null ? `$${summary.total_cost_usd.toFixed(4)}` : '—'} />
        <MetricTile label="Avg Latency" value={summary.avg_latency_ms  != null ? `${Math.round(summary.avg_latency_ms)} ms` : '—'} />
        <MetricTile label="LLM Judge"   value={summary.avg_llm_judge   != null ? summary.avg_llm_judge.toFixed(3) : '—'} />
      </div>
      {summary.started_at && (
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
                        <code
                          className="text-xs font-mono bg-gray-100 px-2 py-0.5 rounded tracking-wide"
                          title={run.run_id}
                        >
                          {run.run_id}
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
                        <span className="inline-block mt-1 text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full font-medium">
                          dry run
                        </span>
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
