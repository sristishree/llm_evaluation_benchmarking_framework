import { useState, useMemo, useEffect } from 'react'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, EmptyState, MetricTile, SectionHeader, Select, CopyButton } from './ui'

function fmt(n, digits = 0) {
  if (n == null) return '—'
  return typeof n === 'number' ? n.toLocaleString(undefined, { maximumFractionDigits: digits }) : n
}

// ── Icons ─────────────────────────────────────────────────────────────────────

function TrashIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
  )
}

function ChevronIcon({ expanded }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className={`w-3 h-3 flex-shrink-0 transition-transform ${expanded ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
    </svg>
  )
}

// ── Score helpers ──────────────────────────────────────────────────────────────

const DIM_LABEL = {
  judge_faithfulness:   'Faithfulness',
  judge_coverage:       'Coverage',
  judge_conciseness:    'Conciseness',
  judge_completeness:   'Completeness',
  judge_precision:      'Precision',
  judge_accuracy:       'Accuracy',
  judge_justifiability: 'Justifiability',
}

const METRIC_FIELDS = [
  { key: 'llm_judge_score', label: 'LLM Judge' },
  { key: 'rouge_l',         label: 'ROUGE-L' },
  { key: 'rouge_1',         label: 'ROUGE-1' },
  { key: 'rouge_2',         label: 'ROUGE-2' },
  { key: 'bert_score',      label: 'BERTScore' },
  { key: 'exact_match',     label: 'Exact Match' },
  { key: 'token_f1',        label: 'Token F1' },
]

function scoreBg(v) {
  if (v == null) return 'bg-gray-100 text-gray-500'
  const pct = v * 100
  return pct >= 70 ? 'bg-green-100 text-green-700'
       : pct >= 40 ? 'bg-yellow-100 text-yellow-700'
       :              'bg-red-100 text-red-700'
}

function ScorePill({ label, value, large }) {
  if (value == null) return null
  const cls = large
    ? `inline-flex items-center text-sm font-bold px-2.5 py-1 rounded-full ${scoreBg(value)}`
    : `inline-flex items-center text-xs font-semibold px-2 py-0.5 rounded-full ${scoreBg(value)}`
  const pct = (value * 100).toFixed(0) + '%'
  return <span className={cls}>{label ? `${label}: ${pct}` : pct}</span>
}

function DimPill({ dimKey, value }) {
  if (value == null) return null
  const pct = value * 100
  const color = pct >= 70 ? 'bg-purple-100 text-purple-700'
              : pct >= 40 ? 'bg-yellow-100 text-yellow-700'
              :              'bg-red-100 text-red-700'
  return (
    <span className={`inline-flex text-xs font-medium px-2 py-0.5 rounded-full ${color}`}>
      {DIM_LABEL[dimKey] ?? dimKey.replace('judge_', '')}: {pct.toFixed(0)}%
    </span>
  )
}

// ── Collapsible text block ─────────────────────────────────────────────────────

function CollapsibleBlock({ label, value, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  if (value == null) return null
  let display = value
  if (typeof value !== 'string') {
    try { display = JSON.stringify(value, null, 2) } catch { display = String(value) }
  }
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1 hover:text-gray-700 transition-colors"
      >
        <ChevronIcon expanded={open} />
        {label}
      </button>
      {open && (
        <pre className="text-xs text-gray-700 bg-gray-50 rounded-lg p-3 whitespace-pre-wrap font-mono max-h-52 overflow-y-auto leading-relaxed border border-gray-100">
          {display}
        </pre>
      )}
    </div>
  )
}

// ── Task card ──────────────────────────────────────────────────────────────────

function TaskCard({ row, judgeModel }) {
  const [open, setOpen] = useState(false)
  const dims    = row.judge_dimensions ? Object.entries(row.judge_dimensions).filter(([, v]) => v != null) : []
  const hasJudge = row.llm_judge_score != null
  const metrics  = METRIC_FIELDS.filter(f => row[f.key] != null)

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden bg-white">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-50 transition-colors text-left"
      >
        <ChevronIcon expanded={open} />
        <code className="text-xs font-mono text-gray-700 flex-1 truncate min-w-0">{row.task_id}</code>
        <div className="flex items-center gap-1.5 flex-shrink-0 flex-wrap justify-end">
          {row.task_type && (
            <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">{row.task_type}</span>
          )}
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

      {open && (
        <div className="border-t border-gray-100 bg-gray-50 px-4 py-4 space-y-4">
          {metrics.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Scores</p>
              <div className="flex flex-wrap gap-1.5">
                {metrics.map(f => <ScorePill key={f.key} label={f.label} value={row[f.key]} />)}
              </div>
            </div>
          )}

          {hasJudge && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 flex-wrap">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">LLM Judge</p>
                <ScorePill value={row.llm_judge_score} large />
                {judgeModel && judgeModel !== 'inline (pre-history)' && (
                  <code className="text-xs font-mono text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded" title={judgeModel}>
                    {judgeModel}
                  </code>
                )}
              </div>
              {dims.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {dims.map(([k, v]) => <DimPill key={k} dimKey={k} value={v} />)}
                </div>
              )}
              {row.judge_reasoning && (
                <div className="bg-amber-50 border border-amber-100 rounded-lg px-3 py-2.5">
                  <p className="text-xs font-semibold text-amber-700 uppercase tracking-wide mb-1">Reasoning</p>
                  <p className="text-xs text-gray-700 leading-relaxed">{row.judge_reasoning}</p>
                </div>
              )}
            </div>
          )}

          {(row.latency_ms != null || row.total_tokens != null) && (
            <div className="flex gap-4 text-xs text-gray-500">
              {row.latency_ms   != null && <span>{Math.round(row.latency_ms)} ms</span>}
              {row.total_tokens != null && <span>{row.total_tokens.toLocaleString()} tokens</span>}
            </div>
          )}

          <div className="space-y-3">
            <CollapsibleBlock label="Input"           value={row.task_input} />
            <CollapsibleBlock label="Expected Output" value={row.expected}   defaultOpen />
            <CollapsibleBlock label="Model Output"    value={row.parsed_output} defaultOpen />
          </div>

          {row.parse_error && (
            <p className="text-xs text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
              <span className="font-semibold">Parse error:</span> {row.parse_error}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

// ── Right pane: run detail ─────────────────────────────────────────────────────

const DIFFICULTY_ORDER = { easy: 0, medium: 1, hard: 2 }

function RunDetail({ runId }) {
  const [sortBy, setSortBy] = useState('judge_desc')

  const { data: summary,    loading: sumLoading  } = useApi(() => api.runSummary(runId), [runId])
  const { data: tasks,      loading: taskLoading } = useApi(() => api.results({ run_id: runId, limit: 1000 }), [runId])
  const { data: judgeRuns } = useApi(() => api.judgeRuns(runId), [runId])

  const inlineJudgeModel = useMemo(() => {
    if (!judgeRuns?.length) return null
    const inline = judgeRuns.find(jr => jr.source === 'benchmark' || jr.source === 'backfill')
    return inline?.judge_model ?? null
  }, [judgeRuns])

  const hasJudge = useMemo(() => tasks?.some(t => t.llm_judge_score != null) ?? false, [tasks])

  const sortOptions = useMemo(() => {
    const opts = []
    if (hasJudge) {
      opts.push({ value: 'judge_desc', label: 'Judge Score ↓' })
      opts.push({ value: 'judge_asc',  label: 'Judge Score ↑' })
    }
    opts.push({ value: 'id_asc',   label: 'Task ID A→Z' })
    opts.push({ value: 'diff_asc', label: 'Difficulty' })
    return opts
  }, [hasJudge])

  // Reset sort to a sensible default when the available options change
  useEffect(() => {
    setSortBy(prev => {
      if (sortOptions.some(o => o.value === prev)) return prev
      return hasJudge ? 'judge_desc' : 'id_asc'
    })
  }, [runId, sortOptions])

  const sorted = useMemo(() => {
    if (!tasks?.length) return []
    const effective = sortOptions.some(o => o.value === sortBy) ? sortBy : sortOptions[0]?.value
    return [...tasks].sort((a, b) => {
      if (effective === 'id_asc')     return a.task_id.localeCompare(b.task_id)
      if (effective === 'diff_asc')   return (DIFFICULTY_ORDER[a.difficulty] ?? 9) - (DIFFICULTY_ORDER[b.difficulty] ?? 9)
      if (effective === 'judge_desc') return (b.llm_judge_score ?? -1) - (a.llm_judge_score ?? -1)
      if (effective === 'judge_asc')  return (a.llm_judge_score ?? -1) - (b.llm_judge_score ?? -1)
      return 0
    })
  }, [tasks, sortBy, sortOptions])

  if (sumLoading || taskLoading) return (
    <div className="flex items-center justify-center gap-2 text-sm text-gray-500 h-full">
      <div className="w-4 h-4 border-2 border-blue-200 border-t-blue-500 rounded-full animate-spin" />
      Loading…
    </div>
  )

  return (
    <div className="h-full flex flex-col gap-4 p-1">
      {summary && Object.keys(summary).length > 0 && (
        <div className="grid grid-cols-5 gap-3 flex-shrink-0">
          <MetricTile label="Tasks"       value={fmt(summary.task_count)} />
          <MetricTile label="Tokens"      value={fmt(summary.total_tokens)} />
          <MetricTile label="Cost (USD)"  value={summary.total_cost_usd  > 0 ? `$${summary.total_cost_usd.toFixed(4)}` : '—'} />
          <MetricTile label="Avg Latency" value={summary.avg_latency_ms  != null ? `${Math.round(summary.avg_latency_ms)} ms` : '—'} />
          <MetricTile label="Avg Judge"   value={summary.avg_llm_judge   != null ? summary.avg_llm_judge.toFixed(3) : '—'} />
        </div>
      )}

      <div className="flex items-center gap-3 flex-shrink-0">
        <div className="w-48">
          <Select
            label="Sort tasks by"
            value={sortBy}
            onChange={setSortBy}
            options={sortOptions}
          />
        </div>
        {tasks?.length > 0 && (
          <span className="text-xs text-gray-400 self-end pb-0.5">{tasks.length} task{tasks.length !== 1 ? 's' : ''}</span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto space-y-1.5 pr-0.5">
        {sorted.length === 0
          ? <EmptyState message="No tasks in this run." />
          : sorted.map(row => <TaskCard key={row.id ?? row.task_id} row={row} judgeModel={inlineJudgeModel} />)
        }
      </div>

      {summary?.started_at && (
        <p className="text-xs text-gray-400 flex-shrink-0">
          Started: {new Date(summary.started_at).toLocaleString()} · Finished: {new Date(summary.finished_at).toLocaleString()}
        </p>
      )}
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function RunHistory({ includeDryRuns = false, pendingHistoryRunId, clearPendingHistoryRunId }) {
  const [refreshKey,      setRefreshKey]      = useState(0)
  const [deletingId,      setDeletingId]      = useState(null)
  const [deleteError,     setDeleteError]     = useState(null)
  const [selectedId,      setSelectedId]      = useState(null)
  const [filterProvider,  setFilterProvider]  = useState('')
  const [filterTimeframe, setFilterTimeframe] = useState('')
  const [sortRunsBy,      setSortRunsBy]      = useState('newest')

  useEffect(() => {
    if (!pendingHistoryRunId) return
    setSelectedId(pendingHistoryRunId)
    setFilterProvider('')
    setFilterTimeframe('')
    clearPendingHistoryRunId?.()
  }, [pendingHistoryRunId])

  const { data: allRuns, loading, error } = useApi(() => api.runs(), [refreshKey])

  const { runs, allProviders } = useMemo(() => {
    if (!allRuns) return { runs: null, allProviders: [] }
    const providers = [...new Set(allRuns.map(r => r.provider).filter(Boolean))].sort()
    let rows = allRuns.filter(r => includeDryRuns || !r.is_dry_run)
    if (filterProvider) rows = rows.filter(r => r.provider === filterProvider)
    if (filterTimeframe) {
      const cutoff = Date.now() - Number(filterTimeframe) * 24 * 60 * 60 * 1000
      rows = rows.filter(r => new Date(r.created_at).getTime() >= cutoff)
    }
    rows = [...rows].sort((a, b) => {
      const ta = new Date(a.created_at).getTime()
      const tb = new Date(b.created_at).getTime()
      return sortRunsBy === 'oldest' ? ta - tb : tb - ta
    })
    return { runs: rows, allProviders: providers }
  }, [allRuns, includeDryRuns, filterProvider, filterTimeframe, sortRunsBy])

  async function handleDelete(run) {
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
      if (selectedId === run.run_id) setSelectedId(null)
      setRefreshKey(k => k + 1)
    } catch (err) {
      setDeleteError(`Failed to delete: ${err.message}`)
    } finally {
      setDeletingId(null)
    }
  }

  if (loading) return <Spinner />
  if (error)   return <ErrorCard message={error} />
  if (!allRuns?.length) return <EmptyState message="No benchmark runs found. Run a benchmark first." />

  const activeId = (selectedId && runs?.some(r => r.run_id === selectedId)) ? selectedId : null

  return (
    <div className="space-y-4">
      <SectionHeader>Run History</SectionHeader>

      {deleteError && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-700">
          {deleteError}
        </div>
      )}

      <div className="flex gap-5" style={{ height: 'calc(100vh - 240px)', minHeight: 520 }}>

        {/* ── Left pane: runs table ── */}
        <div className="flex-[2] min-w-0 border border-gray-200 rounded-xl bg-white flex flex-col overflow-hidden">

          {/* Run filters */}
          <div className="px-4 py-3 border-b border-gray-100 bg-gray-50 flex items-end gap-3 flex-shrink-0 flex-wrap">
            <div className="w-36">
              <Select
                label="Provider"
                value={filterProvider}
                onChange={v => { setFilterProvider(v); setSelectedId(null) }}
                options={[{ value: '', label: 'All providers' }, ...allProviders.map(p => ({ value: p, label: p }))]}
              />
            </div>
            <div className="w-36">
              <Select
                label="Time frame"
                value={filterTimeframe}
                onChange={v => { setFilterTimeframe(v); setSelectedId(null) }}
                options={[
                  { value: '',   label: 'All time' },
                  { value: '1',  label: 'Last 24 h' },
                  { value: '7',  label: 'Last 7 days' },
                  { value: '30', label: 'Last 30 days' },
                  { value: '90', label: 'Last 90 days' },
                ]}
              />
            </div>
            <div className="w-36">
              <Select
                label="Sort by"
                value={sortRunsBy}
                onChange={setSortRunsBy}
                options={[
                  { value: 'newest', label: 'Newest first' },
                  { value: 'oldest', label: 'Oldest first' },
                ]}
              />
            </div>
            {runs && (
              <span className="text-xs text-gray-400 self-end pb-0.5 ml-auto">
                {runs.length} run{runs.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>

          {/* Table */}
          {!runs?.length
            ? <div className="flex-1 flex items-center justify-center">
                <EmptyState message="No runs match these filters." />
              </div>
            : <div className="flex-1 overflow-y-auto overflow-x-hidden">
                <table className="w-full text-sm text-left table-fixed">
                  <colgroup>
                    <col className="w-[130px]" />
                    <col className="w-[90px]" />
                    <col />
                    <col className="w-[46px]" />
                    <col className="w-[80px]" />
                    <col className="w-[32px]" />
                  </colgroup>
                  <thead className="sticky top-0 z-10">
                    <tr className="bg-gray-50 border-b border-gray-200 text-gray-600">
                      <th className="px-3 py-2.5 font-semibold text-xs uppercase tracking-wide">Run ID</th>
                      <th className="px-3 py-2.5 font-semibold text-xs uppercase tracking-wide">Provider</th>
                      <th className="px-3 py-2.5 font-semibold text-xs uppercase tracking-wide">Model</th>
                      <th className="px-3 py-2.5 font-semibold text-xs uppercase tracking-wide text-right">Tasks</th>
                      <th className="px-3 py-2.5 font-semibold text-xs uppercase tracking-wide">Created</th>
                      <th className="px-3 py-2.5 w-8" />
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map(run => {
                      const isSelected = run.run_id === activeId
                      return (
                        <tr
                          key={run.run_id}
                          onClick={() => setSelectedId(prev => prev === run.run_id ? null : run.run_id)}
                          className={`border-b border-gray-100 cursor-pointer select-none transition-colors group ${
                            isSelected ? 'bg-blue-50 border-blue-100' : 'hover:bg-gray-50'
                          }`}
                        >
                          <td className="px-3 py-3">
                            <div className="flex items-center gap-1 min-w-0">
                              <code className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded shrink-0" title={run.run_id}>
                                {run.run_id.slice(0, 8)}…
                              </code>
                              <CopyButton text={run.run_id} className="opacity-0 group-hover:opacity-100 shrink-0" />
                              {run.is_dry_run === 1 && (
                                <span className="text-xs bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded-full font-medium shrink-0">dry</span>
                              )}
                            </div>
                          </td>
                          <td className="px-3 py-3 font-medium text-gray-800 truncate">{run.provider}</td>
                          <td className="px-3 py-3 text-xs text-gray-500 font-mono truncate" title={run.model}>{run.model}</td>
                          <td className="px-3 py-3 text-gray-700 text-right tabular-nums">{run.task_count}</td>
                          <td className="px-3 py-3 text-gray-500">
                            <div className="text-xs whitespace-nowrap">{new Date(run.created_at).toLocaleDateString()}</div>
                            <div className="text-xs text-gray-400 whitespace-nowrap">{new Date(run.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                          </td>
                          <td className="px-3 py-3">
                            <button
                              onClick={e => { e.stopPropagation(); handleDelete(run) }}
                              disabled={deletingId === run.run_id}
                              title="Delete run"
                              className="p-1.5 rounded-lg text-gray-300 hover:text-red-600 hover:bg-red-50 transition-colors disabled:opacity-40 opacity-0 group-hover:opacity-100"
                            >
                              {deletingId === run.run_id ? <span className="text-xs">…</span> : <TrashIcon />}
                            </button>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
          }
        </div>

        {/* ── Right pane: task detail ── */}
        <div className="flex-[3] min-w-0 overflow-hidden">
          {activeId
            ? <RunDetail runId={activeId} />
            : <div className="h-full flex items-center justify-center">
                <EmptyState message="Select a run to inspect its tasks." />
              </div>
          }
        </div>
      </div>
    </div>
  )
}
