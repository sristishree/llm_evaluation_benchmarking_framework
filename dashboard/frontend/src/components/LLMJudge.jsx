import { useState, useEffect, useMemo } from 'react'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, EmptyState, SectionHeader, Select, CopyButton } from './ui'

// ── Constants ─────────────────────────────────────────────────────────────────

const DIM_LABEL = {
  judge_faithfulness:   'Faithfulness',
  judge_coverage:       'Coverage',
  judge_conciseness:    'Conciseness',
  judge_completeness:   'Completeness',
  judge_precision:      'Precision',
  judge_accuracy:       'Accuracy',
  judge_justifiability: 'Justifiability',
}

// ── Score helpers ──────────────────────────────────────────────────────────────

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

// ── Icons ──────────────────────────────────────────────────────────────────────

function ChevronIcon({ expanded }) {
  return (
    <svg className={`w-3 h-3 flex-shrink-0 transition-transform ${expanded ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
    </svg>
  )
}

// ── Collapsible text ───────────────────────────────────────────────────────────

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

// ── Task result card ───────────────────────────────────────────────────────────

function TaskResultCard({ row }) {
  const [open, setOpen] = useState(false)
  const dims = row.judge_dimensions
    ? Object.entries(row.judge_dimensions).filter(([, v]) => v != null)
    : []

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
          {row.llm_judge_score != null && <ScorePill value={row.llm_judge_score} />}
          {row.rubric_overridden === 1 && (
            <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full font-medium">custom rubric</span>
          )}
        </div>
      </button>

      {open && (
        <div className="border-t border-gray-100 bg-gray-50 px-4 py-4 space-y-4">
          {row.llm_judge_score != null && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">LLM Judge</p>
                <ScorePill value={row.llm_judge_score} large />
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

// ── Right pane: judge run results ──────────────────────────────────────────────

function JudgeRunDetail({ judgeRunId, judgeRun }) {
  const { data, loading, error } = useApi(
    () => api.judgeRunResults(judgeRunId),
    [judgeRunId],
  )

  if (loading) return (
    <div className="flex items-center justify-center gap-2 text-sm text-gray-500 h-full">
      <div className="w-4 h-4 border-2 border-blue-200 border-t-blue-500 rounded-full animate-spin" />
      Loading…
    </div>
  )
  if (error) return <ErrorCard message={error} />

  const judged = data?.filter(r => r.llm_judge_score != null) ?? []
  const avgScore = judged.length
    ? judged.reduce((s, r) => s + r.llm_judge_score, 0) / judged.length
    : null

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Session header */}
      <div className="flex items-center gap-3 flex-shrink-0 flex-wrap">
        {judgeRun?.judge_model && (
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-gray-400 uppercase tracking-wide font-semibold">Judge model</span>
            <code className="text-xs font-mono text-gray-800 bg-gray-100 px-2 py-0.5 rounded" title={judgeRun.judge_model}>
              {judgeRun.judge_model}
            </code>
          </div>
        )}
        {avgScore != null && (
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-gray-400 uppercase tracking-wide font-semibold">Avg score</span>
            <ScorePill value={avgScore} large />
          </div>
        )}
        <span className="text-xs text-gray-400 ml-auto">
          {data?.length ?? 0} task{data?.length !== 1 ? 's' : ''}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto space-y-1.5 pr-0.5">
        {!data?.length
          ? <EmptyState message="No results for this judge session." />
          : data.map(row => <TaskResultCard key={row.id} row={row} />)
        }
      </div>
    </div>
  )
}

// ── New judge run form ─────────────────────────────────────────────────────────

function NewJudgeRunForm({ onStarted }) {
  const { data: runs,    loading: runsLoading }    = useApi(() => api.runs())
  const { data: catalog, loading: catalogLoading } = useApi(() => api.catalog())

  const [runId,          setRunId]          = useState('')
  const [judgeProvider,  setJudgeProvider]  = useState('')
  const [judgeModel,     setJudgeModel]     = useState('')
  const [rubricOverride, setRubricOverride] = useState('')
  const [showRubric,     setShowRubric]     = useState(false)
  const [mitigateBias,   setMitigateBias]   = useState(false)
  const [submitting,     setSubmitting]     = useState(false)
  const [error,          setError]          = useState(null)

  // Rubric fetching
  const [taskTypes,      setTaskTypes]      = useState([])
  const [fetchedRubrics, setFetchedRubrics] = useState({}) // task_type → rubric string
  const [rubricLoading,  setRubricLoading]  = useState(false)

  const eligibleRuns = useMemo(
    () => (runs || []).filter(r => !r.is_dry_run),
    [runs],
  )

  const providers = useMemo(() => {
    if (!catalog) return []
    return Object.entries(catalog)
      .filter(([, v]) => typeof v === 'object' && v?.available_models)
      .map(([k]) => k)
      .sort()
  }, [catalog])

  const models = useMemo(() => {
    if (!catalog || !judgeProvider) return []
    return catalog[judgeProvider]?.available_models ?? []
  }, [catalog, judgeProvider])

  // Reset model to provider default when provider changes
  useEffect(() => {
    setJudgeModel(catalog?.[judgeProvider]?.default_model ?? '')
  }, [judgeProvider, catalog])

  // Fetch rubrics when a run is selected
  useEffect(() => {
    if (!runId) { setTaskTypes([]); setFetchedRubrics({}); setRubricOverride(''); return }
    setRubricLoading(true)
    api.results({ run_id: runId, limit: 50 })
      .then(results => {
        const types = [...new Set(results.map(r => r.task_type).filter(Boolean))]
        setTaskTypes(types)
        return Promise.all(
          types.map(t =>
            api.rubric({ task_type: t })
              .then(d => [t, d.rubric])
              .catch(() => [t, null])
          )
        )
      })
      .then(pairs => {
        const map = Object.fromEntries(pairs.filter(([, v]) => v != null))
        setFetchedRubrics(map)
        // Pre-fill override textarea when there is exactly one task type
        const values = Object.values(map)
        if (values.length === 1) setRubricOverride(values[0])
        else setRubricOverride('')
      })
      .catch(() => {})
      .finally(() => setRubricLoading(false))
  }, [runId])

  const canSubmit = runId && judgeModel && !submitting

  async function handleSubmit(e) {
    e.preventDefault()
    if (!canSubmit) return
    setSubmitting(true)
    setError(null)
    try {
      const result = await api.startJudgeRun(runId, {
        judge_provider:         judgeProvider || null,
        judge_model:            judgeModel,
        rubric_override:        showRubric && rubricOverride ? rubricOverride : null,
        mitigate_position_bias: mitigateBias,
      })
      onStarted(result.judge_run_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  if (runsLoading || catalogLoading) return <Spinner />

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {error && <ErrorCard message={error} />}

      {/* Benchmark run selector */}
      <div>
        <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide block mb-1">
          Benchmark Run
        </label>
        <select
          value={runId}
          onChange={e => setRunId(e.target.value)}
          required
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select a run…</option>
          {eligibleRuns.map(r => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id.slice(0, 8)}… — {r.provider} / {r.model} ({r.task_count} tasks · {new Date(r.created_at).toLocaleDateString()})
            </option>
          ))}
        </select>
        {!eligibleRuns.length && !runsLoading && (
          <p className="text-xs text-gray-400 mt-1">No benchmark runs found — run a benchmark first.</p>
        )}
      </div>

      {/* Current rubric(s) for selected run */}
      {runId && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              {taskTypes.length > 1 ? 'Current Rubrics' : 'Current Rubric'}
            </p>
            {rubricLoading && (
              <span className="text-xs text-gray-400 animate-pulse">Loading…</span>
            )}
          </div>

          {!rubricLoading && taskTypes.length === 0 && (
            <p className="text-xs text-gray-400">No tasks found in this run.</p>
          )}

          {taskTypes.map(type => (
            <div key={type} className="border border-gray-200 rounded-lg overflow-hidden">
              {taskTypes.length > 1 && (
                <div className="bg-gray-50 px-3 py-1.5 border-b border-gray-100">
                  <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">{type}</span>
                </div>
              )}
              {fetchedRubrics[type]
                ? <pre className="text-xs text-gray-700 px-3 py-2.5 whitespace-pre-wrap font-mono leading-relaxed max-h-40 overflow-y-auto bg-white">{fetchedRubrics[type]}</pre>
                : <p className="text-xs text-gray-400 px-3 py-2.5">{rubricLoading ? 'Loading…' : 'Rubric unavailable.'}</p>
              }
            </div>
          ))}
        </div>
      )}

      {/* Judge provider + model */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide block mb-1">Judge Provider</label>
          <select
            value={judgeProvider}
            onChange={e => setJudgeProvider(e.target.value)}
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select provider…</option>
            {providers.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide block mb-1">Judge Model</label>
          <select
            value={judgeModel}
            onChange={e => setJudgeModel(e.target.value)}
            required
            disabled={!judgeProvider}
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          >
            <option value="">{judgeProvider ? 'Select model…' : 'Pick provider first'}</option>
            {models.map(m => <option key={m.id} value={m.id}>{m.display_name}</option>)}
          </select>
        </div>
      </div>

      {/* Options */}
      <div className="space-y-2">
        <label className="flex items-center gap-2.5 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={mitigateBias}
            onChange={e => setMitigateBias(e.target.checked)}
            className="w-4 h-4 accent-blue-500"
          />
          <span className="text-sm text-gray-700">Mitigate position bias</span>
          <span className="text-xs text-gray-400">(runs judge twice, averages — doubles cost)</span>
        </label>

        <label className="flex items-center gap-2.5 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showRubric}
            onChange={e => setShowRubric(e.target.checked)}
            className="w-4 h-4 accent-blue-500"
          />
          <span className="text-sm text-gray-700">Override rubric for all tasks</span>
        </label>
      </div>

      {showRubric && (
        <div>
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide block mb-1">
            Rubric Override
            {taskTypes.length > 1 && (
              <span className="ml-1 font-normal text-gray-400 normal-case">· applies to all task types</span>
            )}
          </label>
          <textarea
            value={rubricOverride}
            onChange={e => setRubricOverride(e.target.value)}
            rows={6}
            placeholder="Scoring instructions for the judge…"
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm font-mono bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
          />
        </div>
      )}

      <button
        type="submit"
        disabled={!canSubmit}
        className="px-4 py-1.5 rounded-lg bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {submitting ? 'Starting…' : 'Run Judge'}
      </button>
    </form>
  )
}

// ── Judge run progress poller ──────────────────────────────────────────────────

function JudgeRunProgress({ judgeRunId, onDone }) {
  const [status, setStatus] = useState(null)

  useEffect(() => {
    let cancelled = false
    async function poll() {
      while (!cancelled) {
        try {
          const s = await api.judgeRunStatus(judgeRunId)
          if (!cancelled) setStatus(s)
          if (s.state === 'done' || s.state === 'error') {
            if (!cancelled) onDone(s.state)
            break
          }
        } catch { /* ignore transient errors */ }
        await new Promise(r => setTimeout(r, 1500))
      }
    }
    poll()
    return () => { cancelled = true }
  }, [judgeRunId, onDone])

  if (!status) return <Spinner />

  const { state, progress = 0, total = 0, error } = status
  const pct = total > 0 ? Math.round((progress / total) * 100) : 0
  const isDone  = state === 'done'
  const isError = state === 'error'

  return (
    <div className="space-y-3">
      <div className="flex justify-between text-xs text-gray-600">
        <span>
          {isDone  ? `Done — ${total} task${total !== 1 ? 's' : ''} judged` :
           isError ? 'Judge run failed' :
                     `Judging task ${progress} of ${total}`}
        </span>
        {!isDone && !isError && <span>{pct}%</span>}
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${
            isDone ? 'bg-green-500' : isError ? 'bg-red-500' : 'bg-blue-500'
          }`}
          style={{ width: `${isDone ? 100 : pct}%` }}
        />
      </div>
      {isError && error && (
        <p className="text-xs text-red-600 bg-red-50 rounded-lg px-3 py-2">{error}</p>
      )}
    </div>
  )
}

// ── Left pane: judge run list ──────────────────────────────────────────────────

function JudgeRunListItem({ jr, isSelected, onClick, onRunClick }) {
  const stateColor = jr.state === 'done'    ? 'bg-green-100 text-green-700'
                   : jr.state === 'error'   ? 'bg-red-100 text-red-700'
                   :                          'bg-blue-100 text-blue-700'
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full text-left px-3 py-3 rounded-lg border transition-colors ${
        isSelected ? 'bg-blue-50 border-blue-200' : 'border-transparent hover:bg-gray-50'
      }`}
    >
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1 flex-1 min-w-0">
          <code className="text-xs font-mono text-gray-600">{jr.judge_run_id}</code>
          <CopyButton text={jr.judge_run_id} />
        </div>
        <span className={`text-xs font-medium px-1.5 py-0.5 rounded-full flex-shrink-0 ${stateColor}`}>
          {jr.state}
        </span>
      </div>
      <div className="mt-1 flex items-center gap-1.5 min-w-0">
        <span className="text-xs text-gray-400 shrink-0">Model:</span>
        {jr.judge_model === 'inline (pre-history)'
          ? <span className="text-xs text-gray-400 italic">Unknown (pre-recorded run)</span>
          : <code className="text-xs font-mono text-gray-700 bg-gray-100 px-1.5 py-0.5 rounded truncate" title={jr.judge_model}>{jr.judge_model}</code>
        }
      </div>
      <div className="flex items-center gap-1 mt-1 text-xs text-gray-400 flex-wrap">
        <span>on</span>
        <button
          type="button"
          onClick={e => { e.stopPropagation(); onRunClick?.(jr.run_id) }}
          className="font-mono text-blue-500 hover:text-blue-700 hover:underline"
        >{jr.run_id}</button>
        <CopyButton text={jr.run_id} />
        <span>·</span>
        <span>{jr.task_count} tasks</span>
        <span>·</span>
        <span>{new Date(jr.created_at).toLocaleDateString()}</span>
      </div>
      {jr.rubric_override && (
        <span className="mt-1 inline-block text-xs bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded-full font-medium">
          custom rubric
        </span>
      )}
    </button>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function LLMJudge({ navigateToRun }) {
  const [selectedJudgeRunId, setSelectedJudgeRunId] = useState(null)
  const [showNewForm,        setShowNewForm]        = useState(false)
  const [inFlightId,         setInFlightId]         = useState(null)
  const [refreshKey,         setRefreshKey]         = useState(0)
  const [filterRunId,        setFilterRunId]        = useState('')

  const { data: judgeRuns, loading, error } = useApi(
    () => api.judgeRuns(filterRunId || null),
    [refreshKey, filterRunId],
  )

  const { data: allRuns } = useApi(() => api.runs(), [refreshKey])

  const eligibleRuns = useMemo(
    () => (allRuns || []).filter(r => !r.is_dry_run),
    [allRuns],
  )

  function handleStarted(judgeRunId) {
    setInFlightId(judgeRunId)
    setShowNewForm(false)
    setSelectedJudgeRunId(null)
  }

  function handleDone() {
    setInFlightId(null)
    setRefreshKey(k => k + 1)
    setSelectedJudgeRunId(inFlightId)
  }

  const activeJudgeRunId = showNewForm || inFlightId ? null : selectedJudgeRunId

  return (
    <div className="space-y-4">
      <SectionHeader>LLM Judge</SectionHeader>

      <div className="flex gap-5" style={{ height: 'calc(100vh - 240px)', minHeight: 520 }}>

        {/* ── Left pane: history ── */}
        <div className="flex-[2] min-w-0 border border-gray-200 rounded-xl bg-white flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-100 bg-gray-50 flex flex-col gap-2.5 flex-shrink-0">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Judge Sessions {judgeRuns?.length ? `(${judgeRuns.length})` : ''}
              </p>
              <button
                type="button"
                onClick={() => { setShowNewForm(true); setSelectedJudgeRunId(null) }}
                className="text-xs font-semibold text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 px-3 py-1.5 rounded-lg transition-colors"
              >
                + New Session
              </button>
            </div>
            {eligibleRuns.length > 0 && (
              <Select
                label="Benchmark Run"
                value={filterRunId}
                onChange={v => { setFilterRunId(v); setSelectedJudgeRunId(null) }}
                options={[
                  { value: '', label: 'All runs' },
                  ...eligibleRuns.map(r => ({
                    value: r.run_id,
                    label: `${r.run_id.slice(0, 8)}… · ${r.provider} / ${r.model.split('/').pop()} (${r.task_count} tasks)`,
                  })),
                ]}
              />
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5">
            {loading && <Spinner />}
            {error   && <ErrorCard message={error} />}
            {!loading && !error && !judgeRuns?.length && (
              <div className="p-4">
                <EmptyState message="No judge sessions yet. Start one with the button above." />
              </div>
            )}
            {judgeRuns?.map(jr => (
              <JudgeRunListItem
                key={jr.judge_run_id}
                jr={jr}
                isSelected={jr.judge_run_id === selectedJudgeRunId && !showNewForm}
                onClick={() => { setSelectedJudgeRunId(jr.judge_run_id); setShowNewForm(false) }}
                onRunClick={navigateToRun}
              />
            ))}
          </div>
        </div>

        {/* ── Right pane ── */}
        <div className="flex-[3] min-w-0 overflow-hidden">
          {/* In-flight progress */}
          {inFlightId && (
            <div className="bg-white border border-gray-200 rounded-xl p-5 mb-4">
              <p className="text-sm font-semibold text-gray-700 mb-3">
                Judge session in progress… <code className="text-xs font-mono text-gray-400">{inFlightId.slice(0, 8)}…</code>
              </p>
              <JudgeRunProgress judgeRunId={inFlightId} onDone={handleDone} />
            </div>
          )}

          {/* New run form */}
          {showNewForm && !inFlightId && (
            <div className="bg-white border border-gray-200 rounded-xl p-5 h-full overflow-y-auto">
              <p className="text-sm font-semibold text-gray-800 mb-4">New Judge Session</p>
              <NewJudgeRunForm onStarted={handleStarted} />
            </div>
          )}

          {/* Selected judge run results */}
          {activeJudgeRunId && !inFlightId && (
            <JudgeRunDetail
              judgeRunId={activeJudgeRunId}
              judgeRun={judgeRuns?.find(jr => jr.judge_run_id === activeJudgeRunId)}
            />
          )}

          {/* Default empty state */}
          {!showNewForm && !inFlightId && !activeJudgeRunId && (
            <div className="h-full flex items-center justify-center">
              <EmptyState message="Select a judge session to view results, or start a new one." />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
