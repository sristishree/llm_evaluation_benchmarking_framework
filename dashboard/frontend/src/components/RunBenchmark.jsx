import { useState, useEffect, useMemo } from 'react'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, SectionHeader } from './ui'


const TASK_TYPES = [
  { id: 'classification', label: 'Classification' },
  { id: 'qa',             label: 'Q&A' },
  { id: 'summarization',  label: 'Summarization' },
  { id: 'extraction',     label: 'NER Extraction' },
]

const DIFFICULTIES = [
  { value: '',       label: 'All' },
  { value: 'easy',   label: 'Easy' },
  { value: 'medium', label: 'Medium' },
  { value: 'hard',   label: 'Hard' },
]

const DIFF_COLORS = { easy: '#10b981', medium: '#f59e0b', hard: '#ef4444' }

// ── Difficulty distribution bar ───────────────────────────────────────────

function DiffBar({ counts }) {
  const total = (counts.easy ?? 0) + (counts.medium ?? 0) + (counts.hard ?? 0)
  if (!total) return null
  return (
    <div className="space-y-1">
      <div className="flex rounded-full overflow-hidden h-2">
        {['easy', 'medium', 'hard'].map(d => {
          const pct = ((counts[d] ?? 0) / total) * 100
          return pct > 0 ? (
            <div key={d} style={{ width: `${pct}%`, backgroundColor: DIFF_COLORS[d] }} />
          ) : null
        })}
      </div>
      <div className="flex gap-3 text-xs text-gray-400">
        {['easy', 'medium', 'hard'].map(d => (
          <span key={d} style={{ color: DIFF_COLORS[d] }} className="font-medium">
            {d.charAt(0).toUpperCase() + d.slice(1)}: {counts[d] ?? 0}
          </span>
        ))}
      </div>
    </div>
  )
}

// ── Single dataset card ───────────────────────────────────────────────────

function DatasetCard({ dataset, sel, onChange }) {
  const { file, name, description, total, difficulty: counts, is_combined } = dataset
  const maxForDiff = sel.difficulty
    ? (counts[sel.difficulty] ?? 0)
    : total

  return (
    <div className={`rounded-xl border p-4 transition-colors ${
      sel.enabled
        ? 'border-blue-300 bg-blue-50/40'
        : 'border-gray-200 bg-white opacity-60'
    }`}>
      {/* Header row */}
      <div className="flex items-start gap-3">
        <input
          type="checkbox"
          checked={sel.enabled}
          onChange={e => onChange({ ...sel, enabled: e.target.checked })}
          className="mt-0.5 w-4 h-4 accent-blue-500 shrink-0"
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-sm text-gray-800">{name}</span>
            {is_combined && (
              <span className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">combined</span>
            )}
            <span className="text-xs text-gray-400 ml-auto">{total.toLocaleString()} entries</span>
          </div>
          <p className="text-xs text-gray-500 mt-0.5">{description}</p>
        </div>
      </div>

      {/* Difficulty bar */}
      <div className="mt-3 ml-7">
        <DiffBar counts={counts} />
      </div>

      {/* Controls — only when enabled */}
      {sel.enabled && (
        <div className="mt-3 ml-7 flex items-center gap-4 flex-wrap">
          {/* Difficulty filter — only show difficulties that have entries */}
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-gray-500 whitespace-nowrap">Difficulty:</label>
            <select
              value={sel.difficulty}
              onChange={e => {
                const diff = e.target.value
                const available = diff ? (counts[diff] ?? 0) : total
                onChange({ ...sel, difficulty: diff, limit: Math.min(Math.max(sel.limit, 1), available) })
              }}
              className="border border-gray-200 rounded px-2 py-1 text-xs bg-white focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {DIFFICULTIES.filter(o => !o.value || (counts[o.value] ?? 0) > 0).map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          {/* Sample count */}
          {maxForDiff > 0 ? (
            <div className="flex items-center gap-1.5">
              <label className="text-xs text-gray-500 whitespace-nowrap">Samples:</label>
              <input
                type="number"
                min={1}
                max={maxForDiff}
                value={sel.limit}
                onChange={e => {
                  const v = Number(e.target.value)
                  if (!Number.isNaN(v)) onChange({ ...sel, limit: Math.min(Math.max(v, 1), maxForDiff) })
                }}
                className="border border-gray-200 rounded px-2 py-1 text-xs w-20 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-xs text-gray-400">/ {maxForDiff.toLocaleString()}</span>
            </div>
          ) : (
            <span className="text-xs text-amber-600">No entries for this difficulty</span>
          )}
        </div>
      )}
    </div>
  )
}

// ── Progress bar ──────────────────────────────────────────────────────────

function ProgressBar({ progress, total, state }) {
  const pct = total > 0 ? Math.round((progress / total) * 100) : 0
  const color = state === 'done' ? 'bg-green-500' : state === 'error' ? 'bg-red-500' : 'bg-blue-500'
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>
          {state === 'loading' && 'Loading tasks from task bank…'}
          {state === 'running' && `Running task ${progress} of ${total}`}
          {state === 'done'    && `Done — ${total} task${total !== 1 ? 's' : ''} completed`}
          {state === 'error'   && 'Run failed'}
        </span>
        {state !== 'loading' && <span>{pct}%</span>}
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
        <div
          className={`h-2.5 rounded-full transition-all duration-500 ${color} ${state === 'loading' ? 'animate-pulse w-full' : ''}`}
          style={state !== 'loading' ? { width: `${pct}%` } : undefined}
        />
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────

export default function RunBenchmark({ activeRunId, setActiveRunId, runStatus }) {
  const { data: catalog, loading: catalogLoading, error: catalogError } = useApi(() => api.catalog())
  const { data: taskBank, loading: bankLoading, error: bankError } = useApi(() => api.taskBank())

  // Config
  const [provider, setProvider]     = useState('')
  const [model, setModel]           = useState('')
  const [dryRun, setDryRun]         = useState(false)
  const [notes, setNotes]           = useState('')
  const [useJudge, setUseJudge]     = useState(false)
  const [judgeProvider, setJudgeProvider] = useState('')
  const [judgeModel, setJudgeModel] = useState('')

  // Task type + per-dataset selections
  const [taskType, setTaskType]     = useState('classification')
  const [selections, setSelections] = useState({})

  const [submitErr, setSubmitErr]   = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const status = runStatus  // alias — passed from App

  // Populate provider default once catalog loads
  useEffect(() => {
    if (catalog && !provider) {
      const first = Object.keys(catalog).find(k => typeof catalog[k] === 'object' && catalog[k]?.available_models)
      if (first) {
        setProvider(first); setModel(catalog[first]?.default_model ?? '')
        setJudgeProvider(first); setJudgeModel(catalog[first]?.default_model ?? '')
      }
    }
  }, [catalog])

  useEffect(() => {
    if (catalog && provider) setModel(catalog[provider]?.default_model ?? '')
  }, [provider, catalog])

  useEffect(() => {
    if (catalog && judgeProvider) setJudgeModel(catalog[judgeProvider]?.default_model ?? '')
  }, [judgeProvider, catalog])

  // Re-initialise selections when task type or task bank changes
  useEffect(() => {
    if (!taskBank?.[taskType]) return
    setSelections(prev => {
      const next = {}
      for (const ds of taskBank[taskType]) {
        next[ds.file] = prev[ds.file] ?? {
          enabled:    !ds.is_combined,
          limit:      Math.min(10, ds.total),
          difficulty: '',
        }
      }
      return next
    })
  }, [taskType, taskBank])

  const providerOptions = useMemo(() => {
    if (!catalog) return []
    return Object.entries(catalog)
      .filter(([, v]) => typeof v === 'object' && v?.available_models)
      .map(([k]) => ({ value: k, label: k.charAt(0).toUpperCase() + k.slice(1) }))
  }, [catalog])

  const modelOptions = useMemo(() => {
    if (!catalog || !provider) return []
    return (catalog[provider]?.available_models ?? []).map(m => ({ value: m.id, label: m.display_name }))
  }, [catalog, provider])

  const judgeModelOptions = useMemo(() => {
    if (!catalog || !judgeProvider) return []
    return (catalog[judgeProvider]?.available_models ?? []).map(m => ({ value: m.id, label: m.display_name }))
  }, [catalog, judgeProvider])

  const datasetsForType = taskBank?.[taskType] ?? []

  const totalSelected = datasetsForType.reduce((sum, ds) => {
    const s = selections[ds.file]
    return sum + (s?.enabled ? s.limit : 0)
  }, 0)

  const isRunning = status && (status.state === 'loading' || status.state === 'running')
  const isDone    = status?.state === 'done'
  const isError   = status?.state === 'error'

  async function handleSubmit(e) {
    e.preventDefault()
    setSubmitErr(null)
    setActiveRunId(null)
    setSubmitting(true)
    try {
      const datasets = datasetsForType
        .filter(ds => selections[ds.file]?.enabled)
        .map(ds => ({
          file:       ds.file,
          limit:      selections[ds.file].limit,
          difficulty: selections[ds.file].difficulty || null,
        }))
      if (!datasets.length) throw new Error('Select at least one dataset.')
      const { run_id } = await api.startRun({
        provider, model, datasets, dry_run: dryRun, notes: notes || null,
        use_judge: useJudge && !dryRun,
        judge_provider: useJudge && !dryRun ? judgeProvider : null,
        judge_model:    useJudge && !dryRun ? judgeModel    : null,
      })
      setActiveRunId(run_id)
    } catch (err) {
      setSubmitErr(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  if (catalogLoading || bankLoading) return <Spinner />
  if (catalogError) return <ErrorCard message={catalogError} />
  if (bankError)    return <ErrorCard message={bankError} />

  return (
    <div className="space-y-5">
      <SectionHeader>Run a Benchmark</SectionHeader>

      {/* Task type tabs */}
      <div className="flex gap-1 bg-gray-100 p-1 rounded-xl w-fit">
        {TASK_TYPES.map(t => (
          <button
            key={t.id}
            onClick={() => setTaskType(t.id)}
            disabled={!!isRunning}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
              taskType === t.id
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            } disabled:opacity-50`}
          >
            {t.label}
            {taskBank?.[t.id] && (
              <span className="ml-1.5 text-xs text-gray-400">
                ({taskBank[t.id].length})
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Main two-column layout */}
      <form onSubmit={handleSubmit} className="grid grid-cols-[300px_1fr] gap-6 items-start">

        {/* LEFT — Config */}
        <div className="space-y-4 bg-white rounded-xl border border-gray-200 shadow-sm p-5 sticky top-6">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Configuration</p>

          {/* Provider */}
          <div>
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Provider</label>
            <select value={provider} onChange={e => setProvider(e.target.value)} disabled={!!isRunning}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50">
              {providerOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>

          {/* Model */}
          <div>
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Model</label>
            <select value={model} onChange={e => setModel(e.target.value)} disabled={!!isRunning}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50">
              {modelOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>

          {/* Notes */}
          <div>
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
              Notes <span className="font-normal text-gray-400">(optional)</span>
            </label>
            <input type="text" value={notes} onChange={e => setNotes(e.target.value)} disabled={!!isRunning}
              placeholder="e.g. baseline, temp=0"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50" />
          </div>

          {/* Dry run */}
          <label className="flex items-start gap-2 cursor-pointer">
            <input type="checkbox" checked={dryRun} onChange={e => setDryRun(e.target.checked)} disabled={!!isRunning}
              className="mt-0.5 w-4 h-4 accent-blue-500" />
            <span className="text-sm text-gray-700">
              Dry run
              <span className="block text-xs text-gray-400 mt-0.5">No API calls — placeholder responses</span>
            </span>
          </label>

          {/* LLM-as-Judge */}
          <label className="flex items-start gap-2 cursor-pointer">
            <input type="checkbox" checked={useJudge} onChange={e => setUseJudge(e.target.checked)}
              disabled={!!isRunning || dryRun} className="mt-0.5 w-4 h-4 accent-purple-500" />
            <span className="text-sm text-gray-700">
              LLM-as-Judge
              <span className="block text-xs text-gray-400 mt-0.5">Score outputs with a judge model</span>
            </span>
          </label>

          {useJudge && !dryRun && (
            <div className="ml-6 space-y-2">
              <div>
                <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Judge Provider</label>
                <select value={judgeProvider} onChange={e => setJudgeProvider(e.target.value)} disabled={!!isRunning}
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50">
                  {providerOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Judge Model</label>
                <select value={judgeModel} onChange={e => setJudgeModel(e.target.value)} disabled={!!isRunning}
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50">
                  {judgeModelOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
            </div>
          )}

          <div className="border-t border-gray-100 pt-4 space-y-3">
            {/* Total summary */}
            <p className="text-xs text-gray-500">
              <span className="font-semibold text-gray-800">{totalSelected}</span> tasks selected across{' '}
              <span className="font-semibold text-gray-800">
                {datasetsForType.filter(ds => selections[ds.file]?.enabled).length}
              </span> dataset{datasetsForType.filter(ds => selections[ds.file]?.enabled).length !== 1 ? 's' : ''}
            </p>

            {submitErr && (
              <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-700">{submitErr}</div>
            )}

            <button type="submit" disabled={submitting || !!isRunning || !provider || !model || totalSelected === 0}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed text-white font-semibold text-sm px-4 py-3 rounded-lg transition-colors">
              {submitting ? 'Starting…' : isRunning ? 'Running…' : `▶  Run ${totalSelected} task${totalSelected !== 1 ? 's' : ''}`}
            </button>
          </div>

          {/* Progress */}
          {status && (
            <div className={`rounded-xl border p-4 space-y-3 ${
              isDone ? 'border-green-200 bg-green-50' : isError ? 'border-red-200 bg-red-50' : 'border-blue-200 bg-blue-50'
            }`}>
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold text-gray-800">
                  {isDone ? '✅ Done' : isError ? '❌ Failed' : '⏳ Running'}
                </p>
                {(isDone || isError) && (
                  <button type="button" onClick={() => setActiveRunId(null)}
                    className="text-xs text-gray-400 hover:text-gray-600">reset</button>
                )}
              </div>
              <ProgressBar progress={status.progress ?? 0} total={status.total ?? 0} state={status.state} />
              {isError && status.error && (
                <p className="text-xs text-red-700 break-all">{status.error}</p>
              )}
              {isDone && (
                <p className="text-xs text-green-700">
                  Saved to DB. Check <strong>Run History</strong> or <strong>Score Breakdown</strong>.
                </p>
              )}
              <p className="text-xs text-gray-400 font-mono break-all">{status.run_id?.slice(0, 16)}…</p>
            </div>
          )}
        </div>

        {/* RIGHT — Dataset selector */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Datasets — {TASK_TYPES.find(t => t.id === taskType)?.label}
            </p>
            <div className="flex gap-2">
              <button type="button"
                onClick={() => setSelections(prev => {
                  const next = { ...prev }
                  datasetsForType.forEach(ds => { if (next[ds.file]) next[ds.file] = { ...next[ds.file], enabled: true } })
                  return next
                })}
                className="text-xs text-blue-600 hover:underline">Select all</button>
              <span className="text-gray-300">·</span>
              <button type="button"
                onClick={() => setSelections(prev => {
                  const next = { ...prev }
                  datasetsForType.forEach(ds => { if (next[ds.file]) next[ds.file] = { ...next[ds.file], enabled: false } })
                  return next
                })}
                className="text-xs text-gray-400 hover:underline">Clear</button>
            </div>
          </div>

          {!taskBank?.[taskType] ? (
            <div className="rounded-xl border border-dashed border-gray-300 p-8 text-center text-sm text-gray-400">
              No task bank files found for this task type locally.
            </div>
          ) : (
            datasetsForType.map(ds => (
              <DatasetCard
                key={ds.file}
                dataset={ds}
                sel={selections[ds.file] ?? { enabled: false, limit: 5, difficulty: '' }}
                onChange={next => setSelections(prev => ({ ...prev, [ds.file]: next }))}
              />
            ))
          )}
        </div>
      </form>
    </div>
  )
}
