import { useState, useMemo } from 'react'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, EmptyState, SectionHeader, Select } from './ui'

const SCORE_FIELDS = [
  { key: 'rouge_l',         label: 'ROUGE-L' },
  { key: 'bert_score',      label: 'BERTScore' },
  { key: 'exact_match',     label: 'Exact Match' },
  { key: 'token_f1',        label: 'Token F1' },
  { key: 'llm_judge_score', label: 'LLM Judge' },
]

const DIFFICULTIES = ['easy', 'medium', 'hard']

function ScoreBadge({ label, value }) {
  if (value == null) return null
  const pct = value * 100
  const color = pct >= 70 ? 'bg-green-100 text-green-700' : pct >= 40 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full ${color}`}>
      {label}: {value.toFixed(3)}
    </span>
  )
}

function OutputBlock({ label, value, highlight }) {
  let display = '(none)'
  if (value != null) {
    try { display = JSON.stringify(JSON.parse(value), null, 2) } catch { display = value }
  }
  return (
    <div className={`px-4 py-3 border-t border-gray-100${highlight ? ' bg-red-50' : ''}`}>
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">{label}</p>
      <pre className="text-xs text-gray-700 whitespace-pre-wrap bg-gray-50 rounded-lg p-3 max-h-40 overflow-y-auto font-mono">
        {display}
      </pre>
    </div>
  )
}

const DIM_LABEL = {
  judge_faithfulness:  'Faithful',
  judge_coverage:      'Coverage',
  judge_conciseness:   'Concise',
  judge_completeness:  'Complete',
  judge_precision:     'Precision',
  judge_accuracy:      'Accuracy',
  judge_justifiability:'Justify',
}

function DimBadge({ dimKey, value }) {
  const pct = value * 100
  const color = pct >= 70 ? 'bg-purple-100 text-purple-700'
              : pct >= 40 ? 'bg-yellow-100 text-yellow-700'
              :              'bg-red-100 text-red-700'
  const label = DIM_LABEL[dimKey] ?? dimKey.replace('judge_', '')
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full ${color}`}>
      {label}: {value.toFixed(2)}
    </span>
  )
}

function ResponseCard({ row }) {
  const scores = SCORE_FIELDS.filter(f => row[f.key] != null)
  const isWrong = row.exact_match != null && row.exact_match < 1.0
  const dims = row.judge_dimensions
    ? Object.entries(row.judge_dimensions).filter(([, v]) => v != null)
    : []

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm flex flex-col min-w-0">
      {/* Card header */}
      <div className="px-4 py-3 border-b border-gray-100">
        <div className="flex items-center gap-2 flex-wrap">
          <p className="font-semibold text-gray-800">{row.provider}</p>
          {row.rubric_overridden === 1 && (
            <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full font-semibold">
              custom rubric
            </span>
          )}
        </div>
        <p className="text-xs text-gray-400">{row.model}</p>
        <p className="text-xs text-gray-400 mt-0.5">
          {Math.round(row.latency_ms)} ms · {(row.total_tokens || 0).toLocaleString()} tokens
        </p>
      </div>

      {/* Overall scores */}
      {scores.length > 0 && (
        <div className="px-4 py-2 border-b border-gray-100 flex flex-wrap gap-1">
          {scores.map(f => <ScoreBadge key={f.key} label={f.label} value={row[f.key]} />)}
        </div>
      )}
      {scores.length === 0 && (
        <div className="px-4 py-2 border-b border-gray-100">
          <span className="text-xs text-gray-400">No scores yet</span>
        </div>
      )}

      {/* Judge dimension scores */}
      {dims.length > 0 && (
        <div className="px-4 py-2 border-b border-gray-100 space-y-1">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Judge dimensions</p>
          <div className="flex flex-wrap gap-1">
            {dims.map(([k, v]) => <DimBadge key={k} dimKey={k} value={v} />)}
          </div>
        </div>
      )}

      {/* Expected answer */}
      {row.expected != null && (
        <OutputBlock label="Expected" value={row.expected} />
      )}

      {/* Model prediction */}
      <OutputBlock label="Model Prediction" value={row.parsed_output} highlight={isWrong} />

      {/* Parse error */}
      {row.parse_error && (
        <div className="px-4 py-2 border-t border-red-100 bg-red-50 rounded-b-xl">
          <p className="text-xs text-red-600"><span className="font-semibold">Parse error:</span> {row.parse_error}</p>
        </div>
      )}
    </div>
  )
}

export default function ResponseViewer() {
  const [filterType, setFilterType] = useState('')
  const [filterDiff, setFilterDiff] = useState('')
  const [filterProvider, setFilterProvider] = useState('')
  const [selectedTaskId, setSelectedTaskId] = useState(null)

  const { data, loading, error } = useApi(
    () => api.results({
      task_type: filterType || undefined,
      difficulty: filterDiff || undefined,
      provider: filterProvider || undefined,
      limit: 500,
    }),
    [filterType, filterDiff, filterProvider],
  )

  const { taskIds, taskRows } = useMemo(() => {
    if (!data?.length) return { taskIds: [], taskRows: [] }
    const taskIds = [...new Set(data.map(r => r.task_id))].sort()
    const activeId = selectedTaskId && taskIds.includes(selectedTaskId)
      ? selectedTaskId
      : taskIds[0]
    const taskRows = data.filter(r => r.task_id === activeId)
    return { taskIds, taskRows, activeId }
  }, [data, selectedTaskId])

  const { data: breakdownData } = useApi(() => api.breakdown())
  const allProviders = useMemo(() => {
    if (!breakdownData?.length) return []
    return [...new Set(breakdownData.map(r => r.provider))].sort()
  }, [breakdownData])
  const allTypes = useMemo(() => {
    if (!breakdownData?.length) return []
    return [...new Set(breakdownData.map(r => r.task_type))].sort()
  }, [breakdownData])

  const activeTaskId = selectedTaskId && taskIds.includes(selectedTaskId) ? selectedTaskId : taskIds[0]
  const meta = taskRows[0]

  return (
    <div className="space-y-6">
      <SectionHeader>Response Viewer</SectionHeader>

      {/* Filters */}
      <div className="flex flex-wrap gap-4">
        <div className="w-44">
          <Select
            label="Task Type"
            value={filterType}
            onChange={v => { setFilterType(v); setSelectedTaskId(null) }}
            options={[{ value: '', label: 'All types' }, ...allTypes.map(t => ({ value: t, label: t }))]}
          />
        </div>
        <div className="w-40">
          <Select
            label="Difficulty"
            value={filterDiff}
            onChange={v => { setFilterDiff(v); setSelectedTaskId(null) }}
            options={[{ value: '', label: 'All' }, ...DIFFICULTIES.map(d => ({ value: d, label: d.charAt(0).toUpperCase() + d.slice(1) }))]}
          />
        </div>
        <div className="w-48">
          <Select
            label="Provider"
            value={filterProvider}
            onChange={v => { setFilterProvider(v); setSelectedTaskId(null) }}
            options={[{ value: '', label: 'All providers' }, ...allProviders.map(p => ({ value: p, label: p }))]}
          />
        </div>
      </div>

      {loading && <Spinner />}
      {error && <ErrorCard message={error} />}

      {!loading && !error && !data?.length && (
        <EmptyState message="No results match these filters." />
      )}

      {!loading && !error && data?.length > 0 && (
        <>
          {/* Task ID picker */}
          <div>
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide block mb-1">
              Task ID ({taskIds.length} tasks)
            </label>
            <select
              value={activeTaskId ?? ''}
              onChange={e => setSelectedTaskId(e.target.value)}
              className="border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 min-w-96"
            >
              {taskIds.map(id => (
                <option key={id} value={id}>{id}</option>
              ))}
            </select>
          </div>

          {/* Task metadata */}
          {meta && (
            <div className="flex flex-wrap gap-2">
              {[
                ['Type', meta.task_type],
                ['Difficulty', meta.difficulty],
                ['Domain', meta.domain],
              ].map(([label, val]) => (
                <span key={label} className="inline-flex items-center gap-1.5 text-xs bg-gray-100 text-gray-600 px-3 py-1 rounded-full">
                  <span className="font-medium">{label}:</span> {val}
                </span>
              ))}
            </div>
          )}

          {/* Side-by-side response cards */}
          {taskRows.length > 0 ? (
            <div
              className="grid gap-4"
              style={{ gridTemplateColumns: `repeat(${Math.min(taskRows.length, 3)}, minmax(0, 1fr))` }}
            >
              {taskRows.map(row => (
                <ResponseCard key={row.id} row={row} />
              ))}
            </div>
          ) : (
            <EmptyState message="Select a task ID above to compare responses." />
          )}
        </>
      )}
    </div>
  )
}
