import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ErrorBar,
} from 'recharts'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, EmptyState, SectionHeader, Select } from './ui'

const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4']

const DOMAIN_LABELS = { '__review__': 'General' }
const domainLabel = d => DOMAIN_LABELS[d] ?? d

const TASK_TYPES = [
  { value: '',               label: 'All task types' },
  { value: 'classification', label: 'Classification' },
  { value: 'extraction',     label: 'Extraction (NER)' },
  { value: 'qa',             label: 'Q&A' },
  { value: 'summarization',  label: 'Summarization' },
]

// Same color scale as the difficulty heatmap
function scoreToColor(score) {
  if (score == null) return { bg: '#f3f4f6', text: '#9ca3af' }
  const hue       = Math.round(score * 120)
  const lightness = 38 + (1 - score) * 12
  return { bg: `hsl(${hue}, 62%, ${lightness}%)`, text: '#ffffff' }
}

// ─── Heatmap grid: rows=domain, cols=provider ────────────────────────────────

function DomainHeatmapGrid({ data, taskType }) {
  const { domains, providerKeys, cellMap } = useMemo(() => {
    const filtered = taskType ? data.filter(r => r.task_type === taskType) : data
    const domains      = [...new Set(filtered.map(r => domainLabel(r.domain)))].sort()
    const providerKeys = [...new Set(filtered.map(r => `${r.provider} / ${r.model}`))].sort()
    const cellMap = {}
    filtered.forEach(r => {
      const pk  = `${r.provider} / ${r.model}`
      const key = `${domainLabel(r.domain)}__${pk}`
      if (!cellMap[key] || r.n > cellMap[key].n) {
        cellMap[key] = { score: r.avg_score, std: r.std_score, n: r.n }
      }
    })
    return { domains, providerKeys, cellMap }
  }, [data, taskType])

  if (!domains.length) return <EmptyState message="No data for selected filters." />

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 overflow-x-auto">
      <table className="text-sm border-separate border-spacing-1">
        <thead>
          <tr>
            <th className="px-3 py-2 text-left text-gray-500 font-semibold text-xs uppercase tracking-wide min-w-28">
              Domain
            </th>
            {providerKeys.map(pk => (
              <th key={pk} className="px-2 py-2 text-center text-gray-600 font-semibold min-w-32">
                <div className="text-xs leading-tight">{pk.split(' / ').join('\n')}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {domains.map(domain => (
            <tr key={domain}>
              <td className="px-3 py-1 text-gray-700 font-medium capitalize whitespace-nowrap">
                {domain}
              </td>
              {providerKeys.map(pk => {
                const cell = cellMap[`${domain}__${pk}`]
                const { bg, text } = scoreToColor(cell?.score ?? null)
                return (
                  <td key={pk} className="px-1 py-1">
                    <div
                      className="rounded-lg px-3 py-2.5 text-center font-semibold leading-tight"
                      style={{ backgroundColor: bg, color: text, minWidth: '5rem' }}
                      title={cell ? `n=${cell.n}  ±${cell.std?.toFixed(3) ?? '?'}` : 'no data'}
                    >
                      {cell?.score != null ? cell.score.toFixed(3) : 'n/a'}
                      {cell?.n != null && (
                        <div className="text-xs font-normal mt-0.5" style={{ opacity: 0.85 }}>
                          n={cell.n}
                        </div>
                      )}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Color legend */}
      <div className="flex items-center gap-3 text-xs text-gray-500 mt-4">
        <span>Score:</span>
        <div className="flex rounded overflow-hidden h-3 w-40">
          {Array.from({ length: 20 }, (_, i) => {
            const { bg } = scoreToColor(i / 19)
            return <div key={i} className="flex-1" style={{ backgroundColor: bg }} />
          })}
        </div>
        <span>0.0 → 1.0</span>
        <span className="ml-2 text-gray-400">· hover cell for n and ±σ</span>
      </div>
    </div>
  )
}

// ─── Bar chart: domains on X, grouped bars per provider ─────────────────────

function DomainBarChart({ chartData, providerKeys, providerColors, showErrorBars, taskType }) {
  if (!chartData.length) return <EmptyState message="No data for selected filters." />
  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 64 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="domain"
            tick={{ fontSize: 12 }}
            angle={-35}
            textAnchor="end"
            interval={0}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={v => v.toFixed(1)}
            tick={{ fontSize: 12 }}
            label={{ value: 'Avg Score', angle: -90, position: 'insideLeft', fontSize: 13, offset: 10 }}
          />
          <Tooltip
            formatter={(val, name) => [val != null ? val.toFixed(3) : '—', name]}
            contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb', fontSize: 13 }}
          />
          <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: 12 }} />
          {providerKeys.map(pk => (
            <Bar key={pk} dataKey={pk} fill={providerColors[pk]} radius={[3, 3, 0, 0]}>
              {showErrorBars && (
                <ErrorBar dataKey={`${pk}__std`} width={4} strokeWidth={1.5} stroke={providerColors[pk]} />
              )}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-400 text-center mt-1">
        {taskType || 'All task types'} · grouped by domain
      </p>
    </div>
  )
}

// ─── Root ────────────────────────────────────────────────────────────────────

export default function DomainBreakdown({ includeDryRuns = false }) {
  const { data, loading, error } = useApi(() => api.byDomain(includeDryRuns), [includeDryRuns])
  const [taskType,      setTaskType]      = useState('')
  const [view,          setView]          = useState('heatmap')   // 'heatmap' | 'bar'
  const [showErrorBars, setShowErrorBars] = useState(false)

  const { chartData, providerKeys, providerColors } = useMemo(() => {
    if (!data?.length) return { chartData: [], providerKeys: [], providerColors: {}, domains: [] }

    const filtered     = taskType ? data.filter(r => r.task_type === taskType) : data
    const providerKeys = [...new Set(filtered.map(r => `${r.provider} / ${r.model}`))].sort()
    const providerColors = Object.fromEntries(providerKeys.map((pk, i) => [pk, COLORS[i % COLORS.length]]))
    const domains      = [...new Set(filtered.map(r => r.domain))].sort()

    // Pivot: one row per domain, one key per provider
    const pivot = {}
    filtered.forEach(row => {
      const pk    = `${row.provider} / ${row.model}`
      const label = domainLabel(row.domain)
      if (!pivot[label]) pivot[label] = { domain: label }
      pivot[label][pk]           = row.avg_score ?? null
      pivot[label][`${pk}__std`] = row.std_score ?? null
      pivot[label][`${pk}__n`]   = row.n
    })

    return {
      chartData: domains.map(d => pivot[d]).filter(Boolean),
      providerKeys,
      providerColors,
    }
  }, [data, taskType])

  if (loading) return <Spinner />
  if (error)   return <ErrorCard message={error} />
  if (!data?.length) return <EmptyState message="No domain data yet. Run a benchmark first." />

  const views = [
    { id: 'heatmap', label: 'Heatmap Grid' },
    { id: 'bar',     label: 'Bar Chart' },
  ]

  return (
    <div className="space-y-6">
      <div>
        <SectionHeader>Domain Performance Breakdown</SectionHeader>
        <p className="text-sm text-gray-500 -mt-3">
          Score = first available metric per task type: ROUGE-L → Token F1 → Exact Match → BERTScore
        </p>
      </div>

      {/* Filter / view controls */}
      <div className="bg-gray-50 rounded-xl border border-gray-200 p-4 flex flex-wrap items-end gap-4">
        <div className="w-52">
          <Select label="Task Type" value={taskType} onChange={setTaskType} options={TASK_TYPES} />
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">View</label>
          <div className="flex rounded-lg border border-gray-200 overflow-hidden text-sm">
            {views.map(v => (
              <button
                key={v.id}
                onClick={() => setView(v.id)}
                className={`px-3 py-2 transition-colors ${view === v.id ? 'bg-blue-600 text-white font-medium' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
              >
                {v.label}
              </button>
            ))}
          </div>
        </div>

        {view === 'bar' && (
          <label className="flex items-center gap-2 cursor-pointer select-none text-sm text-gray-600 pb-1">
            <input
              type="checkbox"
              checked={showErrorBars}
              onChange={e => setShowErrorBars(e.target.checked)}
              className="w-4 h-4 accent-blue-500"
            />
            Show ±1σ error bars
          </label>
        )}
      </div>

      {view === 'heatmap' && (
        <DomainHeatmapGrid data={data} taskType={taskType} />
      )}

      {view === 'bar' && (
        <DomainBarChart
          chartData={chartData}
          providerKeys={providerKeys}
          providerColors={providerColors}
          showErrorBars={showErrorBars}
          taskType={taskType}
        />
      )}

    </div>
  )
}
