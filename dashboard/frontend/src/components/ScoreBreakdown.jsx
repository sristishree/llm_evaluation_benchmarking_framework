import { useState, useMemo } from 'react'
import {
  BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, EmptyState, SectionHeader, Select } from './ui'

const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4']

const METRICS = [
  { value: 'avg_rouge_1',     label: 'ROUGE-1' },
  { value: 'avg_rouge_2',     label: 'ROUGE-2' },
  { value: 'avg_rouge_l',     label: 'ROUGE-L' },
  { value: 'avg_bert_score',  label: 'BERTScore' },
  { value: 'avg_exact_match', label: 'Exact Match' },
  { value: 'avg_token_f1',    label: 'Token F1' },
]

const TABLE_COLS = [
  { key: 'provider',        label: 'Provider',    render: r => r.provider,       sortable: true },
  { key: 'model',           label: 'Model',       render: r => r.model,          sortable: true },
  { key: 'task_type',       label: 'Task Type',   render: r => r.task_type,      sortable: true },
  { key: 'n',               label: 'n',           render: r => r.n,              sortable: true },
  { key: 'avg_rouge_1',     label: 'ROUGE-1',     render: r => r.avg_rouge_1?.toFixed(3)     ?? '—', sortable: true },
  { key: 'avg_rouge_2',     label: 'ROUGE-2',     render: r => r.avg_rouge_2?.toFixed(3)     ?? '—', sortable: true },
  { key: 'avg_rouge_l',     label: 'ROUGE-L',     render: r => r.avg_rouge_l?.toFixed(3)     ?? '—', sortable: true },
  { key: 'avg_bert_score',  label: 'BERTScore',   render: r => r.avg_bert_score?.toFixed(3)  ?? '—', sortable: true },
  { key: 'avg_exact_match', label: 'Exact Match', render: r => r.avg_exact_match?.toFixed(3) ?? '—', sortable: true },
  { key: 'avg_token_f1',    label: 'Token F1',    render: r => r.avg_token_f1?.toFixed(3)    ?? '—', sortable: true },
  { key: 'avg_latency_ms',  label: 'Avg Latency', render: r => r.avg_latency_ms != null ? `${Math.round(r.avg_latency_ms)} ms` : '—', sortable: true },
]

// ─── Filter bar ────────────────────────────────────────────────────────────────

function FilterBar({ allProviderKeys, selectedProviders, onToggleProvider, allTaskTypes, selectedTasks, onToggleTask, minSamples, onMinSamples, viewMode, onViewMode, metric, onMetric, availableMetrics, providerColors }) {
  return (
    <div className="bg-gray-50 rounded-xl border border-gray-200 p-4 space-y-3">
      <div className="flex flex-wrap items-end gap-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">View</label>
          <div className="flex rounded-lg border border-gray-200 overflow-hidden text-sm">
            {[['overview', 'Overview'], ['per-task', 'Per Task']].map(([mode, label]) => (
              <button
                key={mode}
                onClick={() => onViewMode(mode)}
                className={`px-3 py-2 transition-colors ${viewMode === mode ? 'bg-blue-600 text-white font-medium' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {viewMode === 'overview' ? (
          <div className="w-44">
            <Select
              label="Metric"
              value={metric}
              onChange={onMetric}
              options={[
                { value: '', label: 'All (avg)' },
                ...(availableMetrics.length ? availableMetrics : METRICS),
              ]}
            />
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Tasks</label>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => onToggleTask('__all__')}
                className={`px-2.5 py-1 rounded-full text-xs border transition-colors ${selectedTasks.size === 0 ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-200 hover:border-blue-400'}`}
              >
                All
              </button>
              {allTaskTypes.map(t => (
                <button
                  key={t}
                  onClick={() => onToggleTask(t)}
                  className={`px-2.5 py-1 rounded-full text-xs border transition-colors ${selectedTasks.has(t) ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-200 hover:border-blue-400'}`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Min Samples (n)</label>
          <input
            type="number"
            min={1}
            value={minSamples}
            onChange={e => onMinSamples(Math.max(1, Number(e.target.value) || 1))}
            className="border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white w-24 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

      </div>

      {allProviderKeys.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide mr-1">Providers</span>
          <button
            onClick={() => onToggleProvider('__all__')}
            className={`px-2.5 py-1 rounded-full text-xs border transition-colors ${selectedProviders.size === 0 ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-200 hover:border-blue-400'}`}
          >
            All
          </button>
          {allProviderKeys.map(pk => {
            const active = selectedProviders.has(pk)
            const color  = providerColors[pk]
            return (
              <button
                key={pk}
                onClick={() => onToggleProvider(pk)}
                className="px-2.5 py-1 rounded-full text-xs border transition-colors"
                style={active
                  ? { backgroundColor: color, borderColor: color, color: '#fff' }
                  : { backgroundColor: '#fff', borderColor: '#e5e7eb', color: '#4b5563' }
                }
              >
                {pk}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─── Overview chart (task type × provider) ─────────────────────────────────────

function OverviewChart({ data, providerKeys, effectiveMetric, providerColors, availableMetrics }) {
  const chartData = useMemo(() => {
    const pivot = {}
    data.forEach(row => {
      const pk = `${row.provider} / ${row.model}`
      if (!pivot[row.task_type]) pivot[row.task_type] = { task_type: row.task_type }
      if (effectiveMetric === '') {
        const vals = availableMetrics.map(m => row[m.value]).filter(v => v != null)
        pivot[row.task_type][pk] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
      } else {
        pivot[row.task_type][pk] = row[effectiveMetric] ?? null
      }
    })
    return Object.values(pivot)
  }, [data, effectiveMetric, availableMetrics])

  const label = effectiveMetric === '' ? 'Average across all metrics' : (METRICS.find(m => m.value === effectiveMetric)?.label ?? effectiveMetric)

  if (!chartData.length) return <EmptyState message="No data matches current filters." />

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="task_type" tick={{ fontSize: 13 }} />
          <YAxis domain={[0, 1]} tickFormatter={v => v.toFixed(1)} tick={{ fontSize: 12 }} />
          <Tooltip
            formatter={(val, name) => [val != null ? val.toFixed(3) : '—', name]}
            contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb', fontSize: 13 }}
          />
          <Legend />
          {providerKeys.map(pk => (
            <Bar key={pk} dataKey={pk} fill={providerColors[pk]} radius={[4, 4, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-400 text-center mt-2">{label} · grouped by task type</p>
    </div>
  )
}

// ─── Per-task charts (one card per task type) ───────────────────────────────────

function PerTaskCharts({ data, providerKeys, availableMetrics, providerColors, selectedTasks }) {
  const taskTypes = useMemo(() => {
    const all = [...new Set(data.map(r => r.task_type))].sort()
    return selectedTasks.size > 0 ? all.filter(t => selectedTasks.has(t)) : all
  }, [data, selectedTasks])

  if (!taskTypes.length) return <EmptyState message="No data matches current filters." />

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
      {taskTypes.map(taskType => {
        const taskRows = data.filter(r => r.task_type === taskType)
        const taskMetrics = availableMetrics.filter(m => taskRows.some(r => r[m.value] != null))

        // X-axis: metric name, bars: providers
        const chartData = taskMetrics.map(m => {
          const entry = { metric: m.label }
          taskRows.forEach(row => {
            entry[`${row.provider} / ${row.model}`] = row[m.value] ?? null
          })
          return entry
        })

        return (
          <div key={taskType} className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
            <h3 className="text-sm font-semibold text-gray-700 mb-4 capitalize">{taskType}</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tickFormatter={v => v.toFixed(1)} tick={{ fontSize: 12 }} />
                <Tooltip
                  formatter={(val, name) => [val != null ? val.toFixed(3) : '—', name]}
                  contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb', fontSize: 13 }}
                />
                <Legend />
                {providerKeys.map(pk => (
                  <Bar key={pk} dataKey={pk} fill={providerColors[pk]} radius={[3, 3, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )
      })}
    </div>
  )
}

// ─── Sortable table ─────────────────────────────────────────────────────────────

function SortableTable({ data }) {
  const [sortKey, setSortKey] = useState(null)
  const [sortDir, setSortDir] = useState('desc')

  const sorted = useMemo(() => {
    if (!sortKey) return data
    return [...data].sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      // nulls always last
      if (av == null && bv == null) return 0
      if (av == null) return 1
      if (bv == null) return -1
      if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
      return sortDir === 'asc' ? av - bv : bv - av
    })
  }, [data, sortKey, sortDir])

  function handleSort(key) {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-200">
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="bg-gray-50 border-b border-gray-200">
            {TABLE_COLS.map(col => (
              <th
                key={col.key}
                onClick={col.sortable ? () => handleSort(col.key) : undefined}
                className={`px-4 py-3 font-semibold text-gray-600 whitespace-nowrap select-none ${col.sortable ? 'cursor-pointer hover:bg-gray-100 transition-colors' : ''}`}
              >
                <span className="inline-flex items-center gap-1">
                  {col.label}
                  {col.sortable && (
                    <span className={`text-xs ${sortKey === col.key ? 'text-blue-500' : 'text-gray-300'}`}>
                      {sortKey === col.key ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}
                    </span>
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={i} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
              {TABLE_COLS.map(col => (
                <td key={col.key} className="px-4 py-2.5 whitespace-nowrap">{col.render(r)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Score scatter (individual results per provider) ───────────────────────────

const SCATTER_METRICS = [
  { value: 'rouge_l',     label: 'ROUGE-L' },
  { value: 'exact_match', label: 'Exact Match' },
  { value: 'token_f1',    label: 'Token F1' },
  { value: 'bert_score',  label: 'BERTScore' },
  { value: 'rouge_1',     label: 'ROUGE-1' },
  { value: 'rouge_2',     label: 'ROUGE-2' },
]

function ScoreScatter({ includeDryRuns, selectedProviders, providerColors, allProviderKeys, scatterMetric, onScatterMetric }) {
  const { data, loading, error } = useApi(() => api.distribution(includeDryRuns), [includeDryRuns])

  const { scatterSeries, xDomain, xTicks, availableMetrics, effectiveMetric } = useMemo(() => {
    if (!data?.length) return { scatterSeries: [], xDomain: [0, 1], xTicks: [], availableMetrics: [], effectiveMetric: scatterMetric }

    const availableMetrics = SCATTER_METRICS.filter(m => data.some(r => r[m.value] != null))
    const effectiveMetric  = availableMetrics.find(m => m.value === scatterMetric)
      ? scatterMetric : availableMetrics[0]?.value ?? scatterMetric

    const activeProviders = allProviderKeys.filter(pk =>
      selectedProviders.size === 0 || selectedProviders.has(pk)
    )

    const xTicks = activeProviders.map((pk, i) => ({ pos: i, label: pk }))

    const scatterSeries = activeProviders.map((pk, provIdx) => ({
      pk,
      color: providerColors[pk] ?? COLORS[provIdx % COLORS.length],
      points: data
        .filter(r => `${r.provider} / ${r.model}` === pk && r[effectiveMetric] != null)
        .map((r, i) => ({
          x: provIdx + ((i * 7919 + 13) % 97) / 97 * 0.5 - 0.25,
          y: r[effectiveMetric],
          task: r.task_type,
          pk,
        })),
    }))

    return {
      scatterSeries,
      xDomain: [-0.5, activeProviders.length - 0.5],
      xTicks,
      availableMetrics,
      effectiveMetric,
    }
  }, [data, scatterMetric, selectedProviders, allProviderKeys, providerColors])

  if (loading) return <Spinner />
  if (error)   return <ErrorCard message={error} />
  if (!data?.length) return null

  const metricLabel = availableMetrics.find(m => m.value === effectiveMetric)?.label ?? effectiveMetric

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <SectionHeader>Individual Score Distribution</SectionHeader>
        <div className="w-44">
          <Select
            label="Metric"
            value={effectiveMetric}
            onChange={onScatterMetric}
            options={availableMetrics.length ? availableMetrics : SCATTER_METRICS}
          />
        </div>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              type="number"
              dataKey="x"
              domain={xDomain}
              ticks={xTicks.map(t => t.pos)}
              tickFormatter={pos => xTicks.find(t => t.pos === pos)?.label ?? ''}
              tick={{ fontSize: 11, angle: -20, textAnchor: 'end' }}
              interval={0}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0, 1]}
              tickFormatter={v => v.toFixed(1)}
              tick={{ fontSize: 12 }}
            />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={({ payload }) => {
                if (!payload?.length) return null
                const { y, task, pk } = payload[0].payload
                return (
                  <div className="bg-white border border-gray-200 rounded-lg px-3 py-2 text-xs shadow-sm">
                    <p className="font-semibold text-gray-800 mb-0.5">{pk}</p>
                    <p className="text-gray-600">{task}</p>
                    <p className="text-gray-900 mt-1">{metricLabel}: <span className="font-medium">{y.toFixed(3)}</span></p>
                  </div>
                )
              }}
            />
            <Legend />
            {scatterSeries.map(({ pk, color, points }) => (
              <Scatter key={pk} name={pk} data={points} fill={color} opacity={0.75} r={5} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-400 text-center mt-1">
          {metricLabel} · each dot is one scored result — hover for details
        </p>
      </div>
    </div>
  )
}

// ─── Root ───────────────────────────────────────────────────────────────────────

export default function ScoreBreakdown({ includeDryRuns = false }) {
  const { data, loading, error } = useApi(() => api.breakdown(includeDryRuns), [includeDryRuns])

  const [metric,        setMetric]        = useState('avg_rouge_l')
  const [scatterMetric, setScatterMetric] = useState('rouge_l')
  const [viewMode,          setViewMode]          = useState('overview')
  const [selectedProviders, setSelectedProviders] = useState(new Set())
  const [selectedTasks,     setSelectedTasks]     = useState(new Set())
  const [minSamples,        setMinSamples]        = useState(1)

  const { filteredData, allProviderKeys, allTaskTypes, providerKeys, availableMetrics, effectiveMetric, providerColors } = useMemo(() => {
    if (!data?.length) return {
      filteredData: [], allProviderKeys: [], allTaskTypes: [], providerKeys: [],
      availableMetrics: [], effectiveMetric: metric, providerColors: {},
    }

    const availableMetrics = METRICS.filter(m => data.some(r => r[m.value] != null))
    const effectiveMetric  = metric === '' ? '' : (
      availableMetrics.find(m => m.value === metric) ? metric : availableMetrics[0]?.value ?? metric
    )

    const allProviderKeys = [...new Set(data.map(r => `${r.provider} / ${r.model}`))].sort()
    const allTaskTypes    = [...new Set(data.map(r => r.task_type))].sort()
    const providerColors  = Object.fromEntries(allProviderKeys.map((pk, i) => [pk, COLORS[i % COLORS.length]]))

    const filteredData = data.filter(r => {
      const pk = `${r.provider} / ${r.model}`
      if (selectedProviders.size > 0 && !selectedProviders.has(pk)) return false
      if (r.n < minSamples) return false
      return true
    })

    const providerKeys = selectedProviders.size > 0
      ? allProviderKeys.filter(pk => selectedProviders.has(pk))
      : allProviderKeys

    return { filteredData, allProviderKeys, allTaskTypes, providerKeys, availableMetrics, effectiveMetric, providerColors }
  }, [data, metric, selectedProviders, minSamples])

  function handleToggleProvider(pk) {
    if (pk === '__all__') { setSelectedProviders(new Set()); return }
    setSelectedProviders(prev => {
      const next = new Set(prev)
      next.has(pk) ? next.delete(pk) : next.add(pk)
      return next
    })
  }

  function handleToggleTask(t) {
    if (t === '__all__') { setSelectedTasks(new Set()); return }
    setSelectedTasks(prev => {
      const next = new Set(prev)
      next.has(t) ? next.delete(t) : next.add(t)
      return next
    })
  }

  if (loading) return <Spinner />
  if (error)   return <ErrorCard message={error} />
  if (!data?.length) return <EmptyState message="No scored results yet." />

  return (
    <div className="space-y-6">
      <SectionHeader>Score Breakdown</SectionHeader>

      <FilterBar
        allProviderKeys={allProviderKeys}
        selectedProviders={selectedProviders}
        onToggleProvider={handleToggleProvider}
        allTaskTypes={allTaskTypes}
        selectedTasks={selectedTasks}
        onToggleTask={handleToggleTask}
        minSamples={minSamples}
        onMinSamples={setMinSamples}
        viewMode={viewMode}
        onViewMode={setViewMode}
        metric={effectiveMetric}
        onMetric={setMetric}
        availableMetrics={availableMetrics}
        providerColors={providerColors}
      />

      {viewMode === 'overview' ? (
        <OverviewChart
          data={filteredData}
          providerKeys={providerKeys}
          effectiveMetric={effectiveMetric}
          providerColors={providerColors}
          availableMetrics={availableMetrics}
        />
      ) : (
        <PerTaskCharts
          data={filteredData}
          providerKeys={providerKeys}
          availableMetrics={availableMetrics}
          providerColors={providerColors}
          selectedTasks={selectedTasks}
        />
      )}

      <SortableTable data={filteredData} />

      <ScoreScatter
        includeDryRuns={includeDryRuns}
        selectedProviders={selectedProviders}
        providerColors={providerColors}
        allProviderKeys={allProviderKeys}
        scatterMetric={scatterMetric}
        onScatterMetric={setScatterMetric}
      />
    </div>
  )
}
