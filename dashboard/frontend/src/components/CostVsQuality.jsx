import { useMemo } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, EmptyState, SectionHeader } from './ui'

const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4']

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-md p-3 text-sm">
      <p className="font-semibold text-gray-800 mb-1">{d.provider} / {d.model}</p>
      <p className="text-gray-600">Task type: <span className="font-medium">{d.task_type}</span></p>
      <p className="text-gray-600">Cost: <span className="font-medium">${(d.total_cost_usd ?? 0).toFixed(4)}</span></p>
      <p className="text-gray-600">Score: <span className="font-medium">{(d.avg_score ?? 0).toFixed(3)}</span></p>
      <p className="text-gray-600">Tasks: <span className="font-medium">{d.n}</span></p>
      <p className="text-xs text-gray-400 mt-1">Run: {d.run_id?.slice(0, 8)}…</p>
    </div>
  )
}

export default function CostVsQuality({ includeDryRuns = false }) {
  const { data, loading, error } = useApi(() => api.costQuality(includeDryRuns), [includeDryRuns])

  const { series, providers } = useMemo(() => {
    if (!data?.length) return { series: [], providers: [] }
    const providers = [...new Set(data.map(r => r.provider))]
    const series = providers.map(p => ({
      provider: p,
      points: data.filter(r => r.provider === p),
    }))
    return { series, providers }
  }, [data])

  if (loading) return <Spinner />
  if (error)   return <ErrorCard message={error} />
  if (!data?.length) return <EmptyState message="No cost data found. Ensure estimated_cost_usd is populated on ScoredResult objects." />

  return (
    <div className="space-y-6">
      <div>
        <SectionHeader>Cost vs Quality</SectionHeader>
        <p className="text-sm text-gray-500 -mt-3">
          Each bubble is one benchmark run. Bubble size scales with task count.
        </p>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <ResponsiveContainer width="100%" height={420}>
          <ScatterChart margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              type="number"
              dataKey="total_cost_usd"
              name="Cost (USD)"
              label={{ value: 'Total Cost (USD)', position: 'insideBottom', offset: -12, fontSize: 13 }}
              tickFormatter={v => `$${v.toFixed(3)}`}
              tick={{ fontSize: 12 }}
            />
            <YAxis
              type="number"
              dataKey="avg_score"
              name="Score"
              domain={[0, 1]}
              label={{ value: 'Average Score', angle: -90, position: 'insideLeft', fontSize: 13 }}
              tickFormatter={v => v.toFixed(1)}
              tick={{ fontSize: 12 }}
            />
            <ZAxis type="number" dataKey="n" range={[60, 500]} name="Tasks" />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            {series.map((s, i) => (
              <Scatter
                key={s.provider}
                name={s.provider}
                data={s.points}
                fill={COLORS[i % COLORS.length]}
                fillOpacity={0.8}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Data table */}
      <div className="overflow-x-auto rounded-xl border border-gray-200">
        <table className="w-full text-sm text-left">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              {['Run ID', 'Provider', 'Model', 'Task Type', 'Tasks', 'Cost (USD)', 'Avg Score'].map(h => (
                <th key={h} className="px-4 py-3 font-semibold text-gray-600 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((r, i) => (
              <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                <td className="px-4 py-2.5"><code className="text-xs bg-gray-100 px-2 py-0.5 rounded">{r.run_id?.slice(0, 8)}…</code></td>
                <td className="px-4 py-2.5">{r.provider}</td>
                <td className="px-4 py-2.5 text-gray-500">{r.model}</td>
                <td className="px-4 py-2.5">{r.task_type}</td>
                <td className="px-4 py-2.5">{r.n}</td>
                <td className="px-4 py-2.5">{r.total_cost_usd != null ? `$${r.total_cost_usd.toFixed(4)}` : '—'}</td>
                <td className="px-4 py-2.5">{r.avg_score != null ? r.avg_score.toFixed(3) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
