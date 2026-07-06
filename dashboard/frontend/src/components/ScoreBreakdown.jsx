import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, EmptyState, SectionHeader, Select } from './ui'

const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4']

const METRICS = [
  { value: 'avg_rouge_l',     label: 'ROUGE-L' },
  { value: 'avg_bert_score',  label: 'BERTScore' },
  { value: 'avg_exact_match', label: 'Exact Match' },
  { value: 'avg_llm_judge',   label: 'LLM Judge Score' },
]

export default function ScoreBreakdown() {
  const { data, loading, error } = useApi(() => api.breakdown())
  const [metric, setMetric] = useState('avg_rouge_l')

  const { chartData, providerKeys, availableMetrics } = useMemo(() => {
    if (!data?.length) return { chartData: [], providerKeys: [], availableMetrics: [] }

    // Pivot: task_type → { "provider/model": score, ... }
    const pivot = {}
    data.forEach(row => {
      const pk = `${row.provider} / ${row.model}`
      if (!pivot[row.task_type]) pivot[row.task_type] = { task_type: row.task_type }
      pivot[row.task_type][pk] = row[metric] ?? null
    })

    const providerKeys = [...new Set(data.map(r => `${r.provider} / ${r.model}`))]
    const availableMetrics = METRICS.filter(m => data.some(r => r[m.value] != null))

    return {
      chartData: Object.values(pivot),
      providerKeys,
      availableMetrics,
    }
  }, [data, metric])

  if (loading) return <Spinner />
  if (error)   return <ErrorCard message={error} />
  if (!data?.length) return <EmptyState message="No scored results yet." />

  const activeMetricLabel = METRICS.find(m => m.value === metric)?.label ?? metric

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <SectionHeader>Score Breakdown by Provider & Task Type</SectionHeader>
        <div className="w-56">
          <Select
            label="Metric"
            value={metric}
            onChange={setMetric}
            options={availableMetrics.length ? availableMetrics : METRICS}
          />
        </div>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="task_type" tick={{ fontSize: 13 }} />
            <YAxis domain={[0, 1]} tickFormatter={v => v.toFixed(1)} tick={{ fontSize: 12 }} />
            <Tooltip
              formatter={(val, name) => [val != null ? val.toFixed(3) : '—', name]}
              contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb', fontSize: 13 }}
            />
            <Legend />
            {providerKeys.map((pk, i) => (
              <Bar key={pk} dataKey={pk} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-400 text-center mt-2">
          {activeMetricLabel} · grouped by task type
        </p>
      </div>

      {/* Raw table */}
      <div className="overflow-x-auto rounded-xl border border-gray-200">
        <table className="w-full text-sm text-left">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              {['Provider', 'Model', 'Task Type', 'n', 'ROUGE-L', 'BERTScore', 'Exact Match', 'LLM Judge', 'Cost (USD)', 'Avg Latency'].map(h => (
                <th key={h} className="px-4 py-3 font-semibold text-gray-600 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((r, i) => (
              <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                <td className="px-4 py-2.5">{r.provider}</td>
                <td className="px-4 py-2.5 text-gray-500">{r.model}</td>
                <td className="px-4 py-2.5">{r.task_type}</td>
                <td className="px-4 py-2.5">{r.n}</td>
                <td className="px-4 py-2.5">{r.avg_rouge_l?.toFixed(3) ?? '—'}</td>
                <td className="px-4 py-2.5">{r.avg_bert_score?.toFixed(3) ?? '—'}</td>
                <td className="px-4 py-2.5">{r.avg_exact_match?.toFixed(3) ?? '—'}</td>
                <td className="px-4 py-2.5">{r.avg_llm_judge?.toFixed(3) ?? '—'}</td>
                <td className="px-4 py-2.5">{r.total_cost_usd != null ? `$${r.total_cost_usd.toFixed(4)}` : '—'}</td>
                <td className="px-4 py-2.5">{r.avg_latency_ms != null ? `${Math.round(r.avg_latency_ms)} ms` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
