import { useState, useMemo } from 'react'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { Spinner, ErrorCard, EmptyState, SectionHeader, Select } from './ui'

const DIFFICULTIES = ['easy', 'medium', 'hard']

function scoreToColor(score) {
  if (score == null) return { bg: '#f3f4f6', text: '#9ca3af' }
  // 0 → red (0°), 0.5 → yellow (60°), 1 → green (120°)
  const hue = Math.round(score * 120)
  const lightness = 38 + (1 - score) * 12  // slightly lighter for low scores
  return { bg: `hsl(${hue}, 62%, ${lightness}%)`, text: '#ffffff' }
}

function HeatmapGrid({ data, taskType }) {
  const rows = useMemo(() => {
    const filtered = data.filter(r => r.task_type === taskType)
    const providers = [...new Set(filtered.map(r => `${r.provider} / ${r.model}`))]
    return providers.map(pk => {
      const [provider, model] = pk.split(' / ')
      const row = { provider, model, pk }
      DIFFICULTIES.forEach(d => {
        const match = filtered.find(r => r.provider === provider && r.model === model && r.difficulty === d)
        row[d] = match ? match.avg_score : null
        row[`${d}_n`] = match ? match.n : 0
      })
      return row
    })
  }, [data, taskType])

  if (!rows.length) return <EmptyState message={`No data for ${taskType}.`} />

  return (
    <div className="overflow-x-auto">
      <table className="text-sm">
        <thead>
          <tr>
            <th className="px-4 py-2 text-left text-gray-600 font-semibold w-64">Provider / Model</th>
            {DIFFICULTIES.map(d => (
              <th key={d} className="px-6 py-2 text-center text-gray-600 font-semibold capitalize w-36">{d}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.pk}>
              <td className="px-4 py-2 text-gray-700 font-medium whitespace-nowrap">
                <div>{row.provider}</div>
                <div className="text-xs text-gray-400">{row.model}</div>
              </td>
              {DIFFICULTIES.map(d => {
                const { bg, text } = scoreToColor(row[d])
                return (
                  <td key={d} className="px-2 py-2 text-center">
                    <div
                      className="rounded-lg px-4 py-3 font-semibold"
                      style={{ backgroundColor: bg, color: text }}
                    >
                      {row[d] != null ? row[d].toFixed(3) : 'n/a'}
                      {row[`${d}_n`] > 0 && (
                        <div className="text-xs font-normal mt-0.5" style={{ opacity: 0.85 }}>
                          n={row[`${d}_n`]}
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
    </div>
  )
}

export default function Heatmap() {
  const { data, loading, error } = useApi(() => api.heatmap())
  const [taskType, setTaskType] = useState(null)

  const taskTypes = useMemo(() => {
    if (!data?.length) return []
    return [...new Set(data.map(r => r.task_type))].sort()
  }, [data])

  const activeType = taskType ?? taskTypes[0]

  if (loading) return <Spinner />
  if (error)   return <ErrorCard message={error} />
  if (!data?.length) return <EmptyState message="No heatmap data available." />

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <SectionHeader>Performance Heatmap — Provider × Difficulty</SectionHeader>
          <p className="text-sm text-gray-500 -mt-3">
            Score = first available metric: ROUGE-L → BERTScore → Exact Match → LLM Judge
          </p>
        </div>
        <div className="w-52">
          <Select
            label="Task Type"
            value={activeType ?? ''}
            onChange={setTaskType}
            options={taskTypes.map(t => ({ value: t, label: t.charAt(0).toUpperCase() + t.slice(1) }))}
          />
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-3 text-xs text-gray-500">
        <span>Score scale:</span>
        <div className="flex rounded overflow-hidden h-4 w-48">
          {Array.from({ length: 20 }, (_, i) => {
            const { bg } = scoreToColor(i / 19)
            return <div key={i} className="flex-1" style={{ backgroundColor: bg }} />
          })}
        </div>
        <span>0.0</span>
        <span className="ml-auto">1.0</span>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        {activeType && <HeatmapGrid data={data} taskType={activeType} />}
      </div>
    </div>
  )
}
