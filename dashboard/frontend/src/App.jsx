import { useState } from 'react'
import RunHistory from './components/RunHistory'
import ScoreBreakdown from './components/ScoreBreakdown'
import Heatmap from './components/Heatmap'
import CostVsQuality from './components/CostVsQuality'
import ResponseViewer from './components/ResponseViewer'

const TABS = [
  { id: 'history',  label: 'Run History',      icon: '📋', component: RunHistory },
  { id: 'scores',   label: 'Score Breakdown',   icon: '📈', component: ScoreBreakdown },
  { id: 'heatmap',  label: 'Heatmap',           icon: '🔥', component: Heatmap },
  { id: 'cost',     label: 'Cost vs Quality',   icon: '💰', component: CostVsQuality },
  { id: 'viewer',   label: 'Response Viewer',   icon: '🔍', component: ResponseViewer },
]

export default function App() {
  const [active, setActive] = useState('history')
  const { component: ActiveView } = TABS.find(t => t.id === active)

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-8 py-5 flex items-center gap-3">
        <span className="text-2xl">📊</span>
        <div>
          <h1 className="text-xl font-bold text-gray-900 leading-none">
            LLM Evaluation Benchmark
          </h1>
          <p className="text-sm text-gray-500 mt-0.5">Multi-provider NLP benchmark results</p>
        </div>
      </header>

      {/* Tab nav */}
      <nav className="bg-white border-b border-gray-200 px-8">
        <div className="flex">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActive(tab.id)}
              className={`flex items-center gap-2 px-5 py-4 text-sm font-medium border-b-2 transition-colors ${
                active === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <span>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <main className="px-8 py-8 max-w-screen-xl mx-auto">
        <ActiveView />
      </main>
    </div>
  )
}
