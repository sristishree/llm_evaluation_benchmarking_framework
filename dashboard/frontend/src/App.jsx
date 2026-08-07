import { useState } from 'react'
import { useRunStatus } from './hooks/useRunStatus'
import RunBenchmark from './components/RunBenchmark'
import RunHistory from './components/RunHistory'
import ScoreBreakdown from './components/ScoreBreakdown'
import Heatmap from './components/Heatmap'
import DomainBreakdown from './components/DomainBreakdown'

const TABS = [
  { id: 'run',      label: 'Run Benchmark',   icon: '▶',  component: RunBenchmark },
  { id: 'history',  label: 'Run History',     icon: '📋', component: RunHistory },
  { id: 'scores',   label: 'Score Breakdown', icon: '📈', component: ScoreBreakdown },
  { id: 'heatmap',  label: 'Heatmap',         icon: '🔥', component: Heatmap },
  { id: 'domain',   label: 'Domain Analysis', icon: '🌐', component: DomainBreakdown },
]

const DRY_RUN_TABS = new Set(['history', 'scores', 'heatmap', 'domain'])

// ── Persistent run-progress toast shown when user is away from Run tab ────

function RunToast({ status, onDismiss, onViewHistory }) {
  if (!status) return null

  const { state, progress = 0, total = 0, error } = status
  const pct      = total > 0 ? Math.round((progress / total) * 100) : 0
  const isRunning = state === 'running' || state === 'loading'
  const isDone    = state === 'done'
  const isError   = state === 'error'

  const border = isDone  ? 'bg-green-50 border-green-200'
               : isError ? 'bg-red-50 border-red-200'
               :            'bg-white border-gray-200'

  return (
    <div className={`fixed bottom-6 right-6 z-50 w-80 rounded-xl border shadow-xl p-4 space-y-2.5 ${border}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {isRunning && (
            <span className="shrink-0 inline-block w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
          )}
          <span className="text-sm font-semibold text-gray-800 truncate">
            {isDone  ? '✅ Benchmark complete!'
           : isError ? '❌ Run failed'
           :            'Benchmark running…'}
          </span>
        </div>
        {(isDone || isError) && (
          <button
            onClick={onDismiss}
            className="shrink-0 text-gray-400 hover:text-gray-600 text-xl leading-none"
          >
            ×
          </button>
        )}
      </div>

      {isRunning && (
        <div>
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>
              {state === 'loading' ? 'Loading tasks…' : `Task ${progress} of ${total}`}
            </span>
            {state !== 'loading' && <span>{pct}%</span>}
          </div>
          <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
            <div
              className={`h-2 rounded-full bg-blue-500 transition-all duration-500 ${
                state === 'loading' ? 'animate-pulse w-full' : ''
              }`}
              style={state !== 'loading' ? { width: `${pct}%` } : undefined}
            />
          </div>
        </div>
      )}

      {isDone && (
        <>
          <p className="text-xs text-gray-400 font-mono">{status.run_id?.slice(0, 16)}…</p>
          <button
            onClick={onViewHistory}
            className="w-full text-xs font-semibold text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 rounded-lg py-2 transition-colors"
          >
            View Run History →
          </button>
        </>
      )}

      {isError && error && (
        <p className="text-xs text-red-700 break-all">{error}</p>
      )}
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────────────

export default function App() {
  const [active, setActive]                 = useState('run')
  const [includeDryRuns, setIncludeDryRuns] = useState(false)

  // Run state lives here so it survives tab navigation
  const [activeRunId, setActiveRunId]       = useState(null)
  const [runStatus, setRunStatus]           = useRunStatus(activeRunId)

const { component: ActiveView } = TABS.find(t => t.id === active)
  const showToast = activeRunId !== null && active !== 'run'

  function dismissRun() {
    setActiveRunId(null)
    setRunStatus(null)
  }

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-8 py-4 flex items-center gap-3">
        <span className="text-2xl">📊</span>
        <div className="flex-1">
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
              {/* Pulse dot on Run tab when a run is active elsewhere */}
              {tab.id === 'run' && activeRunId && active !== 'run' &&
                (runStatus?.state === 'running' || runStatus?.state === 'loading') && (
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              )}
            </button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <main className={active === 'history' ? 'px-3 py-3' : 'px-8 py-8 max-w-screen-xl mx-auto'}>
        {/* Dry-run filter bar — only on tabs where it's relevant */}
        {DRY_RUN_TABS.has(active) && (
          <div className="mb-6 flex justify-end">
            <label className="flex items-center gap-2 cursor-pointer select-none bg-white border border-gray-200 rounded-lg px-3 py-2 shadow-sm hover:bg-gray-50 transition-colors">
              <input
                type="checkbox"
                checked={includeDryRuns}
                onChange={e => setIncludeDryRuns(e.target.checked)}
                className="w-4 h-4 accent-blue-500"
              />
              <span className="text-sm text-gray-600">Show dry runs</span>
            </label>
          </div>
        )}

        <ActiveView
          includeDryRuns={includeDryRuns}
          activeRunId={activeRunId}
          setActiveRunId={setActiveRunId}
          runStatus={runStatus}
        />
      </main>

      {/* Persistent run toast — shown when user navigates away from Run tab */}
      {showToast && (
        <RunToast
          status={runStatus}
          onDismiss={dismissRun}
          onViewHistory={() => { setActive('history'); dismissRun() }}
        />
      )}
    </div>
  )
}
