import { useState, useEffect, useCallback } from 'react'
import { api } from '../api'

const TERMINAL_STATES = new Set(['done', 'error', 'cancelled'])
const MAX_CONSECUTIVE_ERRORS = 4

export function useRunStatus(runId) {
  const [status, setStatus] = useState(null)

  useEffect(() => {
    if (!runId) { setStatus(null); return }

    let active = true
    let consecutiveErrors = 0

    const poll = async () => {
      try {
        const s = await api.runStatus(runId)
        if (!active) return
        consecutiveErrors = 0
        setStatus(s)
        if (!TERMINAL_STATES.has(s.state)) {
          setTimeout(poll, 1500)
        }
      } catch (err) {
        if (!active) return
        // Stop polling on 404 — run doesn't exist
        if (err.message?.includes('404')) {
          setStatus(prev => prev ?? { run_id: runId, state: 'error', error: 'Run not found' })
          return
        }
        consecutiveErrors++
        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          setStatus(prev => ({
            ...(prev ?? { run_id: runId }),
            state: 'error',
            error: 'Lost connection to server — please refresh.',
          }))
          return
        }
        setTimeout(poll, 2000)
      }
    }

    poll()
    return () => { active = false }
  }, [runId])

  const cancelRun = useCallback(async () => {
    if (!runId) return
    try {
      await api.cancelRun(runId)
      setStatus(prev => ({ ...(prev ?? { run_id: runId }), state: 'cancelled' }))
    } catch (err) {
      // If already terminal on server, reflect that
      setStatus(prev => ({ ...(prev ?? { run_id: runId }), state: 'cancelled' }))
    }
  }, [runId])

  return [status, setStatus, cancelRun]
}
