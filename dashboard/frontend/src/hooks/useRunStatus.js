import { useState, useEffect } from 'react'
import { api } from '../api'

export function useRunStatus(runId) {
  const [status, setStatus] = useState(null)

  useEffect(() => {
    if (!runId) { setStatus(null); return }

    let active = true
    const poll = async () => {
      try {
        const s = await api.runStatus(runId)
        if (active) setStatus(s)
        if (active && (s.state === 'running' || s.state === 'loading'))
          setTimeout(poll, 1500)
      } catch {
        if (active) setTimeout(poll, 2000)
      }
    }
    poll()
    return () => { active = false }
  }, [runId])

  return [status, setStatus]
}
