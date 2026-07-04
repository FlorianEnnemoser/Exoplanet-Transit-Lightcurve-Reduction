// App.tsx — guided wizard shell (W-15/16/17, ADR-0011). GPL-3.0-or-later.
import { useEffect, useRef, useState } from 'react'
import { api, type Summary, type Verdict, type WizardState } from './api'
import {
  CheckStep,
  DataStep,
  ExportStep,
  ParamsStep,
  ReviewStep,
  SystemStep,
  paramsProblems,
  systemProblems,
} from './steps'
import StarsStep from './StarsStep'

const STEPS = [
  { id: 'data', label: 'Data' },
  { id: 'check', label: 'Check' },
  { id: 'stars', label: 'Stars' },
  { id: 'params', label: 'Parameters' },
  { id: 'system', label: 'System' },
  { id: 'review', label: 'Review' },
  { id: 'export', label: 'Export' },
] as const

type StepId = (typeof STEPS)[number]['id']

export default function App() {
  const [sid, setSid] = useState<string | null>(null)
  const [state, setState] = useState<WizardState | null>(null)
  const [summary, setSummary] = useState<Summary | null>(null)
  const [verdict, setVerdict] = useState<Verdict | null>(null)
  const [step, setStep] = useState(0)
  const [visited, setVisited] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const booted = useRef(false)

  // session bootstrap: resume from #sid (W-18) or create a fresh one
  useEffect(() => {
    if (booted.current) return
    booted.current = true
    ;(async () => {
      const existing = location.hash.slice(1)
      if (existing) {
        try {
          setState(await api.getConfig(existing))
          setSid(existing)
          return
        } catch {
          /* stale hash — fall through to a new session */
        }
      }
      const s = await api.createSession({})
      location.hash = s.id
      setSid(s.id)
      setState(s.state)
    })().catch((e: Error) => setError(e.message))
  }, [])

  if (!sid || !state) {
    return (
      <main>
        <h1>
          exotransit · <span className="accent">transit reduction</span>
        </h1>
        <p className="muted">{error ?? 'Starting session…'}</p>
      </main>
    )
  }

  // W-16: why the Next action is disabled, in plain language
  const blockedReason = (id: StepId): string | null => {
    switch (id) {
      case 'data':
        return state.paths.lights ? null : 'Set a lights directory or upload light frames.'
      case 'check':
        if (!summary) return 'Frame check has not run yet.'
        if (summary.lights.count === 0) return 'No light frames found.'
        return summary.lights.problems[0] ?? null
      case 'stars':
        if (!state.stars.science) return 'Click the science target on the frame.'
        if (state.stars.calibrators.length < 1) return 'Pick at least one calibrator star.'
        return null
      case 'params':
        return paramsProblems(state)[0] ?? null
      case 'system':
        return systemProblems(state)[0] ?? null
      case 'review':
        if (!verdict) return 'Validation pending…'
        return verdict.valid ? null : verdict.error
      case 'export':
        return null
    }
  }

  const navigate = async (to: number) => {
    try {
      setError(null)
      await api.putConfig(sid, state) // persist every transition (W-18)
      if (STEPS[to].id === 'check') setSummary(await api.getSummary(sid))
      if (STEPS[to].id === 'review' || STEPS[to].id === 'export')
        setVerdict(await api.validate(sid))
      setStep(to)
      setVisited((v) => Math.max(v, to))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const id = STEPS[step].id
  const reason = blockedReason(id)

  return (
    <main>
      <h1>
        exotransit · <span className="accent">transit reduction</span>
      </h1>
      <p className="muted">
        From a directory of FITS frames to a validated pipeline config — no hand-edited text.
      </p>

      <nav className="steps" aria-label="wizard steps">
        {STEPS.map((s, i) => (
          <button
            key={s.id}
            className={i === step ? 'current' : i <= visited ? 'done' : ''}
            disabled={i > visited || i === step}
            onClick={() => void navigate(i)}
          >
            {i + 1} · {s.label}
          </button>
        ))}
      </nav>

      {error && (
        <div className="panel error" role="alert">
          {error}
        </div>
      )}

      <section className="panel">
        {id === 'data' && <DataStep sid={sid} state={state} onChange={setState} />}
        {id === 'check' && (
          <CheckStep summary={summary} onRefresh={() => void api.getSummary(sid).then(setSummary)} />
        )}
        {id === 'stars' && (
          <StarsStep sid={sid} state={state} summary={summary} onChange={setState} />
        )}
        {id === 'params' && <ParamsStep state={state} onChange={setState} />}
        {id === 'system' && <SystemStep state={state} onChange={setState} />}
        {id === 'review' && <ReviewStep state={state} verdict={verdict} />}
        {id === 'export' && <ExportStep sid={sid} state={state} verdict={verdict} />}
      </section>

      <div className="nav">
        <button className="ghost" disabled={step === 0} onClick={() => void navigate(step - 1)}>
          Back
        </button>
        {reason && step < STEPS.length - 1 && <span className="blocked-reason">{reason}</span>}
        {step < STEPS.length - 1 && (
          <button
            className="primary"
            disabled={reason !== null}
            onClick={() => void navigate(step + 1)}
          >
            Next
          </button>
        )}
      </div>
    </main>
  )
}
