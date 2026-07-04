// PreviewStep.tsx — "7 · Preview": tracked target + differential lightcurve (297, W-12/W-22).
// GPL-3.0-or-later.
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type PreviewJob, type Summary, type WizardState } from './api'
import FrameViewer, { type Mark } from './FrameViewer'

const SCIENCE_COLOR = '#fc3d21'
const CAL_COLOR = '#5b8def'

// differential ensemble flux vs frame — normalised to the median so the baseline
// sits near 1.0 and the transit dip is obvious. Inline SVG, no chart dependency.
function LightCurve(props: { ensemble: (number | null)[] }) {
  const pts = props.ensemble
    .map((f, i) => ({ i, f }))
    .filter((p): p is { i: number; f: number } => p.f !== null && Number.isFinite(p.f))
  if (pts.length < 2) return <p className="muted">Not enough measured frames to plot.</p>
  const med = [...pts.map((p) => p.f)].sort((a, b) => a - b)[Math.floor(pts.length / 2)] || 1
  const ys = pts.map((p) => p.f / med)
  const W = 720
  const H = 240
  const n = props.ensemble.length
  const lo = Math.min(...ys)
  const hi = Math.max(...ys)
  const pad = (hi - lo) * 0.1 || 0.02
  const x = (i: number) => (i / Math.max(n - 1, 1)) * W
  const y = (v: number) => H - ((v - (lo - pad)) / (hi - lo + 2 * pad)) * H
  const y1 = y(1)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="lc-svg" role="img" aria-label="differential light curve">
      <line x1={0} y1={y1} x2={W} y2={y1} stroke="var(--border)" strokeDasharray="4 3" />
      {pts.map((p, k) => (
        <circle key={p.i} cx={x(p.i)} cy={y(ys[k])} r={2} fill={SCIENCE_COLOR} />
      ))}
    </svg>
  )
}

export default function PreviewStep(props: { sid: string; state: WizardState; summary: Summary | null }) {
  const { sid, state, summary } = props
  const [job, setJob] = useState<PreviewJob | null>(null)
  const [index, setIndex] = useState(0)
  const started = useRef(false)

  useEffect(() => {
    let stop = false
    const poll = async () => {
      try {
        const j = await api.getPreview(sid)
        if (!stop) setJob(j)
        if (!stop && j.status === 'running') setTimeout(poll, 1000)
      } catch {
        /* transient — retry on the next tick */
        if (!stop) setTimeout(poll, 1500)
      }
    }
    if (!started.current) {
      started.current = true
      api.startPreview(sid).then((j) => !stop && setJob(j)).catch(() => {})
    }
    poll()
    return () => {
      stop = true
    }
  }, [sid])

  const result = job?.result ?? null
  const frames = summary?.lights.frames ?? []
  const dims = summary?.lights.dims ?? null

  // science + calibrators, all offset by the per-frame tracking shift (dx=row, dy=col)
  const marksFor = useCallback(
    (i: number): Mark[] => {
      if (!result) return []
      const [dx, dy] = result.shifts[i] ?? [0, 0]
      const region = { aperture: state.photometry.aperture_radius, annulus: [state.photometry.annulus_inner, state.photometry.annulus_outer] as [number, number] }
      return [
        { row: result.science.x + dx, col: result.science.y + dy, color: SCIENCE_COLOR, label: result.science.name, ...region },
        ...state.stars.calibrators.map((s) => ({ row: s.x + dx, col: s.y + dy, color: CAL_COLOR, label: s.name, ...region })),
      ]
    },
    [result, state.photometry, state.stars.calibrators],
  )

  return (
    <div>
      <h2>Preview the reduction</h2>
      {(!job || job.status === 'running') && (
        <div>
          <p className="muted">{job?.stage ? `Reducing — ${job.stage}…` : 'Starting reduction…'}</p>
          <div className="progress">
            <div className="progress-bar" style={{ width: `${Math.round((job?.progress ?? 0) * 100)}%` }} />
          </div>
        </div>
      )}
      {job?.status === 'error' && <p className="error">⚠ Preview failed: {job.error}</p>}
      {job?.status === 'done' && result && (
        <div>
          <p className="muted">
            Science target tracked across {frames.length} frames; differential flux below (normalised to the baseline median).
          </p>
          <FrameViewer sid={sid} frames={frames} dims={dims} marksFor={marksFor} onIndexChange={setIndex} />
          <h3 style={{ marginTop: '1rem' }}>Differential light curve</h3>
          <LightCurve ensemble={result.ensemble} />
          <p className="muted">
            Frame {index + 1}: {result.quality[index] ?? '—'}
          </p>
        </div>
      )}
    </div>
  )
}
