// StarsStep.tsx — pick stars on the frame, tune photometry regions, curve of growth.
// (W-4/5/14, 294, 295). GPL-3.0-or-later.
import { useCallback, useEffect, useState } from 'react'
import { api, type GrowthCurve, type Star, type Summary, type WizardState } from './api'
import FrameViewer, { type Mark } from './FrameViewer'
import { Num } from './steps'

const SCIENCE_COLOR = '#fc3d21' // --nasa-red
const CAL_COLOR = '#5b8def'

// small inline SVG curve of growth: enclosed flux vs aperture diameter, with a
// vertical guide at the current aperture diameter. No chart dependency (295).
function GrowthPlot(props: { curve: GrowthCurve; color: string }) {
  const { curve, color } = props
  const pts = curve.diameter
    .map((d, i) => ({ d, f: curve.flux[i] }))
    .filter((p): p is { d: number; f: number } => p.f !== null && Number.isFinite(p.f))
  if (pts.length < 2) return <p className="muted">no profile (star near edge)</p>
  const W = 200
  const H = 90
  const dMax = pts[pts.length - 1].d
  const fMax = Math.max(...pts.map((p) => p.f), 1e-9)
  const fMin = Math.min(...pts.map((p) => p.f), 0)
  const x = (d: number) => (d / dMax) * W
  const y = (f: number) => H - ((f - fMin) / (fMax - fMin)) * H
  const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.d).toFixed(1)},${y(p.f).toFixed(1)}`).join(' ')
  const apX = x(2 * curve.aperture_radius)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="growth-svg" role="img" aria-label={`${curve.name} curve of growth`}>
      <line x1={apX} y1={0} x2={apX} y2={H} stroke="var(--muted)" strokeDasharray="3 2" />
      <path d={path} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  )
}

export default function StarsStep(props: {
  sid: string
  state: WizardState
  summary: Summary | null
  onChange: (s: WizardState) => void
}) {
  const { sid, state, summary, onChange } = props
  const [index, setIndex] = useState(0)
  const [curves, setCurves] = useState<GrowthCurve[]>([])

  const frames = summary?.lights.frames ?? []
  const dims = summary?.lights.dims ?? null
  const p = state.photometry

  const marksFor = useCallback((): Mark[] => {
    const region = { aperture: p.aperture_radius, annulus: [p.annulus_inner, p.annulus_outer] as [number, number], crop: state.stars.crop_half_width }
    return [
      ...(state.stars.science
        ? [{ row: state.stars.science.x, col: state.stars.science.y, color: SCIENCE_COLOR, label: state.stars.science.name, ...region }]
        : []),
      ...state.stars.calibrators.map((s) => ({ row: s.x, col: s.y, color: CAL_COLOR, label: s.name, ...region })),
    ]
  }, [state.stars, p.aperture_radius, p.annulus_inner, p.annulus_outer])

  const nStars = (state.stars.science ? 1 : 0) + state.stars.calibrators.length

  // curve of growth for the current frame; debounced so slider drags don't spam (295)
  useEffect(() => {
    if (nStars === 0 || frames.length === 0) {
      setCurves([])
      return
    }
    const t = setTimeout(() => {
      api.frameGrowth(sid, index).then((r) => setCurves(r.stars)).catch(() => setCurves([]))
    }, 250)
    return () => clearTimeout(t)
  }, [sid, index, nStars, p.aperture_radius, p.annulus_inner, p.annulus_outer, state.stars.crop_half_width, frames.length])

  const pick = (row: number, col: number) => {
    if (!state.stars.science) {
      const name = state.observation.target.replace(/\s+/g, '') || 'Science'
      onChange({ ...state, stars: { ...state.stars, science: { name, x: row, y: col } } })
    } else {
      const cal: Star = { name: `Calibrator_${state.stars.calibrators.length + 1}`, x: row, y: col }
      onChange({ ...state, stars: { ...state.stars, calibrators: [...state.stars.calibrators, cal] } })
    }
  }

  const setPhot = (k: keyof WizardState['photometry'], v: number) =>
    onChange({ ...state, photometry: { ...state.photometry, [k]: v } })

  const removeCal = (i: number) =>
    onChange({ ...state, stars: { ...state.stars, calibrators: state.stars.calibrators.filter((_, j) => j !== i) } })

  const rename = (which: 'science' | number, name: string) => {
    if (which === 'science' && state.stars.science)
      onChange({ ...state, stars: { ...state.stars, science: { ...state.stars.science, name } } })
    else if (typeof which === 'number')
      onChange({
        ...state,
        stars: { ...state.stars, calibrators: state.stars.calibrators.map((s, j) => (j === which ? { ...s, name } : s)) },
      })
  }

  if (!summary || frames.length === 0)
    return <p className="muted">No frame summary yet — go back to the Check step first.</p>

  const colorOf = (name: string) => (state.stars.science?.name === name ? SCIENCE_COLOR : CAL_COLOR)

  return (
    <div>
      <h2>Pick the stars</h2>
      <p className="muted">
        Click the <strong style={{ color: SCIENCE_COLOR }}>science target</strong> first, then each{' '}
        <strong style={{ color: CAL_COLOR }}>calibrator</strong>. Drag to pan, scroll to zoom.
      </p>
      {/* live photometry-region sizes (294): edited here, drawn on the frame, shared with Parameters */}
      <div className="grid region-fields">
        <Num label="Aperture radius [px]" value={p.aperture_radius} onChange={(v) => setPhot('aperture_radius', v ?? 0)} />
        <Num label="Annulus inner [px]" value={p.annulus_inner} onChange={(v) => setPhot('annulus_inner', v ?? 0)} />
        <Num label="Annulus outer [px]" value={p.annulus_outer} onChange={(v) => setPhot('annulus_outer', v ?? 0)} />
        <Num label="FWHM [px]" value={p.fwhm} onChange={(v) => setPhot('fwhm', v ?? 0)} />
        <Num
          label="Crop half-width [px]"
          value={state.stars.crop_half_width}
          onChange={(v) => onChange({ ...state, stars: { ...state.stars, crop_half_width: v ?? 0 } })}
        />
      </div>
      <div className="viewer-split">
        <div className="viewer-main">
          <FrameViewer sid={sid} frames={frames} dims={dims} marksFor={marksFor} onPick={pick} onIndexChange={setIndex} />
        </div>
        {/* integrated flux vs diameter beside the image (295) */}
        <div className="growth-panel">
          <h3>Curve of growth</h3>
          <p className="muted">Enclosed flux vs aperture diameter (dashed = current aperture).</p>
          {curves.length === 0 && <p className="muted">Pick a star to see its profile.</p>}
          {curves.map((c) => (
            <div key={c.name} className="growth-item">
              <span className="growth-name" style={{ color: colorOf(c.name) }}>
                {c.name}
              </span>
              <GrowthPlot curve={c} color={colorOf(c.name)} />
            </div>
          ))}
        </div>
      </div>
      <ul className="star-list">
        {state.stars.science && (
          <li>
            <span className="badge science">science</span>
            <input value={state.stars.science.name} onChange={(e) => rename('science', e.target.value)} />
            <span className="coords">
              x={state.stars.science.x} y={state.stars.science.y}
            </span>
            <button title="remove" onClick={() => onChange({ ...state, stars: { ...state.stars, science: null } })}>
              ✕
            </button>
          </li>
        )}
        {state.stars.calibrators.map((s, i) => (
          <li key={i}>
            <span className="badge calibrator">calibrator</span>
            <input value={s.name} onChange={(e) => rename(i, e.target.value)} />
            <span className="coords">
              x={s.x} y={s.y}
            </span>
            <button title="remove" onClick={() => removeCal(i)}>
              ✕
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
