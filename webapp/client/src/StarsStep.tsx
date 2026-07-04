// StarsStep.tsx — frame viewer, click-to-pick stars, timeline (W-4/5/14). GPL-3.0-or-later.
//
// Coordinate convention: the backend PNG keeps array row 0 at the top and is
// down-sampled by a uniform integer step, so array coordinates are display
// coordinates × (dims[0] / naturalHeight). Config stars use the legacy
// indexing star.x = row, star.y = column (see exotransit.photometry).
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Star, type Summary, type WizardState } from './api'

const VIEW_H = 520
const SCIENCE_COLOR = '#fc3d21' // --nasa-red
const CAL_COLOR = '#5b8def'

interface View {
  k: number
  tx: number
  ty: number
}

export default function StarsStep(props: {
  sid: string
  state: WizardState
  summary: Summary | null
  onChange: (s: WizardState) => void
}) {
  const { sid, state, summary, onChange } = props
  const [index, setIndex] = useState(0)
  const [scale, setScale] = useState('zscale')
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const view = useRef<View | null>(null)
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null)

  const frames = summary?.lights.frames ?? []
  const dims = summary?.lights.dims ?? null
  const nFrames = frames.length

  // load the current frame; prefetch neighbours so scrubbing stays fluid (W-NFR-2)
  useEffect(() => {
    let cancelled = false
    const im = new Image()
    im.onload = () => {
      if (!cancelled) setImg(im)
    }
    im.src = api.frameUrl(sid, index, scale)
    for (const d of [1, -1, 2, -2, 3, -3]) {
      const j = index + d
      if (j >= 0 && j < nFrames) new Image().src = api.frameUrl(sid, j, scale)
    }
    return () => {
      cancelled = true
    }
  }, [sid, index, scale, nFrames])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !img) return
    const w = canvas.clientWidth
    canvas.width = w
    canvas.height = VIEW_H
    if (!view.current) {
      // first image: fit and centre
      const k = Math.min(w / img.naturalWidth, VIEW_H / img.naturalHeight)
      view.current = {
        k,
        tx: (w - img.naturalWidth * k) / 2,
        ty: (VIEW_H - img.naturalHeight * k) / 2,
      }
    }
    const { k, tx, ty } = view.current
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, w, VIEW_H)
    ctx.imageSmoothingEnabled = k < 3 // show hard pixels when zoomed far in
    ctx.setTransform(k, 0, 0, k, tx, ty)
    ctx.drawImage(img, 0, 0)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    if (!dims) return
    const step = dims[0] / img.naturalHeight
    const marks = [
      ...(state.stars.science ? [{ s: state.stars.science, color: SCIENCE_COLOR }] : []),
      ...state.stars.calibrators.map((s) => ({ s, color: CAL_COLOR })),
    ]
    for (const { s, color } of marks) {
      const dx = (s.y / step) * k + tx // column -> display x
      const dy = (s.x / step) * k + ty // row -> display y
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(dx, dy, 10, 0, 2 * Math.PI)
      ctx.stroke()
      ctx.fillStyle = color
      ctx.font = '12px Helvetica, Arial, sans-serif'
      ctx.fillText(s.name, dx + 13, dy - 9)
    }
  }, [img, dims, state.stars])

  useEffect(draw, [draw])

  // wheel zoom around the cursor (native listener: React's is passive)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      if (!view.current) return
      const rect = canvas.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const f = e.deltaY < 0 ? 1.2 : 1 / 1.2
      const v = view.current
      const k = Math.min(Math.max(v.k * f, 0.05), 40)
      view.current = {
        k,
        tx: mx - ((mx - v.tx) / v.k) * k,
        ty: my - ((my - v.ty) / v.k) * k,
      }
      draw()
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [draw])

  const pick = (row: number, col: number) => {
    if (!state.stars.science) {
      const name = state.observation.target.replace(/\s+/g, '') || 'Science'
      onChange({ ...state, stars: { ...state.stars, science: { name, x: row, y: col } } })
    } else {
      const cal: Star = {
        name: `Calibrator_${state.stars.calibrators.length + 1}`,
        x: row,
        y: col,
      }
      onChange({
        ...state,
        stars: { ...state.stars, calibrators: [...state.stars.calibrators, cal] },
      })
    }
  }

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { x: e.clientX, y: e.clientY, moved: false }
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drag.current || !view.current) return
    const dx = e.clientX - drag.current.x
    const dy = e.clientY - drag.current.y
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true
    if (drag.current.moved) {
      view.current = { ...view.current, tx: view.current.tx + dx, ty: view.current.ty + dy }
      drag.current = { x: e.clientX, y: e.clientY, moved: true }
      draw()
    }
  }

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const wasDrag = drag.current?.moved ?? true
    drag.current = null
    if (wasDrag || !img || !dims || !view.current) return
    const rect = e.currentTarget.getBoundingClientRect()
    const v = view.current
    const step = dims[0] / img.naturalHeight
    const col = ((e.clientX - rect.left - v.tx) / v.k) * step
    const row = ((e.clientY - rect.top - v.ty) / v.k) * step
    if (row < 0 || col < 0 || row >= dims[0] || col >= dims[1]) return
    pick(Math.round(row), Math.round(col))
  }

  const removeCal = (i: number) =>
    onChange({
      ...state,
      stars: { ...state.stars, calibrators: state.stars.calibrators.filter((_, j) => j !== i) },
    })

  const rename = (which: 'science' | number, name: string) => {
    if (which === 'science' && state.stars.science)
      onChange({ ...state, stars: { ...state.stars, science: { ...state.stars.science, name } } })
    else if (typeof which === 'number')
      onChange({
        ...state,
        stars: {
          ...state.stars,
          calibrators: state.stars.calibrators.map((s, j) => (j === which ? { ...s, name } : s)),
        },
      })
  }

  if (!summary || nFrames === 0)
    return <p className="muted">No frame summary yet — go back to the Check step first.</p>

  const stamp = frames[index]?.time_obs ?? ''

  return (
    <div>
      <h2>Pick the stars</h2>
      <p className="muted">
        Click the <strong style={{ color: SCIENCE_COLOR }}>science target</strong> first, then each{' '}
        <strong style={{ color: CAL_COLOR }}>calibrator</strong>. Drag to pan, scroll to zoom.
      </p>
      <div className="viewer-bar">
        <label>
          Scaling{' '}
          <select value={scale} onChange={(e) => setScale(e.target.value)} style={{ width: 'auto' }}>
            <option value="zscale">zscale</option>
            <option value="linear">linear</option>
            <option value="log">log</option>
          </select>
        </label>
        <span className="spacer" />
        <button
          className="ghost"
          onClick={() => {
            view.current = null
            draw()
          }}
        >
          Reset view
        </button>
      </div>
      <div className="viewer">
        <canvas
          ref={canvasRef}
          style={{ height: VIEW_H }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        />
      </div>
      {/* timeline scrubber (W-14): native range input = free keyboard stepping */}
      <div className="timeline">
        <input
          type="range"
          min={0}
          max={Math.max(nFrames - 1, 0)}
          value={index}
          aria-label="frame timeline"
          onChange={(e) => setIndex(Number(e.target.value))}
        />
        <span className="stamp">
          frame {index + 1}/{nFrames}
          {stamp ? ` · ${stamp}` : ''}
        </span>
      </div>
      <ul className="star-list">
        {state.stars.science && (
          <li>
            <span className="badge science">science</span>
            <input
              value={state.stars.science.name}
              onChange={(e) => rename('science', e.target.value)}
            />
            <span className="coords">
              x={state.stars.science.x} y={state.stars.science.y}
            </span>
            <button
              title="remove"
              onClick={() => onChange({ ...state, stars: { ...state.stars, science: null } })}
            >
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
