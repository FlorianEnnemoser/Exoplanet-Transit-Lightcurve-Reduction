// FrameViewer.tsx — reusable frame canvas: pan/zoom, scrubber, region overlays.
// GPL-3.0-or-later.
//
// Shared by the Stars step (pickable) and the Preview step (read-only, tracked).
// Coordinate convention: the backend PNG keeps array row 0 at the top and is
// down-sampled by a uniform integer step, so array coordinates map to display
// coordinates by dims[0] / naturalHeight. A star's array position is
// (row = x, col = y), matching the legacy exotransit.photometry indexing.
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type CategorySummary } from './api'

const VIEW_H = 520

type Frame = CategorySummary['frames'][number]

export interface Mark {
  row: number // array row (x)
  col: number // array column (y)
  color: string
  label?: string
  aperture?: number // aperture radius [array px]; falls back to a fixed 10 px ring
  annulus?: [number, number] // inner/outer sky annulus [array px]
  crop?: number // crop half-width [array px]
}

interface View {
  k: number
  tx: number
  ty: number
}

export default function FrameViewer(props: {
  sid: string
  frames: Frame[]
  dims: [number, number] | null
  marksFor: (index: number) => Mark[]
  onPick?: (row: number, col: number) => void
  onIndexChange?: (index: number) => void
}) {
  const { sid, frames, dims, marksFor, onPick, onIndexChange } = props
  const [index, setIndex] = useState(0)
  const [scale, setScale] = useState('zscale')
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const view = useRef<View | null>(null)
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null)
  const nFrames = frames.length

  useEffect(() => onIndexChange?.(index), [index, onIndexChange])

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
    ctx.imageSmoothingEnabled = k < 3
    ctx.setTransform(k, 0, 0, k, tx, ty)
    ctx.drawImage(img, 0, 0)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    if (!dims) return
    const step = dims[0] / img.naturalHeight
    const scl = k / step // array px -> display px
    for (const m of marksFor(index)) {
      const dx = (m.col / step) * k + tx
      const dy = (m.row / step) * k + ty
      ctx.strokeStyle = m.color
      ctx.lineWidth = 1.5
      ctx.setLineDash([])
      ctx.beginPath()
      ctx.arc(dx, dy, m.aperture != null ? m.aperture * scl : 10, 0, 2 * Math.PI)
      ctx.stroke()
      if (m.annulus) {
        ctx.setLineDash([4, 3])
        for (const r of m.annulus) {
          ctx.beginPath()
          ctx.arc(dx, dy, r * scl, 0, 2 * Math.PI)
          ctx.stroke()
        }
      }
      if (m.crop != null) {
        ctx.setLineDash([1, 2])
        ctx.strokeRect(dx - m.crop * scl, dy - m.crop * scl, 2 * m.crop * scl, 2 * m.crop * scl)
      }
      ctx.setLineDash([])
      if (m.label) {
        ctx.fillStyle = m.color
        ctx.font = '12px Helvetica, Arial, sans-serif'
        ctx.fillText(m.label, dx + (m.aperture != null ? m.aperture * scl : 10) + 4, dy - 4)
      }
    }
  }, [img, dims, marksFor, index])

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
      view.current = { k, tx: mx - ((mx - v.tx) / v.k) * k, ty: my - ((my - v.ty) / v.k) * k }
      draw()
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [draw])

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
    if (wasDrag || !onPick || !img || !dims || !view.current) return
    const rect = e.currentTarget.getBoundingClientRect()
    const v = view.current
    const step = dims[0] / img.naturalHeight
    const col = ((e.clientX - rect.left - v.tx) / v.k) * step
    const row = ((e.clientY - rect.top - v.ty) / v.k) * step
    if (row < 0 || col < 0 || row >= dims[0] || col >= dims[1]) return
    onPick(Math.round(row), Math.round(col))
  }

  const stamp = frames[index]?.time_obs ?? ''

  return (
    <div>
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
          style={{ height: VIEW_H, cursor: onPick ? 'crosshair' : 'grab' }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        />
      </div>
      {/* timeline scrubber (W-14): native range input = free keyboard stepping */}
      <div className="timeline">
        <button
          className="ghost step"
          aria-label="previous frame"
          disabled={index <= 0}
          onClick={() => setIndex(Math.max(index - 1, 0))}
        >
          −
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(nFrames - 1, 0)}
          value={index}
          aria-label="frame timeline"
          onChange={(e) => setIndex(Number(e.target.value))}
        />
        <button
          className="ghost step"
          aria-label="next frame"
          disabled={index >= nFrames - 1}
          onClick={() => setIndex(Math.min(index + 1, nFrames - 1))}
        >
          +
        </button>
        <span className="stamp">
          frame {index + 1}/{nFrames}
          {stamp ? ` · ${stamp}` : ''}
        </span>
      </div>
    </div>
  )
}
