// steps.tsx — wizard form steps (W-1/2/8/9/10-lite, W-21). GPL-3.0-or-later.
import { useState } from 'react'
import { api, type CategorySummary, type Summary, type Verdict, type WizardState } from './api'

type Patch = (s: WizardState) => void

// ---- shared field helpers --------------------------------------------------

function Text(props: {
  label: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
  invalid?: boolean
}) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input
        className={props.invalid ? 'invalid' : ''}
        value={props.value}
        placeholder={props.placeholder}
        onChange={(e) => props.onChange(e.target.value)}
      />
    </label>
  )
}

function Num(props: {
  label: string
  value: number | null
  onChange: (v: number | null) => void
  invalid?: boolean
}) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input
        className={props.invalid ? 'invalid' : ''}
        type="number"
        step="any"
        value={props.value ?? ''}
        onChange={(e) => props.onChange(e.target.value === '' ? null : Number(e.target.value))}
      />
    </label>
  )
}

function Select(props: {
  label: string
  value: string
  options: string[]
  onChange: (v: string) => void
}) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <select value={props.value} onChange={(e) => props.onChange(e.target.value)}>
        {props.options.map((o) => (
          <option key={o}>{o}</option>
        ))}
      </select>
    </label>
  )
}

// ---- live validation rules (W-9); App gates Next on the first message ------

export function paramsProblems(s: WizardState): string[] {
  const p = s.photometry
  const out: string[] = []
  const num = (v: number | null): v is number => v !== null && Number.isFinite(v)
  if (!num(p.aperture_radius) || p.aperture_radius <= 0)
    out.push('Aperture radius must be greater than 0.')
  if (!num(p.annulus_inner) || !num(p.annulus_outer) || !(0 < p.annulus_inner && p.annulus_inner < p.annulus_outer))
    out.push('Annulus must satisfy 0 < inner < outer.')
  if (!num(p.fwhm) || p.fwhm <= 0) out.push('FWHM must be greater than 0.')
  if (!num(p.threshold_factor) || p.threshold_factor <= 0)
    out.push('Detection threshold factor must be greater than 0.')
  if (!num(s.stars.crop_half_width) || s.stars.crop_half_width < Math.ceil(p.annulus_outer))
    out.push(`Crop half-width must be at least the outer annulus radius (${Math.ceil(p.annulus_outer)}).`)
  return out
}

export function systemProblems(s: WizardState): string[] {
  const y = s.system
  const out: string[] = []
  const need = (v: number | null, name: string) => {
    if (v === null || !Number.isFinite(v)) out.push(`${name} is required.`)
  }
  need(y.r_star, 'Stellar radius')
  need(y.semi_major_axis, 'Semi-major axis')
  need(y.period, 'Orbital period')
  need(y.m_planet, 'Planet mass')
  need(y.transit_duration, 'Transit duration')
  if (!y.ra || !y.dec) out.push('Target RA and Dec are required (for BJD_TDB).')
  need(y.site.lat, 'Observatory latitude')
  need(y.site.lon, 'Observatory longitude')
  need(y.site.height, 'Observatory height')
  return out
}

// ---- 1 · Data (W-1) ---------------------------------------------------------

export function DataStep(props: { sid: string; state: WizardState; onChange: Patch }) {
  const { sid, state, onChange } = props
  const [uploadNote, setUploadNote] = useState('')
  const paths = state.paths
  const setPath = (k: keyof WizardState['paths'], v: string) =>
    onChange({ ...state, paths: { ...paths, [k]: v } })

  const upload = async (category: string, files: FileList | null) => {
    if (!files || files.length === 0) return
    const r = await api.upload(sid, category, files)
    onChange({ ...state, paths: { ...paths, [category]: r.path } })
    setUploadNote(`Stored ${r.stored} ${category} frame(s) in the session.`)
  }

  return (
    <div>
      <h2>Where are the frames?</h2>
      <p className="muted">
        Point at server-side directories (suffix naming: <code>*.BIAS.FIT</code> /{' '}
        <code>*.DARK.FIT</code> / rest = lights), or upload frames per category.
      </p>
      <div className="grid">
        <Text label="Lights directory" value={paths.lights} onChange={(v) => setPath('lights', v)} placeholder="_WASP52b/WASP52b" />
        <Text label="Darks directory" value={paths.darks} onChange={(v) => setPath('darks', v)} />
        <Text label="Bias directory" value={paths.bias} onChange={(v) => setPath('bias', v)} />
        <Text label="Flats directory (optional)" value={paths.flats} onChange={(v) => setPath('flats', v)} />
        <Text label="Output directory" value={paths.output} onChange={(v) => setPath('output', v)} />
      </div>
      <h2 style={{ marginTop: '1rem' }}>…or upload</h2>
      <div className="grid">
        {(['lights', 'darks', 'bias', 'flats'] as const).map((cat) => (
          <label className="field" key={cat}>
            <span>Upload {cat}</span>
            <input type="file" multiple accept=".fit,.fits,.FIT,.FITS" onChange={(e) => void upload(cat, e.target.files)} />
          </label>
        ))}
      </div>
      {uploadNote && <p className="muted">{uploadNote}</p>}
    </div>
  )
}

// ---- 2 · Check (W-2) --------------------------------------------------------

function CategoryRow(props: { name: string; cat: CategorySummary }) {
  const { name, cat } = props
  return (
    <>
      <tr>
        <td>{name}</td>
        <td>{cat.count}</td>
        <td>{cat.dims ? `${cat.dims[0]} × ${cat.dims[1]}` : '—'}</td>
        <td>{cat.exptime_range ? `${cat.exptime_range[0]}–${cat.exptime_range[1]} s` : '—'}</td>
        <td>{cat.time_obs_range ? `${cat.time_obs_range[0]} → ${cat.time_obs_range[1]}` : '—'}</td>
      </tr>
      {cat.problems.map((p) => (
        <tr key={p}>
          <td colSpan={5} className="warn">
            ⚠ {name}: {p}
          </td>
        </tr>
      ))}
    </>
  )
}

export function CheckStep(props: { summary: Summary | null; onRefresh: () => void }) {
  const { summary, onRefresh } = props
  if (!summary) return <p className="muted">Reading FITS headers…</p>
  return (
    <div>
      <h2>Frame check</h2>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>Frames</th>
            <th>Dimensions</th>
            <th>Exposure</th>
            <th>TIME-OBS range</th>
          </tr>
        </thead>
        <tbody>
          {(['lights', 'darks', 'bias', 'flats'] as const).map((c) => (
            <CategoryRow key={c} name={c} cat={summary[c]} />
          ))}
        </tbody>
      </table>
      <div className="nav">
        <button className="ghost" onClick={onRefresh}>
          Re-check
        </button>
      </div>
    </div>
  )
}

// ---- 4 · Parameters (W-8, W-9) ----------------------------------------------

export function ParamsStep(props: { state: WizardState; onChange: Patch }) {
  const { state, onChange } = props
  const p = state.photometry
  const setP = (k: keyof WizardState['photometry'], v: number | string | null) =>
    onChange({ ...state, photometry: { ...p, [k]: v } })
  const setR = (k: keyof WizardState['reduction'], v: number | string | null) =>
    onChange({ ...state, reduction: { ...state.reduction, [k]: v } })
  const setT = (k: keyof WizardState['tracking'], v: number | string | null) =>
    onChange({ ...state, tracking: { ...state.tracking, [k]: v } })

  const annulusBad = !(0 < p.annulus_inner && p.annulus_inner < p.annulus_outer)

  return (
    <div>
      <h2>Reduction</h2>
      <div className="grid">
        <Select label="Method" value={state.reduction.method} options={['standard', 'none', 'bias', 'dark_bias']} onChange={(v) => setR('method', v)} />
        <Select label="Per-frame cut" value={state.reduction.cut} options={['none', 'median', 'average', 'min', 'sigma_clip']} onChange={(v) => setR('cut', v)} />
        <Select label="Master dark combine" value={state.reduction.master_dark_combine} options={['median', 'mean']} onChange={(v) => setR('master_dark_combine', v)} />
        <Select label="Master bias combine" value={state.reduction.master_bias_combine} options={['median', 'mean']} onChange={(v) => setR('master_bias_combine', v)} />
      </div>
      <h2>Photometry</h2>
      <div className="grid">
        <Num label="Aperture radius [px]" value={p.aperture_radius} invalid={!(p.aperture_radius > 0)} onChange={(v) => setP('aperture_radius', v)} />
        <Num label="Annulus inner [px]" value={p.annulus_inner} invalid={annulusBad} onChange={(v) => setP('annulus_inner', v)} />
        <Num label="Annulus outer [px]" value={p.annulus_outer} invalid={annulusBad} onChange={(v) => setP('annulus_outer', v)} />
        <Num label="FWHM [px]" value={p.fwhm} invalid={!(p.fwhm > 0)} onChange={(v) => setP('fwhm', v)} />
        <Num label="Threshold factor" value={p.threshold_factor} invalid={!(p.threshold_factor > 0)} onChange={(v) => setP('threshold_factor', v)} />
        <Select label="Threshold statistic" value={p.threshold_stat} options={['std', 'mean', 'median']} onChange={(v) => setP('threshold_stat', v)} />
        <Num label="Background sigma" value={p.background_sigma} onChange={(v) => setP('background_sigma', v)} />
        <Num
          label="Crop half-width [px]"
          value={state.stars.crop_half_width}
          invalid={state.stars.crop_half_width < Math.ceil(p.annulus_outer)}
          onChange={(v) => onChange({ ...state, stars: { ...state.stars, crop_half_width: v ?? 0 } })}
        />
      </div>
      <h2>Tracking</h2>
      <div className="grid">
        <Select label="Mode" value={state.tracking.mode} options={['auto', 'manual', 'off']} onChange={(v) => setT('mode', v)} />
        <Num label="Reference frame" value={state.tracking.reference_frame} onChange={(v) => setT('reference_frame', v ?? 0)} />
      </div>
      {paramsProblems(state).map((m) => (
        <p key={m} className="warn">
          ⚠ {m}
        </p>
      ))}
    </div>
  )
}

// ---- 5 · System & transit (manual entry; catalogue lookup lands in M2) ------

export function SystemStep(props: { state: WizardState; onChange: Patch }) {
  const { state, onChange } = props
  const y = state.system
  const setY = (k: string, v: number | string | null) =>
    onChange({ ...state, system: { ...y, [k]: v } })
  const setSite = (k: string, v: number | string | null) =>
    onChange({ ...state, system: { ...y, site: { ...y.site, [k]: v } } })
  const setObs = (k: 'target' | 'casename', v: string) =>
    onChange({ ...state, observation: { ...state.observation, [k]: v } })
  const setTransit = (k: string, v: string) =>
    onChange({ ...state, transit: { ...state.transit, [k]: v } })

  return (
    <div>
      <h2>Observation</h2>
      <div className="grid">
        <Text label="Target name" value={state.observation.target} onChange={(v) => setObs('target', v)} placeholder="WASP-52 b" />
        <Text label="Case name (filenames)" value={state.observation.casename} onChange={(v) => setObs('casename', v)} />
      </div>
      <h2>Star & planet (NASA Exoplanet Archive values)</h2>
      <div className="grid">
        <Num label="Stellar radius [R☉]" value={y.r_star} invalid={y.r_star === null} onChange={(v) => setY('r_star', v)} />
        <Num label="Stellar radius error" value={y.r_star_err} onChange={(v) => setY('r_star_err', v ?? 0)} />
        <Num label="Semi-major axis [au]" value={y.semi_major_axis} invalid={y.semi_major_axis === null} onChange={(v) => setY('semi_major_axis', v)} />
        <Num label="Period [days]" value={y.period} invalid={y.period === null} onChange={(v) => setY('period', v)} />
        <Num label="Planet mass [M♃]" value={y.m_planet} invalid={y.m_planet === null} onChange={(v) => setY('m_planet', v)} />
        <Num label="Planet mass error" value={y.m_planet_err} onChange={(v) => setY('m_planet_err', v ?? 0)} />
        <Num label="Transit duration [min]" value={y.transit_duration} invalid={y.transit_duration === null} onChange={(v) => setY('transit_duration', v)} />
        <Text label="RA (ICRS)" value={y.ra} invalid={!y.ra} onChange={(v) => setY('ra', v)} placeholder="23h13m58.76s" />
        <Text label="Dec (ICRS)" value={y.dec} invalid={!y.dec} onChange={(v) => setY('dec', v)} placeholder="+08d45m40.6s" />
      </div>
      <h2>Observatory site (for BJD_TDB)</h2>
      <div className="grid">
        <Text label="Site name" value={y.site.name} onChange={(v) => setSite('name', v)} placeholder="Lustbuehel Observatory" />
        <Num label="Latitude [deg]" value={y.site.lat} invalid={y.site.lat === null} onChange={(v) => setSite('lat', v)} />
        <Num label="Longitude [deg]" value={y.site.lon} invalid={y.site.lon === null} onChange={(v) => setSite('lon', v)} />
        <Num label="Height [m]" value={y.site.height} invalid={y.site.height === null} onChange={(v) => setSite('height', v)} />
      </div>
      <h2>Transit window</h2>
      <div className="grid">
        <Text label="Predicted ingress (UTC HH:MM:SS)" value={state.transit.predicted_start} onChange={(v) => setTransit('predicted_start', v)} placeholder="21:48:00" />
        <Text label="Predicted egress (UTC HH:MM:SS)" value={state.transit.predicted_end} onChange={(v) => setTransit('predicted_end', v)} placeholder="23:36:00" />
        <Select label="Baseline fit" value={state.transit.baseline_fit} options={['linear', 'median']} onChange={(v) => setTransit('baseline_fit', v)} />
      </div>
      {systemProblems(state).map((m) => (
        <p key={m} className="warn">
          ⚠ {m}
        </p>
      ))}
    </div>
  )
}

// ---- 6 · Review (W-21) --------------------------------------------------------

function reviewRows(state: WizardState): [string, string, string][] {
  const rows: [string, string, string][] = []
  const fmt = (v: unknown) => (v === null || v === '' ? '—' : JSON.stringify(v))
  const walk = (section: string, obj: Record<string, unknown>) => {
    for (const [k, v] of Object.entries(obj)) {
      if (v !== null && typeof v === 'object' && !Array.isArray(v))
        walk(`${section}.${k}`, v as Record<string, unknown>)
      else rows.push([section, k, fmt(v)])
    }
  }
  for (const [k, v] of Object.entries(state)) {
    if (v !== null && typeof v === 'object' && !Array.isArray(v))
      walk(k, v as Record<string, unknown>)
    else rows.push(['', k, fmt(v)])
  }
  return rows
}

export function ReviewStep(props: { state: WizardState; verdict: Verdict | null }) {
  const { state, verdict } = props
  return (
    <div>
      <h2>Review every value before export</h2>
      {verdict && !verdict.valid && <p className="error">⚠ {verdict.error}</p>}
      {verdict?.valid && <p className="muted">Configuration is valid — the pipeline accepts it as-is.</p>}
      <table>
        <thead>
          <tr>
            <th>Section</th>
            <th>Field</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {reviewRows(state).map(([s, k, v]) => (
            <tr key={`${s}.${k}`}>
              <td className="muted">{s}</td>
              <td>{k}</td>
              <td>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---- 7 · Export (W-11) --------------------------------------------------------

export function ExportStep(props: { sid: string; state: WizardState; verdict: Verdict | null }) {
  const { sid, state, verdict } = props
  const casename = state.observation.casename || 'exotransit'
  return (
    <div>
      <h2>Export pipeline configuration</h2>
      <p className="muted">
        Run it with: <code>uv run exotransit reduce {casename}.toml</code>
      </p>
      <p>
        <a className="primary" href={api.exportUrl(sid)} download={`${casename}.toml`}>
          Download {casename}.toml
        </a>
      </p>
      {verdict?.toml && <pre style={{ overflowX: 'auto' }}>{verdict.toml}</pre>}
    </div>
  )
}
