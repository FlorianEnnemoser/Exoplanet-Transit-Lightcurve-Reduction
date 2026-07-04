// api.ts — typed fetch layer over the S-30 endpoints. GPL-3.0-or-later.

export interface Star {
  name: string
  x: number // array row (legacy indexing, see exotransit.photometry)
  y: number // array column
}

export interface WizardState {
  schema_version: number
  observation: { target: string; casename: string }
  paths: { lights: string; darks: string; bias: string; flats: string; output: string }
  stars: { science: Star | null; calibrators: Star[]; crop_half_width: number }
  reduction: {
    method: string
    cut: string
    cut_sigma: number
    cut_maxiters: number
    master_dark_combine: string
    master_bias_combine: string
  }
  photometry: {
    fwhm: number
    threshold_factor: number
    threshold_stat: string
    background_sigma: number
    background_maxiters: number
    aperture_radius: number
    annulus_inner: number
    annulus_outer: number
    method: string
    subpixels: number
    saturation: number
  }
  tracking: {
    mode: string
    reference_frame: number
    manual_shifts: { frame: number; dx: number; dy: number }[]
  }
  transit: { predicted_start: string; predicted_end: string; baseline_fit: string }
  system: {
    r_star: number | null
    r_star_err: number
    semi_major_axis: number | null
    period: number | null
    m_planet: number | null
    m_planet_err: number
    transit_duration: number | null
    ra: string
    dec: string
    site: { name: string; lat: number | null; lon: number | null; height: number | null }
  }
  output: { write_csv: boolean; figures: boolean; colormap: string; figsize: number[] }
  logging: { level: string; file: string }
}

export interface CategorySummary {
  count: number
  dims: [number, number] | null // (rows, cols)
  exptime_range: [number, number] | null
  time_obs_range: [string, string] | null
  problems: string[]
  frames: { name: string; time_obs: string | null; exptime: number | null }[]
}

export type Summary = Record<'lights' | 'darks' | 'bias' | 'flats', CategorySummary>

export interface Verdict {
  valid: boolean
  error: string | null
  toml: string | null
}

export interface LookupResult {
  found: boolean
  values: Partial<WizardState['system']>
  note: string
}

export interface GrowthCurve {
  name: string
  diameter: number[]
  flux: (number | null)[]
  aperture_radius: number
}

export interface PreviewResult {
  labels: string[]
  ensemble: (number | null)[]
  ratios: Record<string, (number | null)[]>
  science: { name: string; x: number; y: number }
  quality: string[]
  shifts: [number, number][]
  ingress: number | null
  egress: number | null
}

export interface PreviewJob {
  status: 'idle' | 'running' | 'done' | 'error'
  progress: number
  stage: string
  result: PreviewResult | null
  error: string | null
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init)
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : resp.statusText)
  }
  return resp.json() as Promise<T>
}

const jsonInit = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  createSession: (paths: Partial<WizardState['paths']>) =>
    request<{ id: string; state: WizardState }>('/api/sessions', jsonInit('POST', paths)),

  getConfig: (sid: string) => request<WizardState>(`/api/sessions/${sid}/config`),

  putConfig: (sid: string, state: WizardState) =>
    request<{ ok: boolean }>(`/api/sessions/${sid}/config`, jsonInit('PUT', state)),

  getSummary: (sid: string) => request<Summary>(`/api/sessions/${sid}/summary`),

  validate: (sid: string) =>
    request<Verdict>(`/api/sessions/${sid}/validate`, { method: 'POST' }),

  upload: (sid: string, category: string, files: FileList) => {
    const form = new FormData()
    for (const f of files) form.append('files', f)
    return request<{ stored: number; path: string; state: WizardState }>(
      `/api/sessions/${sid}/upload?category=${category}`,
      { method: 'POST', body: form },
    )
  },

  frameUrl: (sid: string, index: number, scale: string) =>
    `/api/sessions/${sid}/frames/${index}/png?scale=${scale}`,

  frameGrowth: (sid: string, index: number) =>
    request<{ stars: GrowthCurve[] }>(`/api/sessions/${sid}/frames/${index}/growth`),

  lookup: (target: string) =>
    request<LookupResult>(`/api/lookup?target=${encodeURIComponent(target)}`),

  startPreview: (sid: string) =>
    request<PreviewJob>(`/api/sessions/${sid}/preview`, { method: 'POST' }),

  getPreview: (sid: string) => request<PreviewJob>(`/api/sessions/${sid}/preview`),

  exportUrl: (sid: string) => `/api/sessions/${sid}/config/export`,
}
