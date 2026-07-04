# webapp.server.rendering — FITS frame -> display PNG (W-4, W-NFR-2).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Render a light frame as an 8-bit grayscale PNG for the viewer.

Display only: frames are down-sampled to at most ``MAX_DISPLAY_PX`` on the
longer side (W-NFR-2) while photometry always reads full resolution
(W-NFR-3). Scaling reuses ``astropy.visualization``; PNG encoding reuses
matplotlib — no new imaging dependency.
"""

import io
from functools import lru_cache

import numpy as np
from astropy.io import fits
from astropy.visualization import LogStretch, ZScaleInterval
from matplotlib import image as mpl_image

MAX_DISPLAY_PX = 1024
SCALES = ("linear", "log", "zscale")


@lru_cache(maxsize=64)  # frames are immutable per session; key = (path, scale)
def render_png(path: str, scale: str) -> bytes:
    """Render the FITS image at ``path`` with the given scale mode.

    The PNG keeps matplotlib's default ``origin='upper'``: display row 0 is
    array row 0, so a client click maps to array indices by the down-sample
    factor alone — and ``star.x`` = row, ``star.y`` = column, matching the
    legacy indexing in :func:`exotransit.photometry.measure_star`.
    """
    data = fits.getdata(path).astype(np.float32)
    step = max(1, -(-max(data.shape) // MAX_DISPLAY_PX))  # ceil division
    data = data[::step, ::step]
    if scale == "zscale":
        lo, hi = ZScaleInterval().get_limits(data)
    else:
        lo, hi = np.nanpercentile(data, [0.5, 99.9])  # robust linear/log range
    norm = np.clip((data - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    if scale == "log":
        norm = LogStretch()(norm)
    buf = io.BytesIO()
    mpl_image.imsave(buf, norm, cmap="gray", vmin=0.0, vmax=1.0, format="png")
    return buf.getvalue()
