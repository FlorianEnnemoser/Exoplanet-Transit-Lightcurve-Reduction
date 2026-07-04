# webapp.server.growth — integrated-flux-vs-diameter curve of growth (W-6 aid).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Background-subtracted curve of growth for a picked star.

Shows how the enclosed flux grows with aperture diameter so the user can size
the aperture at the plateau. Reads the **full-resolution** raw FITS (W-NFR-3 —
display down-sampling never touches the numbers) and reuses
``photutils.profiles.CurveOfGrowth`` plus the same sky estimator family as
:mod:`exotransit.photometry`.
"""

import logging
import math

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAnnulus
from photutils.profiles import CurveOfGrowth

logger = logging.getLogger(__name__)


def curve(path: str, star: dict, phot: dict, half_width: int) -> dict:
    """Return ``{name, diameter, flux, aperture_radius}`` for one star.

    ``star`` is ``{name, x (row), y (col)}``; ``phot`` is the wizard
    ``photometry`` block. Flux is background-subtracted so it plateaus at the
    star's total flux. Returns an empty ``flux`` list if the star sits too close
    to the frame edge to build the sky annulus.
    """
    data = fits.getdata(path).astype(np.float32)
    rows, cols = data.shape
    row, col = int(star["x"]), int(star["y"])
    xy = (float(col), float(row))  # photutils x = column = star.y, y = row = star.x

    r_out = float(phot["annulus_outer"])
    r_in = float(phot["annulus_inner"])
    # need room for the sky annulus; bail gracefully at the edge
    if not (r_out <= col < cols - r_out and r_out <= row < rows - r_out):
        return {
            "name": star["name"],
            "diameter": [],
            "flux": [],
            "aperture_radius": phot["aperture_radius"],
        }

    max_r = int(min(half_width, math.ceil(r_out) + 4))
    radii = np.arange(1, max_r + 1, dtype=float)
    cog = CurveOfGrowth(data, xy, radii)

    # local sky per pixel: sigma-clipped median in the config annulus
    annulus = CircularAnnulus([xy], r_in=r_in, r_out=r_out)
    mask = annulus.to_mask(method="center")[0]
    sky_pixels = mask.get_values(data)
    sky = (
        float(sigma_clipped_stats(sky_pixels, sigma=3.0, maxiters=5)[1]) if sky_pixels.size else 0.0
    )

    flux = cog.profile - sky * math.pi * radii**2
    return {
        "name": star["name"],
        "diameter": (2.0 * radii).tolist(),
        "flux": flux.tolist(),
        "aperture_radius": phot["aperture_radius"],
    }
