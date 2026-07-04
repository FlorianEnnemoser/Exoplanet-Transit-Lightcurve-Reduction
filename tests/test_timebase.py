# BJD_TDB computation + coord/site config validation (S-14, S-25). GPL-3.0-or-later.
from __future__ import annotations

import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from conftest import write_toml

from exotransit import config, timebase
from exotransit.config import ConfigError

# Frozen reference: WASP-52 target + Lustbühel site, DATE-OBS of the first
# _WASP52b frame, EXPTIME 10 s (mid-exposure +5 s), computed once with astropy
# 6.1.7. Barycentric correction for this pointing/epoch is ~472 s.
REF_BJD_TDB = 2457662.386552
DATE_OBS = "2016-09-30T21:07:32.369"
TIME_OBS = "21:07:32.369"
EXPTIME = 10.0


def _target_site():
    target = SkyCoord("23h13m58.76s", "+08d45m40.6s", frame="icrs")
    site = EarthLocation(lat=47.0678 * u.deg, lon=15.4936 * u.deg, height=480.0 * u.m)
    return target, site


def test_bjd_tdb_matches_reference():
    target, site = _target_site()
    b = timebase.bjd_tdb(DATE_OBS, TIME_OBS, EXPTIME, target, site)
    assert b == pytest.approx(REF_BJD_TDB, abs=1.0 / 86400.0)  # < 1 second (S-25)


def test_barycentric_correction_within_light_travel_bound():
    target, site = _target_site()
    b = timebase.bjd_tdb(DATE_OBS, TIME_OBS, EXPTIME, target, site)
    t_mid = Time(DATE_OBS, scale="utc", format="isot", location=site) + (EXPTIME / 2) * u.s
    # independent sanity: |BJD − mid-exposure TDB| cannot exceed Earth–barycentre
    # light travel time (~8.32 min); a much larger value means a broken correction.
    assert abs(b - t_mid.tdb.jd) * 86400.0 < 8.4 * 60.0


def test_iso_uses_full_date_obs_when_present():
    # DATE-OBS already carries the time part → TIME-OBS is not appended.
    assert timebase._iso(DATE_OBS, TIME_OBS) == DATE_OBS
    # date-only DATE-OBS → time joined on.
    assert timebase._iso("2016-09-30", "21:07:32") == "2016-09-30T21:07:32"


def test_unparseable_ra_rejected(tmp_path, wasp_config_text):
    text = wasp_config_text.replace('ra = "23h13m58.76s"', 'ra = "not-a-coordinate"')
    with pytest.raises(ConfigError, match="ra"):
        config.load(write_toml(tmp_path, text))


def test_missing_site_lat_rejected(tmp_path, wasp_config_text):
    text = wasp_config_text.replace("lat = 47.0678\n", "")  # drop required site.lat
    with pytest.raises(ConfigError, match="site"):
        config.load(write_toml(tmp_path, text))
