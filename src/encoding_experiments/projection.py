"""Map geographic ``(lat, lon)`` to a local Cartesian metre frame about a reference origin.

The encoders that use position geometry (``rolling`` and the ``bs_*`` family) and the
representation probes all need to turn lat/lon into east/north metres relative to some
origin (the train-set mean, or the base station). Historically that math was the spherical
small-angle approximation, copied inline at several sites. This module centralises it and
adds a true UTM option, selectable by name:

* ``"equirectangular"`` - the spherical equidistant-cylindrical approximation used originally
  throughout the project::

      x_east  = R * (lon - lon0) * cos(lat0)
      y_north = R * (lat - lat0)               (angles in radians, R = mean Earth radius)

  Dependency-free and accurate to sub-centimetre over the few-hundred-metre extent of a
  single DeepSense scenario.

* ``"utm"`` (default) - a true Universal Transverse Mercator projection on the **WGS84
  ellipsoid** (semi-major axis 6 378 137 m, 1/f = 298.257223563) via the :mod:`utm` package
  (Tobias Bieniek, https://github.com/Turbo87/utm) - the library cited by Morais et al. Both
  the point and the reference origin are projected to absolute UTM easting/northing and the
  origin is subtracted, so the output is the **same local-origin frame** as the equirectangular
  path - only the underlying projection differs. The UTM zone (number + latitude band) is
  chosen from the origin's mean longitude/latitude and forced for every row (each scenario sits
  in one zone), so the choice is deterministic and carries no train/test leakage.

Both :func:`to_local_xy` and its inverse :func:`to_latlon` accept scalars or array-likes for
every argument (the origin may be a single point or one-per-row) and return ``numpy`` arrays.
"""

from __future__ import annotations

import numpy as np

# Spherical mean Earth radius for the equirectangular path (matches the historic constant).
EARTH_RADIUS_M = 6_371_000
# Project-wide default. 'utm' (true Universal Transverse Mercator on WGS84) since 2026-06-15;
# 'equirectangular' (the historic spherical approximation) remains available per-encoder via
# params['projection'] / the ENCODING_PROJECTION env var.
DEFAULT_METHOD = "utm"


def _utm_zone(lat_ref: float, lon_ref: float) -> tuple[int, str]:
    """WGS84 / UTM zone (number 1..60, latitude-band letter) containing ``(lat_ref, lon_ref)``.

    The band letter encodes the hemisphere for the :mod:`utm` package (any letter >= 'N' is
    treated as northern), so forcing it on every row keeps all points in one consistent
    easting/northing frame regardless of which band an individual fix falls in.
    """
    import utm

    zone_number = utm.latlon_to_zone_number(lat_ref, lon_ref)
    zone_letter = utm.latitude_to_zone_letter(lat_ref)
    return zone_number, zone_letter


def _ref_point(lat0, lon0) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (scalar lat ref, scalar lon ref, lat0 array, lon0 array). The scalar refs
    (mean of the origin) pick the UTM zone; the arrays carry the per-row origin."""
    lat0a = np.asarray(lat0, dtype=float)
    lon0a = np.asarray(lon0, dtype=float)
    return float(np.nanmean(lat0a)), float(np.nanmean(lon0a)), lat0a, lon0a


def to_local_xy(lat, lon, lat0, lon0, method: str = DEFAULT_METHOD):
    """Project ``(lat, lon)`` to local ``(x_east_m, y_north_m)`` relative to ``(lat0, lon0)``."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_ref, lon_ref, lat0a, lon0a = _ref_point(lat0, lon0)
    if method == "equirectangular":
        x = EARTH_RADIUS_M * (np.radians(lon) - np.radians(lon0a)) * np.cos(np.radians(lat0a))
        y = EARTH_RADIUS_M * (np.radians(lat) - np.radians(lat0a))
        return x, y
    if method == "utm":
        import utm

        zone_number, zone_letter = _utm_zone(lat_ref, lon_ref)
        e, n, _, _ = utm.from_latlon(lat, lon, force_zone_number=zone_number, force_zone_letter=zone_letter)
        e0, n0, _, _ = utm.from_latlon(lat0a, lon0a, force_zone_number=zone_number, force_zone_letter=zone_letter)
        return np.asarray(e) - np.asarray(e0), np.asarray(n) - np.asarray(n0)
    raise ValueError(f"Unknown projection method {method!r}; use 'equirectangular' or 'utm'.")


def to_latlon(x, y, lat0, lon0, method: str = DEFAULT_METHOD):
    """Inverse of :func:`to_local_xy`: local ``(x_east_m, y_north_m)`` back to ``(lat, lon)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lat_ref, lon_ref, lat0a, lon0a = _ref_point(lat0, lon0)
    if method == "equirectangular":
        lat = lat0a + np.degrees(y / EARTH_RADIUS_M)
        lon = lon0a + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(lat0a))))
        return np.asarray(lat), np.asarray(lon)
    if method == "utm":
        import utm

        zone_number, zone_letter = _utm_zone(lat_ref, lon_ref)
        e0, n0, _, _ = utm.from_latlon(lat0a, lon0a, force_zone_number=zone_number, force_zone_letter=zone_letter)
        lat, lon = utm.to_latlon(np.asarray(e0) + x, np.asarray(n0) + y,
                                 zone_number, zone_letter, strict=False)
        return np.asarray(lat), np.asarray(lon)
    raise ValueError(f"Unknown projection method {method!r}; use 'equirectangular' or 'utm'.")
