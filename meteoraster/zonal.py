'''
Zonal (polygon) aggregation of gridded data.

Independent of MeteoRaster: everything takes plain latitude/longitude arrays and an
array whose last two axes are (y, x), so one production block streamed off disk can
be aggregated without building a cube. `osgeo.ogr` is imported on demand.

Properties worth knowing:

- `cell_corners` builds the corner lattice with numpy, extrapolating half a cell at
  the grid boundary, so every cell participates including the outer ring. On a
  regular grid it is identical to interpolating the cell centres; on a curvilinear
  grid it is bilinear.
- A `Coverage` is sparse (CSR-style per-zone cell indices and weights), so
  `apply_coverage` gathers the ~10^2-10^3 cells a catchment touches rather than
  weighting the whole grid. Peak memory is one block, not block x zones.
- Missing data is renormalised and the coverage threshold applied *per zone and per
  slice*, never taking one zone as representative of the others.
- Coordinates are WGS84 lon/lat of cell *centres*; KML edges are straight lines
  in lon/lat. Overlaps are cut in that plane.
- Weights default to true WGS84 areas (`area_weighting='ellipsoidal'`, exact on
  the authalic sphere). `'planar'` (square degrees, the v3.0 default) is kept as a
  fall-back. Normalising per zone does *not* cancel the cos(latitude) factor:
  planar weights within a zone are off by ~tan(lat) * lat_span (5 deg span at
  60 deg: ~15 %). `'cos_lat'` scales each overlap by its cell-centre cos(lat).
- A zone partly outside the grid is averaged over the part inside; the fraction
  inside is `Coverage.grid_fraction` and `on_partial` warns about it.

Since v3.0 MeteoRaster.get_values_from_KML and get_quantiles_from_KML both run on
this module; the coverage builder they used before got the first and third points
wrong, which is why their numbers changed. See AGENTS.md, Known rough edges.
'''

import dataclasses
import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import _lazy

#: Weight normalisations. 'area' makes each zone's weights sum to 1 (an
#: area-weighted mean); 'cell_fraction' makes each weight the fraction of that
#: cell inside the zone, in (0, 1]; 'none' keeps raw areas. Means, quantiles and
#: the coverage test are all invariant to a per-zone rescaling, so this only
#: changes how the stored weights read.
NORMALISATIONS = ('area', 'cell_fraction', 'none')

AREA_WEIGHTINGS = ('ellipsoidal', 'cos_lat', 'planar')

DEFAULT_AREA_WEIGHTING = 'ellipsoidal'

#: A zone whose grid_fraction is below 1 - this is reported by on_partial.
GRID_FRACTION_TOLERANCE = 1e-3

# WGS84 semi-major axis (m) and first eccentricity.
_WGS84_A = 6378137.0
_WGS84_E = 0.0818191908426215

#: Weighted-quantile plotting positions. See weighted_quantiles() for why
#: 'hazen' is the default rather than the numpy-'linear'-compatible option.
QUANTILE_METHODS = ('hazen', 'linear')

#: Default minimum fraction of a zone's area that must carry finite data.
MIN_COVERAGE = 0.8

_KML_NAME_TAGS = ('SimpleData', 'Data')


# --------------------------------------------------------------------------- #
# grid geometry
# --------------------------------------------------------------------------- #

def as_grid(latitudes, longitudes):
    '''
    Returns latitudes/longitudes as 2-D float arrays, meshing 1-D inputs.

    No latitude flip is applied: the caller owns row order, and it must match the
    array later handed to apply_coverage.
    '''

    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)

    if latitudes.ndim == 1 and longitudes.ndim == 1:
        longitudes, latitudes = np.meshgrid(longitudes, latitudes)

    if latitudes.shape != longitudes.shape:
        raise ValueError('latitudes %s and longitudes %s must have the same shape.'
                         % (latitudes.shape, longitudes.shape))
    if latitudes.ndim != 2:
        raise ValueError('latitudes/longitudes must be 1-D or 2-D (got %d-D).' % latitudes.ndim)
    if min(latitudes.shape) < 2:
        raise ValueError('A coverage needs at least a 2x2 grid (got %s).' % (latitudes.shape,))

    return latitudes, longitudes


def _extended(a):
    '''
    Pads a 2-D array with one ring of linearly extrapolated values.
    '''

    p = np.empty((a.shape[0] + 2, a.shape[1] + 2), dtype=float)
    p[1:-1, 1:-1] = a
    p[0, 1:-1] = 2 * a[0] - a[1]
    p[-1, 1:-1] = 2 * a[-1] - a[-2]
    p[:, 0] = 2 * p[:, 1] - p[:, 2]
    p[:, -1] = 2 * p[:, -2] - p[:, -3]

    return p


def cell_corners(latitudes, longitudes):
    '''
    Returns (corner_lat, corner_lon), each (rows+1, cols+1).

    Cell (i, j) has corners corner[i:i+2, j:j+2]. Every cell is covered,
    including the boundary ring - the half-cell extrapolation is what the old
    interpolator-based version could not do.
    '''

    latitudes, longitudes = as_grid(latitudes, longitudes)

    def corners(a):
        p = _extended(a)
        return 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])

    return corners(latitudes), corners(longitudes)


def cell_bounds(corner_lat, corner_lon):
    '''
    Returns (lat_min, lat_max, lon_min, lon_max), each (rows, cols).
    '''

    def bounds(c):
        stack = np.stack((c[:-1, :-1], c[:-1, 1:], c[1:, :-1], c[1:, 1:]))
        return stack.min(axis=0), stack.max(axis=0)

    lat_min, lat_max = bounds(corner_lat)
    lon_min, lon_max = bounds(corner_lon)

    return lat_min, lat_max, lon_min, lon_max


def cell_areas(corner_lat, corner_lon, area_weighting='planar'):
    '''
    Returns each cell's area (rows, cols) from its quad.

    'planar': square degrees (shoelace). 'ellipsoidal': km^2 on WGS84.
    '''

    x = np.stack((corner_lon[:-1, :-1], corner_lon[:-1, 1:],
                  corner_lon[1:, 1:], corner_lon[1:, :-1]))
    y = np.stack((corner_lat[:-1, :-1], corner_lat[:-1, 1:],
                  corner_lat[1:, 1:], corner_lat[1:, :-1]))

    if area_weighting == 'ellipsoidal':
        return ring_area(x, y)

    total = np.zeros(x.shape[1:], dtype=float)
    for i in range(4):
        j = (i + 1) % 4
        total += x[i] * y[j] - x[j] * y[i]

    return np.abs(total) * 0.5


def authalic_latitude(latitudes):
    '''
    Returns the WGS84 authalic latitude (degrees): equal-area latitude on a sphere.
    '''

    e = _WGS84_E
    sin_phi = np.sin(np.radians(np.clip(np.asarray(latitudes, dtype=float), -90.0, 90.0)))

    def q(s):
        return (1 - e ** 2) * (s / (1 - (e * s) ** 2)
                               - np.log((1 - e * s) / (1 + e * s)) / (2 * e))

    return np.degrees(np.arcsin(np.clip(q(sin_phi) / q(1.0), -1.0, 1.0)))


def _authalic_radius_km():
    e = _WGS84_E
    q_pole = (1 - e ** 2) * (1 / (1 - e ** 2) - np.log((1 - e) / (1 + e)) / (2 * e))

    return _WGS84_A * np.sqrt(q_pole / 2) / 1000.0


def ring_area(lon, lat):
    '''
    Area (km^2) enclosed by rings on WGS84, edges straight in lon/lat.

    lon, lat: [n_vertices, ...] degrees; the ring closes itself (a repeated
    first vertex adds a zero-length edge). Exact on the authalic sphere for edges
    linear in (lon, authalic lat), i.e. the integral of cos(beta) dbeta dlon.
    No antimeridian handling.
    '''

    lam = np.radians(np.asarray(lon, dtype=float))
    beta = np.radians(authalic_latitude(lat))

    d_lam = np.roll(lam, -1, axis=0) - lam
    beta_next = np.roll(beta, -1, axis=0)
    # (cos b1 - cos b2) / (b2 - b1), stable as b2 -> b1
    strip = np.sin(0.5 * (beta + beta_next)) * np.sinc((beta_next - beta) / (2 * np.pi))

    return np.abs((d_lam * strip).sum(axis=0)) * _authalic_radius_km() ** 2


def geometry_area(geometry):
    '''
    Area (km^2) on WGS84 of an ogr polygon, multipolygon or collection (holes subtracted).
    '''

    name = geometry.GetGeometryName()
    if name == 'POLYGON':
        total = 0.0
        for i in range(geometry.GetGeometryCount()):
            points = np.asarray(geometry.GetGeometryRef(i).GetPoints() or [], dtype=float)
            if points.shape[0] < 3:
                continue
            area = float(ring_area(points[:, 0], points[:, 1]))
            total += area if i == 0 else -area
        return total
    if name in ('MULTIPOLYGON', 'GEOMETRYCOLLECTION'):
        return sum(geometry_area(geometry.GetGeometryRef(i))
                   for i in range(geometry.GetGeometryCount()))

    return 0.0


def grid_fingerprint(latitudes, longitudes):
    '''
    Returns a sha256 hex digest of the grid geometry.

    Cache key for a stored coverage. It hashes the coordinates, not just the
    shape, so a grid that keeps its shape but shifts its geolocation - the case
    that otherwise yields plausible numbers on the wrong pixels - is a miss.
    '''

    latitudes, longitudes = as_grid(latitudes, longitudes)

    digest = hashlib.sha256()
    digest.update(b'meteoraster-grid-v1')
    digest.update(np.asarray(latitudes.shape, dtype='int64').tobytes())
    for array in (latitudes, longitudes):
        digest.update(np.ascontiguousarray(np.round(array, 9), dtype='float64').tobytes())

    return digest.hexdigest()


def _dilate(mask, rings):
    '''
    Grows a boolean mask by `rings` cells in the four cardinal directions.
    '''

    out = mask
    for _ in range(int(rings)):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown

    return out


# --------------------------------------------------------------------------- #
# the coverage
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Coverage:
    '''
    Per-zone cell indices and weights, stored CSR-style.

    zones      : labels, in payload order
    rows, cols : the grid the coverage was built for
    offsets    : len(zones)+1 offsets into cell_index / weight
    cell_index : flat (row * cols + col) indices, concatenated over zones
    weight     : matching weights, concatenated over zones
    grid_fraction : per zone, share of the polygon's area that lies on the grid
                    (None for coverages built before v3.1)
    '''

    zones: list
    rows: int
    cols: int
    offsets: np.ndarray
    cell_index: np.ndarray
    weight: np.ndarray
    centroids: pd.DataFrame = None
    normalize: str = 'area'
    area_weighting: str = 'planar'
    fingerprint: str = ''
    grid_fraction: np.ndarray = None

    def __len__(self):
        return len(self.zones)

    def __post_init__(self):
        self.zones = list(self.zones)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        self.cell_index = np.asarray(self.cell_index, dtype=np.int64)
        self.weight = np.asarray(self.weight, dtype=np.float64)
        if self.offsets.size != len(self.zones) + 1:
            raise ValueError('offsets must have len(zones)+1 entries.')
        if self.cell_index.size != self.weight.size:
            raise ValueError('cell_index and weight must have the same length.')
        if self.grid_fraction is not None:
            self.grid_fraction = np.asarray(self.grid_fraction, dtype=np.float64)
            if self.grid_fraction.size != len(self.zones):
                raise ValueError('grid_fraction must have one entry per zone.')

    @property
    def shape(self):
        return (int(self.rows), int(self.cols))

    @property
    def cell_count(self):
        return int(self.offsets[-1])

    def cells(self, zone):
        '''
        Returns (cell_index, weight) for one zone, by position.
        '''

        start, end = int(self.offsets[zone]), int(self.offsets[zone + 1])

        return self.cell_index[start:end], self.weight[start:end]

    def total_weight(self):
        '''
        Returns each zone's summed weight.
        '''

        return np.array([w.sum() for _, w in (self.cells(i) for i in range(len(self)))])

    def empty_zones(self):
        '''
        Returns the labels of zones that caught no cell.
        '''

        return [self.zones[i] for i in range(len(self)) if self.offsets[i + 1] == self.offsets[i]]

    def to_dense(self):
        '''
        Returns the legacy dense matrix, [1, 1, 1, rows, cols, n_zones] float32.
        '''

        dense = np.zeros((1, 1, 1, int(self.rows), int(self.cols), len(self)), dtype=np.single)
        flat = dense.reshape(int(self.rows) * int(self.cols), len(self))
        for zone in range(len(self)):
            index, weight = self.cells(zone)
            flat[index, zone] = weight

        return dense

    @classmethod
    def from_dense(cls, dense, zones=None, centroids=None, normalize='area',
                   area_weighting='planar', fingerprint=''):
        '''
        Builds a Coverage from the legacy dense matrix.
        '''

        dense = np.asarray(dense)
        if dense.ndim != 6:
            raise ValueError('A dense coverage must be 6-D [1, 1, 1, rows, cols, zones].')

        rows, cols, count = dense.shape[3], dense.shape[4], dense.shape[5]
        flat = dense.reshape(rows * cols, count)
        if zones is None:
            zones = [str(i) for i in range(count)]

        offsets = [0]
        indices, weights = [], []
        for zone in range(count):
            nonzero = np.flatnonzero(flat[:, zone])
            indices.append(nonzero)
            weights.append(flat[nonzero, zone].astype(np.float64))
            offsets.append(offsets[-1] + nonzero.size)

        return cls(zones=list(zones), rows=rows, cols=cols,
                   offsets=np.asarray(offsets, dtype=np.int64),
                   cell_index=(np.concatenate(indices) if indices else np.zeros(0, np.int64)),
                   weight=(np.concatenate(weights) if weights else np.zeros(0, float)),
                   centroids=centroids, normalize=normalize,
                   area_weighting=area_weighting, fingerprint=fingerprint)

    def save(self, file):
        '''
        Writes the coverage to a single .npz holding only arrays.

        No pickled objects, so it loads with allow_pickle=False on any install.
        '''

        centroids = self.centroids
        if centroids is None:
            centroids = pd.DataFrame({'x': np.full(len(self), np.nan),
                                      'y': np.full(len(self), np.nan)}, index=self.zones)

        extra = {}
        if self.grid_fraction is not None:
            extra['grid_fraction'] = self.grid_fraction

        np.savez_compressed(
            str(file),
            **extra,
            zones=np.asarray([str(z) for z in self.zones], dtype='U'),
            grid=np.asarray([self.rows, self.cols], dtype='int64'),
            offsets=self.offsets,
            cell_index=self.cell_index,
            weight=self.weight,
            centroid_x=np.asarray(centroids['x'].values, dtype=float),
            centroid_y=np.asarray(centroids['y'].values, dtype=float),
            meta=np.asarray([self.normalize, self.area_weighting, self.fingerprint], dtype='U'),
        )

    @classmethod
    def load(cls, file):
        '''
        Reads a coverage written by save().
        '''

        with np.load(str(file), allow_pickle=False) as payload:
            zones = [str(z) for z in payload['zones']]
            rows, cols = (int(v) for v in payload['grid'])
            meta = [str(v) for v in payload['meta']]
            centroids = pd.DataFrame({'x': payload['centroid_x'],
                                      'y': payload['centroid_y']}, index=zones)
            grid_fraction = None
            if 'grid_fraction' in payload.files:
                grid_fraction = payload['grid_fraction']
                centroids['grid_fraction'] = grid_fraction

            return cls(zones=zones, rows=rows, cols=cols,
                       offsets=payload['offsets'],
                       cell_index=payload['cell_index'],
                       weight=payload['weight'],
                       centroids=centroids,
                       normalize=meta[0], area_weighting=meta[1], fingerprint=meta[2],
                       grid_fraction=grid_fraction)

    def matches(self, latitudes, longitudes):
        '''
        True when this coverage was built for exactly this grid.
        '''

        latitudes, longitudes = as_grid(latitudes, longitudes)
        if latitudes.shape != self.shape:
            return False
        if not self.fingerprint:
            return True

        return self.fingerprint == grid_fingerprint(latitudes, longitudes)


# --------------------------------------------------------------------------- #
# KML reading
# --------------------------------------------------------------------------- #

def _local(tag):
    return str(tag).rsplit('}', 1)[-1]


def _iter_local(element, name):
    for child in element.iter():
        if _local(child.tag) == name:
            yield child


def _parse_coordinates(text):
    '''
    Parses a KML <coordinates> body into [(lon, lat), ...].

    Tolerates any whitespace between tuples and around the commas inside them.
    '''

    points = []
    for token in re.sub(r'\s*,\s*', ',', text or '').split():
        parts = token.split(',')
        if len(parts) < 2:
            continue
        points.append((float(parts[0]), float(parts[1])))

    return points


def _placemark_name(placemark, name_field, position):
    '''
    Returns a placemark's zone label, or None when name_field is not present.
    '''

    if name_field is None:
        return str(position)

    for tag in _KML_NAME_TAGS:
        for element in _iter_local(placemark, tag):
            if element.get('name') != name_field:
                continue
            if tag == 'SimpleData':
                if element.text and element.text.strip():
                    return element.text.strip()
            else:
                for value in _iter_local(element, 'value'):
                    if value.text and value.text.strip():
                        return value.text.strip()

    if name_field == 'name':
        for element in _iter_local(placemark, 'name'):
            if element.text and element.text.strip():
                return element.text.strip()

    return None


def _ring_polygon(ogr, points):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in points:
        ring.AddPoint_2D(float(x), float(y))
    if points[0] != points[-1]:
        ring.AddPoint_2D(float(points[0][0]), float(points[0][1]))

    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    return polygon


def _polygon_from_rings(ogr, outer, inner):
    '''
    Unions the outer rings and subtracts the inner ones.

    One Union per ring: the previous code iterated the *columns* of the parsed
    coordinate frame, so it built and unioned each identical ring three times.
    Difference, not SymmetricDifference, which was wrong for a hole not strictly
    inside the outer ring.
    '''

    polygon = None
    for points in outer:
        if len(points) < 3:
            continue
        piece = _ring_polygon(ogr, points)
        polygon = piece if polygon is None else polygon.Union(piece)

    if polygon is None:
        return None

    holes = None
    for points in inner:
        if len(points) < 3:
            continue
        piece = _ring_polygon(ogr, points)
        holes = piece if holes is None else holes.Union(piece)
    if holes is not None:
        polygon = polygon.Difference(holes)

    if hasattr(polygon, 'IsValid') and hasattr(polygon, 'MakeValid'):
        try:
            if not polygon.IsValid():
                repaired = polygon.MakeValid()
                if repaired is not None:
                    polygon = repaired
        except Exception:
            pass

    return polygon


def read_kml_polygons(kml, name_field=None, on_error='raise'):
    '''
    Reads (name, ogr.Geometry) pairs from the Placemarks of a KML file.

    Tags are matched on their local name, so a namespaced Google Earth KML parses
    as-is. The previous implementation stripped the namespace by rewriting the
    caller's KML in place (os.remove + os.rename), which failed on a read-only
    mount and mangled any path containing '.kml' in a directory name.

    name_field names an <ExtendedData> field (<SimpleData name=...> or
    <Data name=...><value>); 'name' also matches the placemark's <name> element.
    With name_field=None, zones are labelled by placemark position.

    on_error: 'raise' (default) reports every unreadable placemark in one error;
    'warn' and 'skip' drop them, which is what the old code did silently.
    '''

    import xml.etree.ElementTree as et

    if on_error not in ('raise', 'warn', 'skip'):
        raise ValueError("on_error must be 'raise', 'warn' or 'skip'.")

    ogr = _lazy.ogr()
    root = et.parse(str(kml)).getroot()

    named, failures = [], []
    placemarks = list(_iter_local(root, 'Placemark'))
    for position, placemark in enumerate(placemarks):
        try:
            name = _placemark_name(placemark, name_field, position)
            if name is None:
                raise KeyError('no field %r' % name_field)

            outer, inner = [], []
            for polygon in _iter_local(placemark, 'Polygon'):
                for boundary in _iter_local(polygon, 'outerBoundaryIs'):
                    outer.extend(_parse_coordinates(c.text)
                                 for c in _iter_local(boundary, 'coordinates'))
                for boundary in _iter_local(polygon, 'innerBoundaryIs'):
                    inner.extend(_parse_coordinates(c.text)
                                 for c in _iter_local(boundary, 'coordinates'))

            geometry = _polygon_from_rings(ogr, outer, inner)
            if geometry is None:
                raise ValueError('no polygon geometry')

            named.append((name, geometry))
        except Exception as ex:
            failures.append('placemark %d: %s: %s' % (position, type(ex).__name__, ex))

    if failures:
        message = ('Could not read %d of %d placemarks in %s:\n  %s'
                   % (len(failures), len(placemarks), kml, '\n  '.join(failures)))
        if on_error == 'raise':
            raise ValueError(message + '\n(pass on_error="skip" to ignore them)')
        if on_error == 'warn':
            warnings.warn(message)

    if not named:
        raise ValueError('No usable Placemark polygons found in %s%s.'
                         % (kml, '' if name_field is None else ' for field %r' % name_field))

    labels = [name for name, _ in named]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        message = ('Zone names are not unique: %s. Each zone becomes a column, so '
                   'duplicates cannot be told apart.' % ', '.join(duplicates))
        if on_error == 'raise':
            raise ValueError(message)
        warnings.warn(message)

    return named


# --------------------------------------------------------------------------- #
# building a coverage
# --------------------------------------------------------------------------- #

def _as_geometry(ogr, geometry):
    '''
    Accepts an ogr.Geometry, WKB bytes or a WKT string.

    The bytes path is what lets PostGIS geometry feed straight in.
    '''

    if geometry is None:
        raise ValueError('geometry is None')
    if isinstance(geometry, (bytes, bytearray, memoryview)):
        return ogr.CreateGeometryFromWkb(bytes(geometry))
    if isinstance(geometry, str):
        return ogr.CreateGeometryFromWkt(geometry)
    if hasattr(geometry, 'GetEnvelope'):
        return geometry
    if hasattr(geometry, 'wkb'):
        return ogr.CreateGeometryFromWkb(bytes(geometry.wkb))
    if hasattr(geometry, 'wkt'):
        return ogr.CreateGeometryFromWkt(str(geometry.wkt))

    raise TypeError('Cannot interpret %r as a geometry.' % type(geometry).__name__)


def coverage_from_geometries(latitudes, longitudes, geometries, names=None,
                             halo=1, buffer=None, normalize='area',
                             area_weighting=DEFAULT_AREA_WEIGHTING, min_area=0.0,
                             on_empty='warn', on_partial='warn'):
    '''
    Builds a Coverage by exact fractional overlap of each cell with each polygon.

    geometries: ogr.Geometry, WKB bytes or WKT strings, WGS84 lon/lat.
    area_weighting: 'ellipsoidal' (km^2 on WGS84, default), 'cos_lat', or
                'planar' (square degrees; reproduces v3.0).
    min_area:   overlaps at or below this are dropped, in the units of the
                overlap before normalisation (square degrees for 'planar' and
                'cos_lat', km^2 for 'ellipsoidal').
    halo:       extra rings of candidate cells. The envelope prefilter is already
                conservative, so this is only numerical insurance; it is cheap
                because non-overlapping candidates contribute zero area and are
                dropped. The old code buffered the polygon envelope by a
                hard-coded 2 *degrees*, which at 0.04 deg pads every catchment by
                +/-50 cells and turned a 26-zone layer into ~10^5-10^6 OGR
                intersections where ~10^4 suffice.
    buffer:     legacy envelope padding in degrees, applied in addition to halo.
    on_empty:   what to do about a zone that caught no cell ('warn', 'raise',
                'ignore'). It used to be a ValueError from min() of an empty array.
    on_partial: what to do about a zone partly outside the grid ('warn', 'raise',
                'ignore'). Its mean is taken over the part inside, renormalised,
                so it looks plausible; `grid_fraction` says how much is inside.
    '''

    if normalize not in NORMALISATIONS:
        raise ValueError('normalize must be one of %s.' % ', '.join(NORMALISATIONS))
    if area_weighting not in AREA_WEIGHTINGS:
        raise ValueError('area_weighting must be one of %s.' % ', '.join(AREA_WEIGHTINGS))
    if on_empty not in ('warn', 'raise', 'ignore'):
        raise ValueError("on_empty must be 'warn', 'raise' or 'ignore'.")
    if on_partial not in ('warn', 'raise', 'ignore'):
        raise ValueError("on_partial must be 'warn', 'raise' or 'ignore'.")

    ogr = _lazy.ogr()
    latitudes, longitudes = as_grid(latitudes, longitudes)
    rows, cols = latitudes.shape

    corner_lat, corner_lon = cell_corners(latitudes, longitudes)
    lat_min, lat_max, lon_min, lon_max = cell_bounds(corner_lat, corner_lon)
    areas = None
    if normalize == 'cell_fraction':
        areas = cell_areas(corner_lat, corner_lon,
                           'ellipsoidal' if area_weighting == 'ellipsoidal' else 'planar')

    scale = None
    if area_weighting == 'cos_lat':
        scale = np.cos(np.radians(np.clip(latitudes, -89.999, 89.999)))

    geometries = list(geometries)
    if names is None:
        names = [str(i) for i in range(len(geometries))]
    names = [str(n) for n in names]
    if len(names) != len(geometries):
        raise ValueError('Got %d names for %d geometries.' % (len(names), len(geometries)))

    offsets = [0]
    all_index, all_weight = [], []
    centroid_x, centroid_y = [], []
    grid_fraction = []
    empty, wide = [], []

    pad = 0.0 if buffer is None else float(buffer)

    for name, raw in zip(names, geometries):
        polygon = _as_geometry(ogr, raw)
        west, east, south, north = polygon.GetEnvelope()
        if east - west > 180:
            wide.append(name)
        polygon_area = geometry_area(polygon)
        inside_area = 0.0

        # Conservative: a cell can only intersect the polygon if its bounding box
        # overlaps the polygon's envelope.
        candidates = ((lon_min <= east + pad) & (lon_max >= west - pad) &
                      (lat_min <= north + pad) & (lat_max >= south - pad))
        if halo:
            candidates = _dilate(candidates, halo)

        centroid = polygon.Centroid()
        if centroid is not None and not centroid.IsEmpty():
            x, y, *_ = centroid.GetPoint()
        else:
            x, y = np.nan, np.nan
        centroid_x.append(x)
        centroid_y.append(y)

        index, weight = [], []
        for row, col in zip(*np.where(candidates)):
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint_2D(float(corner_lon[row, col]), float(corner_lat[row, col]))
            ring.AddPoint_2D(float(corner_lon[row, col + 1]), float(corner_lat[row, col + 1]))
            ring.AddPoint_2D(float(corner_lon[row + 1, col + 1]), float(corner_lat[row + 1, col + 1]))
            ring.AddPoint_2D(float(corner_lon[row + 1, col]), float(corner_lat[row + 1, col]))
            ring.AddPoint_2D(float(corner_lon[row, col]), float(corner_lat[row, col]))

            cell = ogr.Geometry(ogr.wkbPolygon)
            cell.AddGeometry(ring)

            overlap = cell.Intersection(polygon)
            if overlap is None or overlap.IsEmpty():
                continue
            # A cell that only touches the polygon along an edge intersects as a
            # line or a point; Area() would warn on stdout and return 0 anyway.
            if overlap.GetDimension() != 2:
                continue
            true_area = geometry_area(overlap)
            inside_area += true_area
            area = true_area if area_weighting == 'ellipsoidal' else overlap.Area()
            if area <= min_area:
                continue

            if normalize == 'cell_fraction':
                cell_area = areas[row, col]
                area = area / cell_area if cell_area > 0 else 0.0
            if scale is not None:
                area = area * scale[row, col]

            index.append(row * cols + col)
            weight.append(area)

        if not index:
            empty.append(name)
        grid_fraction.append(min(inside_area / polygon_area, 1.0) if polygon_area > 0 else 0.0)

        weight = np.asarray(weight, dtype=float)
        if normalize == 'area' and weight.size and weight.sum() > 0:
            weight = weight / weight.sum()

        all_index.append(np.asarray(index, dtype=np.int64))
        all_weight.append(weight)
        offsets.append(offsets[-1] + weight.size)

    if empty:
        message = ('%d zone(s) caught no grid cell: %s. They are outside the grid or '
                   'smaller than the cell tolerance, and will aggregate to NaN.'
                   % (len(empty), ', '.join(empty)))
        if on_empty == 'raise':
            raise ValueError(message)
        if on_empty == 'warn':
            warnings.warn(message)

    partial = ['%s (%.1f%%)' % (name, 100 * fraction) for name, fraction in zip(names, grid_fraction)
               if name not in empty and fraction < 1 - GRID_FRACTION_TOLERANCE]
    if partial and on_partial != 'ignore':
        message = ('%d zone(s) lie partly outside the grid: %s (share inside). Their means '
                   'cover only that share.' % (len(partial), ', '.join(partial)))
        if on_partial == 'raise':
            raise ValueError(message)
        warnings.warn(message)

    if wide:
        warnings.warn('%d zone(s) span more than 180 deg of longitude: %s. Polygons crossing '
                      'the antimeridian are not supported; split them at +/-180.'
                      % (len(wide), ', '.join(wide)))

    centroids = pd.DataFrame({'x': centroid_x, 'y': centroid_y,
                              'grid_fraction': grid_fraction}, index=names)

    return Coverage(
        zones=names, rows=rows, cols=cols,
        offsets=np.asarray(offsets, dtype=np.int64),
        cell_index=(np.concatenate(all_index) if all_index else np.zeros(0, np.int64)),
        weight=(np.concatenate(all_weight) if all_weight else np.zeros(0, float)),
        centroids=centroids, normalize=normalize, area_weighting=area_weighting,
        fingerprint=grid_fingerprint(latitudes, longitudes),
        grid_fraction=np.asarray(grid_fraction, dtype=float),
    )


def coverage_from_kml(kml, latitudes, longitudes, name_field=None, on_error='raise',
                      **kwargs):
    '''
    Builds a Coverage from the Placemark polygons of a KML file.
    '''

    named = read_kml_polygons(kml, name_field=name_field, on_error=on_error)

    return coverage_from_geometries(latitudes, longitudes,
                                    [geometry for _, geometry in named],
                                    names=[name for name, _ in named], **kwargs)


# --------------------------------------------------------------------------- #
# applying a coverage
# --------------------------------------------------------------------------- #

def _flatten_grid(coverage, values):
    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError('values must have at least the two grid axes.')
    if values.shape[-2:] != coverage.shape:
        raise ValueError('values end in %s but the coverage was built for %s.'
                         % (values.shape[-2:], coverage.shape))

    return values.reshape(values.shape[:-2] + (coverage.rows * coverage.cols,))


def apply_coverage(coverage, values, min_coverage=MIN_COVERAGE, dtype=None):
    '''
    Area-weighted mean of `values` over each zone.

    values: any array whose last two axes are (y, x) - a 5-D cube, one
            production block [member, leadtime, y, x], or [leadtime, y, x].
    returns: the same leading axes with (y, x) replaced by a zone axis.

    Weights are renormalised over the cells that carry finite data, and a slice
    is only reported when at least `min_coverage` of the zone's weight has data -
    both per zone, unlike the old code which took zone 0 as representative.
    '''

    values = np.asarray(values)
    flat = _flatten_grid(coverage, values)

    if dtype is None:
        dtype = np.result_type(values.dtype, np.float32)
    out = np.full(values.shape[:-2] + (len(coverage),), np.nan, dtype=dtype)

    for zone in range(len(coverage)):
        index, weight = coverage.cells(zone)
        if index.size == 0:
            continue
        total = weight.sum()
        if not total > 0:
            continue

        sample = flat[..., index]
        finite = np.isfinite(sample)
        numerator = np.where(finite, sample, 0.0) @ weight
        denominator = finite @ weight

        with np.errstate(invalid='ignore', divide='ignore'):
            mean = numerator / denominator
        out[..., zone] = np.where((denominator >= min_coverage * total) & (denominator > 0),
                                  mean, np.nan)

    return out


# --------------------------------------------------------------------------- #
# weighted quantiles
# --------------------------------------------------------------------------- #

def _first_usable(usable):
    '''
    Index of the first True along the last axis (0 when there is none).
    '''

    return np.where(usable.any(axis=-1), np.argmax(usable, axis=-1), 0)


def _last_usable(usable):
    '''
    Index of the last True along the last axis (0 when there is none).
    '''

    reversed_position = np.argmax(usable[..., ::-1], axis=-1)

    return np.where(usable.any(axis=-1), usable.shape[-1] - 1 - reversed_position, 0)


def weighted_quantiles(values, weights, quantiles, method='hazen'):
    '''
    Quantiles of a weighted sample along the last axis.

    values:  [..., n]; non-finite entries are dropped (they carry no weight)
    weights: [n] or anything broadcastable to values.shape
    returns: [..., len(quantiles)]

    This replaces the "unfold each cell round(w/precision) times and take a plain
    nanquantile" approach, which was not only 10x the memory but lossy: at
    precision=0.1 a cell with 4% coverage dropped out of the sample entirely and
    14% counted the same as 10%.

    method 'hazen' (the default) puts each value at the centre of its share of the
    cumulative weight, p_i = (cw_i - w_i/2) / W. Two properties make it the right
    default: with equal weights it is exactly np.quantile(..., method='hazen'),
    and with integer weights each position lands exactly on the centre of that
    value's block in the replicated sample - so it agrees with what the old
    "unfold" trick was reaching for, without the quantisation.

    method 'linear' rescales those positions so the first is 0 and the last is 1,
    which reproduces np.quantile(..., method='linear') exactly for equal weights.
    Be aware that any endpoint-anchored rule loses weight sensitivity as the
    sample shrinks - with only two distinct values it can only ever return their
    midpoint for the median, whatever the weights - which is why it is not the
    default.

    Both clamp: q=0 returns the smallest value with weight and q=1 the largest.
    '''

    if method not in QUANTILE_METHODS:
        raise ValueError('method must be one of %s.' % ', '.join(QUANTILE_METHODS))

    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        raise ValueError('values must have a sample axis.')
    weights = np.broadcast_to(np.asarray(weights, dtype=float), values.shape)
    probabilities = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('quantiles must lie in [0, 1].')

    order = np.argsort(values, axis=-1, kind='stable')
    sorted_values = np.take_along_axis(values, order, axis=-1)
    sorted_weights = np.take_along_axis(weights, order, axis=-1)

    usable = np.isfinite(sorted_values) & np.isfinite(sorted_weights) & (sorted_weights > 0)
    sorted_weights = np.where(usable, sorted_weights, 0.0)

    # Forward-fill the tail: argsort puts non-finite values last, and clamping the
    # interpolation can land there. Values are ascending, so a running maximum
    # makes those slots repeat the largest usable value.
    filled = np.maximum.accumulate(np.where(usable, sorted_values, -np.inf), axis=-1)

    cumulative = np.cumsum(sorted_weights, axis=-1)
    total = cumulative[..., -1:]

    with np.errstate(invalid='ignore', divide='ignore'):
        positions = (cumulative - 0.5 * sorted_weights) / total

        if method == 'linear':
            # Anchor the first and last usable positions at 0 and 1. With equal
            # weights (i - 0.5)/n becomes (i - 1)/(n - 1), i.e. numpy's 'linear'.
            first = np.take_along_axis(positions, _first_usable(usable)[..., None], axis=-1)
            last = np.take_along_axis(positions, _last_usable(usable)[..., None], axis=-1)
            span = last - first
            positions = np.where(span > 0, (positions - first) / span, 0.0)

    # A zero-weight slot leaves the cumulative sum unchanged, so it repeats the
    # previous position rather than breaking monotonicity, and `filled` already
    # carries the previous usable value there. Nothing further is needed; a
    # non-finite position only happens when no weight is usable at all, which the
    # degenerate branch below turns into NaN.
    positions = np.where(np.isfinite(positions), positions, 0.0)

    count = usable.sum(axis=-1)
    result = np.empty(values.shape[:-1] + (probabilities.size,), dtype=float)

    last_index = values.shape[-1] - 1
    for i, probability in enumerate(probabilities):
        # Bracket the requested probability. Clamping instead of extrapolating is
        # what makes q=0 the smallest weighted value and q=1 the largest.
        upper = np.sum(positions < probability, axis=-1)
        upper = np.clip(upper, min(1, last_index), last_index)
        lower = np.maximum(upper - 1, 0)

        p_low = np.take_along_axis(positions, lower[..., None], axis=-1)[..., 0]
        p_high = np.take_along_axis(positions, upper[..., None], axis=-1)[..., 0]
        v_low = np.take_along_axis(filled, lower[..., None], axis=-1)[..., 0]
        v_high = np.take_along_axis(filled, upper[..., None], axis=-1)[..., 0]

        with np.errstate(invalid='ignore', divide='ignore'):
            gap = p_high - p_low
            fraction = np.where(gap > 0, (probability - p_low) / gap, 0.0)
            fraction = np.clip(np.where(np.isfinite(fraction), fraction, 0.0), 0.0, 1.0)
            result[..., i] = np.where(np.isfinite(v_low) & np.isfinite(v_high),
                                      v_low + fraction * (v_high - v_low), v_low)

    # Degenerate samples: one usable value means every quantile is that value,
    # none means the whole slice is missing.
    single = count == 1
    if single.any():
        only = filled[..., -1]
        result[single] = only[single][..., None]
    none = count == 0
    if none.any():
        result[none] = np.nan

    return result


def quantiles_from_coverage(coverage, values, quantiles, pool_axes=(),
                            min_coverage=MIN_COVERAGE, method='hazen'):
    '''
    Area-weighted spatial quantiles per zone.

    values:    [..., y, x]
    pool_axes: leading axes folded into the sample alongside the zone's cells,
               e.g. the leadtime axis to pool an hourly forecast into daily
               distributions.
    returns:   the kept leading axes + (len(quantiles), n_zones).

    A cell half inside a zone contributes half the probability mass, which is
    what the coverage weights mean. The coverage test is area-weighted, matching
    apply_coverage - the old code counted cells instead.
    '''

    values = np.asarray(values)
    flat = _flatten_grid(coverage, values)
    leading = values.ndim - 2

    pool = tuple(sorted({int(axis) % leading for axis in pool_axes}))
    keep = tuple(axis for axis in range(leading) if axis not in pool)

    # [keep..., pool..., cells]
    flat = np.transpose(flat, keep + pool + (leading,))
    keep_shape = tuple(values.shape[axis] for axis in keep)
    pool_size = 1
    for axis in pool:
        pool_size *= int(values.shape[axis])

    probabilities = np.atleast_1d(np.asarray(quantiles, dtype=float))
    out = np.full(keep_shape + (probabilities.size, len(coverage)), np.nan, dtype=float)

    for zone in range(len(coverage)):
        index, weight = coverage.cells(zone)
        if index.size == 0:
            continue
        total = weight.sum()
        if not total > 0:
            continue

        sample = flat[..., index].reshape(keep_shape + (pool_size * index.size,))
        sample_weight = np.broadcast_to(weight, flat[..., index].shape)
        sample_weight = sample_weight.reshape(sample.shape)

        # Area-weighted share of the pooled sample that carries data.
        finite = np.isfinite(sample)
        available = (finite * sample_weight).sum(axis=-1) / (pool_size * total)

        computed = weighted_quantiles(sample, sample_weight, probabilities, method=method)
        out[..., zone] = np.where((available >= min_coverage)[..., None], computed, np.nan)

    return out
