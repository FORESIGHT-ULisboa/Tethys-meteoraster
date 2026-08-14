from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shutil
import tempfile
import os
import zipfile
import numpy as np
import pandas as pd
import pytest

from meteoraster import MeteoRaster

# osgeo is imported inside the fixtures that need it, not here: importing it at
# module scope made the whole suite hard-error on an install without GDAL, which
# is exactly the slim-container case the package is supposed to support.
gdal_required = pytest.mark.skipif(
    __import__('importlib').util.find_spec('osgeo') is None,
    reason='GDAL/osgeo is not installed (conda-only dependency)',
)


# --------------------------------------------------------------------------- #
# real-data fixtures
# --------------------------------------------------------------------------- #

#: Real GRIB/NetCDF inputs. They are large, so a clone may not carry all of them;
#: every test that reads one goes through require_resource() and skips by name.
RESOURCES = Path(__file__).resolve().parent / 'resources'


def require_resource(name):
    '''
    Path to a file in tests/resources, skipping the test when it is absent.
    '''

    path = RESOURCES / name
    if not path.exists():
        pytest.skip('test resource %s is not present in %s' % (name, RESOURCES))

    return path


def _grib_unavailable():
    '''
    Why GRIB cannot be read here, or None when it can.

    Not pytest.importorskip: cfgrib raises RuntimeError - not an ImportError - when
    the ecCodes binary is unreachable, and importorskip only catches ImportError
    subclasses, so a module-level importorskip('cfgrib') would fail *collection*.
    On Windows the ecCodes DLL lives in <env>/Library/bin and findlibs only sees it
    through PATH, so the conda environment must be activated, not just its
    python.exe called directly.
    '''

    try:
        import cfgrib  # noqa: F401
    except Exception as ex:
        return '%s: %s' % (type(ex).__name__, ex)

    return None


_GRIB_UNAVAILABLE = _grib_unavailable()

grib_required = pytest.mark.skipif(
    _GRIB_UNAVAILABLE is not None,
    reason='cfgrib/ecCodes cannot read GRIB here (%s)' % _GRIB_UNAVAILABLE,
)


@pytest.fixture(scope='session')
def extracted_grib(tmp_path_factory):
    '''
    Extracts a GRIB fixture from its zip and returns the path to the member.

    The readers take a filesystem path and cfgrib cannot read from memory, so the
    single `data.grib` member is unpacked once per session under a name derived
    from the zip.
    '''

    directory = tmp_path_factory.mktemp('grib')
    unpacked = {}

    def extract(zip_name):
        if zip_name not in unpacked:
            archive = require_resource(zip_name)
            with zipfile.ZipFile(archive) as handle:
                member = handle.infolist()[0]
                target = directory / (Path(zip_name).stem + '.grib')
                with handle.open(member) as source, open(target, 'wb') as sink:
                    shutil.copyfileobj(source, sink)
            unpacked[zip_name] = target

        return unpacked[zip_name]

    return extract


@pytest.fixture
def sample_meteoraster():
    # Create a 5D data array
    # Dimensions: [production_datetime, ensemble_member, leadtime, y, x]
    # Shape: [2, 3, 2, 4, 5]
    n_prod = 2
    n_ens = 3
    n_lead = 2
    n_lat = 4
    n_lon = 5
    
    data = np.random.rand(n_prod, n_ens, n_lead, n_lat, n_lon)
    
    # Latitudes and longitudes
    lats = np.linspace(40, 30, n_lat)
    lons = np.linspace(-10, 10, n_lon)
    latitudes, longitudes = np.meshgrid(lats, lons)
    latitudes = latitudes.T
    longitudes = longitudes.T

    # Production dates
    production_datetime = pd.to_datetime(["2023-01-01", "2023-01-02"]).values
    
    # Leadtimes
    leadtimes = np.array([pd.Timedelta(days=1), pd.Timedelta(days=2)])

    mr = MeteoRaster(
        data=data,
        latitudes=lats,
        longitudes=lons,
        production_datetime=production_datetime,
        leadtimes=leadtimes,
        units="mm",
        variable="precip",
        verbose=False
    )
    return mr

@pytest.fixture
def sample_kml():
    # Remove XML namespace to avoid ElementTree finding quirks in the code under test
    kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml>
  <Document>
    <Placemark>
      <ExtendedData>
        <SchemaData schemaUrl="#test">
            <SimpleData name="zone_id">ZoneA</SimpleData>
        </SchemaData>
      </ExtendedData>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>-2.0,32.0,0 2.0,32.0,0 2.0,38.0,0 -2.0,38.0,0 -2.0,32.0,0</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
    <Placemark>
      <ExtendedData>
        <SchemaData schemaUrl="#test">
            <SimpleData name="zone_id">ZoneB</SimpleData>
        </SchemaData>
      </ExtendedData>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>-1.0,33.0,0 1.0,33.0,0 1.0,37.0,0 -1.0,37.0,0 -1.0,33.0,0</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>
"""
    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".kml", delete=False, mode='w') as tmp:
        tmp.write(kml_content)
        tmp_path = Path(tmp.name)
    
    yield tmp_path
    
    if tmp_path.exists():
        os.remove(tmp_path)

@pytest.fixture
def sample_shapefile():
    ogr = pytest.importorskip('osgeo.ogr')
    osr = pytest.importorskip('osgeo.osr')

    # Create a temporary directory for shapefile components
    tmp_dir = tempfile.mkdtemp()
    shp_path = os.path.join(tmp_dir, "test_shape.shp")

    # Create driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(shp_path)

    # Create spatial reference (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # Create layer
    layer = data_source.CreateLayer("test_shape", srs, ogr.wkbPolygon)

    # Add a field
    field_defn = ogr.FieldDefn("name", ogr.OFTString)
    field_defn.SetWidth(24)
    layer.CreateField(field_defn)

    # Create feature
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("name", "TestPolygon")

    # Create geometry
    # A simple square
    wkt = "POLYGON ((-1 33, 1 33, 1 37, -1 37, -1 33))"
    poly = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(poly)

    # Create the feature in the layer
    layer.CreateFeature(feature)

    # cleanup
    feature = None
    data_source = None
    
    yield Path(shp_path)

    # Cleanup directory
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# a grid whose zonal weights are computable on paper
# --------------------------------------------------------------------------- #

#: 3x3 grid of 1-degree cells, north-first. Cell (i, j) spans
#: lon [j-0.5, j+0.5] and lat [2-i-0.5, 2-i+0.5], so cell corners run
#: -0.5 .. 2.5 in both directions and every cell has area exactly 1.
GRID_LATS = np.array([2.0, 1.0, 0.0])
GRID_LONS = np.array([0.0, 1.0, 2.0])

#: Rings covering exactly known cells of that grid.
RING_WHOLE = '-0.5,-0.5,0 2.5,-0.5,0 2.5,2.5,0 -0.5,2.5,0 -0.5,-0.5,0'   # all 9
RING_MIDDLE = '0.5,0.5,0 1.5,0.5,0 1.5,1.5,0 0.5,1.5,0 0.5,0.5,0'        # cell (1,1)
RING_WEST_HALF = '-0.5,1.5,0 0.0,1.5,0 0.0,2.5,0 -0.5,2.5,0 -0.5,1.5,0'  # west half of (0,0)
RING_ONE_AND_QUARTER = ('-0.5,1.5,0 0.75,1.5,0 0.75,2.5,0 -0.5,2.5,0 '
                        '-0.5,1.5,0')                                    # (0,0) + 1/4 of (0,1)


def write_kml(directory, zones, name_field='zone_id', namespaced=True):
    '''
    Writes a KML with one Placemark per (name, ring) pair.

    namespaced=True declares xmlns, which is what Google Earth and QGIS emit.
    '''

    root = '<kml xmlns="http://www.opengis.net/kml/2.2">' if namespaced else '<kml>'
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', root, '<Document>']
    for name, ring in zones:
        lines.append(
            '<Placemark><ExtendedData><SchemaData>'
            '<SimpleData name="%s">%s</SimpleData></SchemaData></ExtendedData>'
            '<Polygon><outerBoundaryIs><LinearRing><coordinates>%s'
            '</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>'
            % (name_field, name, ring))
    lines += ['</Document>', '</kml>', '']

    path = Path(directory) / 'zones.kml'
    path.write_text('\n'.join(lines), encoding='utf-8')

    return path


@pytest.fixture
def grid_kml(tmp_path):
    '''
    A namespaced KML with two zones on the 3x3 grid: the whole grid and cell (1,1).
    '''

    return write_kml(tmp_path, [('whole', RING_WHOLE), ('middle', RING_MIDDLE)])


@pytest.fixture
def grid_raster():
    '''
    A MeteoRaster on the 3x3 grid: 2 productions x 2 members x 3 hourly leadtimes.

    The field is 10 everywhere except a 1 in the centre cell, so the area-weighted
    mean over the whole grid (9.0) differs from the centre value - which is what
    makes a coverage that silently drops the border visible.
    '''

    field = np.array([[10.0, 10.0, 10.0], [10.0, 1.0, 10.0], [10.0, 10.0, 10.0]])
    data = np.empty((2, 2, 3, 3, 3), dtype=np.float32)
    for production in range(2):
        for member in range(2):
            for leadtime in range(3):
                data[production, member, leadtime] = field * (leadtime + 1)

    return MeteoRaster(
        data=data,
        latitudes=GRID_LATS, longitudes=GRID_LONS,
        production_datetime=pd.to_datetime(['2024-01-01', '2024-01-02']).values,
        leadtimes=np.array([pd.Timedelta(hours=h) for h in range(3)]),
        units='mm', variable='precip', verbose=False,
    )
