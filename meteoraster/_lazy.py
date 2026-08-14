'''
On-demand imports for the optional heavy dependencies.

GDAL/OGR, matplotlib, seaborn and cartopy are imported through here instead of at
module scope so that `import meteoraster` - and therefore save/load, the header
readers and the streaming readers - works in a slim container that has none of
them. Before this, `from osgeo import ogr` at the top of meteoraster.py meant a
pip-only install could not import the package at all, even though GDAL is
deliberately not a pip dependency.

Each accessor raises an ImportError naming what to install, at the point of use.
'''

import importlib


def _load(module, requirement, install):
    try:
        return importlib.import_module(module)
    except ImportError as ex:
        raise ImportError('%s requires %s, which is not installed.\n    %s'
                          % (requirement, module.split('.')[0], install)) from ex


def ogr():
    '''
    osgeo.ogr - vector geometry, needed by the KML/zonal helpers.
    '''

    return _load('osgeo.ogr', 'Zonal aggregation from vector geometry',
                 'GDAL is a system library pip cannot install reliably; use '
                 'conda: conda install -c conda-forge gdal')


def pyplot():
    '''
    matplotlib.pyplot. Set MPLBACKEND=Agg for a headless container.
    '''

    return _load('matplotlib.pyplot', 'Plotting', 'pip install matplotlib')


def seaborn():
    '''
    seaborn - statistical plots.
    '''

    return _load('seaborn', 'Plotting', 'pip install seaborn')


def ccrs():
    '''
    cartopy.crs - map projections.
    '''

    return _load('cartopy.crs', 'Map plotting', 'pip install cartopy')


def cfeature():
    '''
    cartopy.feature - coastlines and borders. Downloads Natural Earth data on
    first use, so it needs network access on a cold cache.
    '''

    return _load('cartopy.feature', 'Map plotting', 'pip install cartopy')


def shapereader():
    '''
    cartopy.io.shapereader - shapefile reading for map overlays.
    '''

    return _load('cartopy.io.shapereader', 'Map plotting', 'pip install cartopy')
