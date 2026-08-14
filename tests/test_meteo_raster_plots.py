from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Plotting is optional, so the whole module is skipped when it is not installed -
# otherwise collection fails with ModuleNotFoundError and takes the session down,
# which is exactly what the minimal CI job installs.
plt = pytest.importorskip('matplotlib.pyplot', reason='matplotlib is not installed')
ccrs = pytest.importorskip('cartopy.crs', reason='cartopy is not installed')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meteoraster import MeteoRaster

from conftest import require_resource

def test_create_plot(sample_meteoraster):
    mr = sample_meteoraster
    # Test with default arguments
    ax = mr.create_plot(central_longitude=0)
    assert ax is not None
    # Check if it has a projection
    assert hasattr(ax, 'projection')
    plt.close()

    # Test with custom projection
    ax = mr.create_plot(central_longitude=0, projection_fun=ccrs.Robinson)
    # Check if the projection is correct type (instance check might be tricky with cartopy objects, let's just check creation)
    assert isinstance(ax.projection, ccrs.Robinson)
    plt.close()

def test_plot_mean(sample_meteoraster):
    mr = sample_meteoraster
    
    # Test return type
    ax, cbar = mr.plot_mean(block=False)
    assert ax is not None
    assert cbar is not None
    plt.close()
    
    # Test without colorbar
    ax, cbar = mr.plot_mean(colorbar=False, block=False)
    assert ax is not None
    assert cbar is None
    plt.close()

def test_plot_coordinates(sample_meteoraster):
    mr = sample_meteoraster
    
    # Just ensure it runs without error
    try:
        mr.plot_coordinates(block=False)
    except Exception as e:
        pytest.fail(f"plot_coordinates raised an exception: {e}")
    plt.close()

def test_plot_seasonal(sample_meteoraster):
    mr = sample_meteoraster
    lat = 35.0
    lon = 0.0
    
    # Ensure it returns an axes object
    ax = mr.plot_seasonal(lat=lat, lon=lon, block=False)
    assert ax is not None
    plt.close()

def test_plot_availability(sample_meteoraster):
    mr = sample_meteoraster
    
    # Test plotting availability
    try:
        mr.plot_availability()
    except Exception as e:
        pytest.fail(f"plot_availability raised an exception: {e}")
    plt.close()

    # Test with individualMembers=True
    try:
        mr.plot_availability(individualMembers=True)
    except Exception as e:
        pytest.fail(f"plot_availability(individualMembers=True) raised an exception: {e}")
    plt.close()

def test_plot_mean_projected(sample_meteoraster):
    mr = sample_meteoraster
    
    # Ensure it runs
    try:
        mr.plot_mean_projected(block=False)
    except Exception as e:
        pytest.fail(f"plot_mean_projected raised an exception: {e}")
    plt.close()

def test_real_raster_read_and_plot():
    '''
    The only test that plots real data rather than the synthetic fixture. It replaces
    the GLEAM one, which carried a 6.2 MB resource and its own ad-hoc reader.
    '''

    file = require_resource('tethys_NOAA_GFS_0.25_2026.05.18.nct')
    forecast = MeteoRaster.load(file, verbose=False)

    assert forecast.units == 'C'
    assert forecast.data.shape[0] > 1                  # several productions

    ax = forecast.create_plot(central_longitude=5)
    ax, cbar = forecast.plot_mean(
        ax=ax,
        coastline=True,
        borders=True,
        colorbar=True,
        cmap='viridis',
        central_longitude=5,
        central_latitude=50,
        block=False,
    )

    assert ax is not None
    assert cbar is not None
    plt.close()


def test_add_shapefile(sample_meteoraster, sample_shapefile):
    mr = sample_meteoraster
    
    # Create a plot first
    # plot_mean returns (ax, cbar) tuple
    ax, cbar = mr.plot_mean(central_longitude=0)
    
    # Test adding shapefile
    try:
        mr.add_shapefile(ax, str(sample_shapefile))
    except Exception as e:
        pytest.fail(f"add_shapefile raised an exception: {e}")
        
    try:
        mr.add_shape(ax, str(sample_shapefile))
    except Exception as e:
        pytest.fail(f"add_shape raised an exception: {e}")
        
    plt.close()

