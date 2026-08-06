from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meteoraster import MeteoRaster

RESOURCES = Path(__file__).resolve().parent / "resources"


def read_gleam(file):
    data = {}
    with xr.open_dataset(file) as ds:
        data['latitudes'] = ds.lat.data
        data['longitudes'] = ds.lon.data
        data['production_datetime'] = ds.time.data
        data['leadtimes'] = np.array([pd.DateOffset(years=0)])
        tmp = ds['E'][:, :, :].data
        data['data'] = np.expand_dims(tmp, [1, 2])
    return MeteoRaster(data, units='mm/year', variable='AET')

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

def test_gleam_read_and_plot():
    gleam_file = RESOURCES / "E_2016_GLEAM_v4.2b_YR.nc"
    aet = read_gleam(gleam_file)

    assert aet is not None
    assert aet.variable == 'AET'
    assert aet.units == 'mm/year'

    ax = aet.create_plot(central_longitude=20)
    ax, cbar = aet.plot_mean(
        ax=ax,
        coastline=True,
        borders=True,
        colorbar=True,
        cmap='viridis',
        central_longitude=20,
        central_latitude=0,
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

