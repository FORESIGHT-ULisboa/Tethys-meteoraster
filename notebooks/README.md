# `meteoraster` example notebooks

Seven short notebooks, one idea each, run against the **real** files already committed in
[../tests/resources](../tests/resources) — converted Tethys forecasts (NOAA GFS, ECMWF
SEAS5, IPMA) and a 26-catchment KML of the Vesdre basin. They are committed **with their
outputs**, so every table and figure below is readable on GitHub without running anything.

| # | Notebook | What it shows |
|---|---|---|
| 01 | [The 5-D data model](01_the_5d_data_model.ipynb) | The `[production, member, leadtime, lat, lon]` cube, 2-D coordinate grids, event time, and the completeness diagnostics |
| 02 | [Loading, joining, saving, xarray](02_load_join_save_xarray.ipynb) | `read_header`, full and partial `load`, `join`, `save` round-trip, `to_xarray`, streaming with `iter_productions` |
| 03 | [Cropping](03_cropping.ipynb) | `get_cropped` in space, in production time and in leadtime |
| 04 | [Plotting](04_plotting.ipynb) | `plot_mean` on a map, `plot_mean_projected`, `plot_coordinates`, `plot_availability`, `plot_seasonal` |
| 05 | [Point time series](05_point_time_series.ipynb) | `get_values_from_latlon`, forecast spaghetti against event time, ensemble spread |
| 06 | [Zonal statistics from KML](06_zonal_statistics_from_kml.ipynb) | Catchment means and area-weighted quantiles, coverage weights, coverage caching |
| 07 | [Resampling](07_resampling.ipynb) | `sum`/`mean`/`linear`/`max` semantics, mass conservation, uneven leadtime spacing, forecast vs time-series models |

Read them in order the first time — 01 sets up the vocabulary (production date, leadtime,
event date) that the rest assume, and 07 uses the catchment aggregate built in 06.

## Running them

```bat
conda env create -f ../environment.yml
conda activate tethys_rasters
pip install -e "..[dev]"
jupyter lab
```

Activate the environment rather than calling its `python.exe` directly.

### What each notebook needs

| Needs | Notebooks |
|---|---|
| Core only — numpy, scipy, pandas, xarray, h5netcdf, h5py | 01, 02 |
| `matplotlib` | 03, 05, 06, 07 |
| `matplotlib` + **`cartopy`** (map projections and the Natural Earth coastlines) | 04 |
| **GDAL / `osgeo`** (conda, not pip) for the KML polygons | 06, and section 5 of 07 |

Each notebook falls back to the repository root on `sys.path` when `meteoraster` is not
installed in the environment, so a plain clone works without `pip install -e ..`.

The data is in `tests/resources`, which is **not** shipped in the sdist — these notebooks
run from a clone of the repository, not from `pip install meteoraster`.

Nothing here writes into the repository: the notebooks that need to write a file do it in a
`tempfile.TemporaryDirectory()` that is discarded at the end of the cell.

## Not covered here: GRIB

`meteoraster.utils` reads ERA5-Land GRIB (`readERA5Land_monthly`, `read_ERA5Land_hourly`,
the latter de-accumulating to per-hour increments). It is shown as a code block in notebook
02 but not executed, because `cfgrib` needs the ecCodes binary on `PATH` — on Windows it
lives in `<env>/Library/bin` and is only found through an *activated* environment, so an
executed cell would fail for many readers. `tests/test_utils_readers.py` covers those
readers against the GRIB fixtures.
