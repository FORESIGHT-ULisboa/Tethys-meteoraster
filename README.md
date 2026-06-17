<p align="center">
  <img src="images/foresight.png" alt="FORESIGHT" width="320">
</p>

# Tethys meteoraster

**`meteoraster`** is a Python library, developed by [FORESIGHT — Forecasting and
Optimization for Resilient Environmental Systems through Investigation with
Groundbreaking Hydrological Tools](https://foresight.tecnico.ulisboa.pt/), for
**handling distributed ensemble and probabilistic meteorological forecasts**.

Its core data structure, `MeteoRaster`, holds a single **5-D** array indexed by
`[production_datetime, ensemble_member, leadtime, latitude, longitude]`, and
wraps it with everything you need to turn raw reanalysis/forecast files into
report-ready time series, maps and aggregated statistics:

- read ERA5, ERA5-Land, GFS, C3S (seasonal) and CORDEX files into a common model,
- crop, join, trim, resample and check the completeness of the data,
- extract point time series or **catchment/zone aggregates from KML polygons**
  (area-weighted, missing-data aware),
- persist to compressed NetCDF (`h5netcdf`) and reload losslessly,
- and plot means, seasonal cycles, coordinates and availability with `cartopy`.

---

## Features

| Category | What you get |
|---|---|
| **Data model** | `MeteoRaster` — 5-D `[production_datetime, ensemble_member, leadtime, lat, lon]`; 2-D lat/lon grids; automatic Greenwich (0–360 → −180–180) fix and latitude flip |
<!-- | **Format readers** | `readERA5_monthly`, `readERA5Land_monthly`, `read_ERA5Land_hourly`, `read_GFS`, `read_C3S`, `readCORDEX_monthly` | -->
| **Subsetting / merging** | `get_cropped`, `join`, `trim`, `adjust_leadtimes` |
| **Resampling** | `resample_timestep` (pandas rules) and the standalone `Resample` class |
| **Extraction** | `get_values_from_latlon`, `get_values_from_KML`, `get_quantiles_from_KML`*, `get_values_from_latlon_by_event` |
| **Spatial aggregation** | area-weighted coverage matrices from KML polygons, missing-data aware |
| **Quality / diagnostics** | `is_complete`, `get_complete_index`, `get_missing`, `get_completeness` |
| **I/O** | `save` / `load` (compressed NetCDF via `h5netcdf`), `to_xarray` |
| **Visualisation** | `plot_mean`, `plot_mean_projected`, `plot_seasonal`, `plot_coordinates`, `plot_availability`, `add_shapefile`, `add_shape` |

\* `get_quantiles_from_KML` is currently under review and not functional — see
[AGENTS.md](AGENTS.md).

---

## Installation

`meteoraster` relies on a geospatial stack (**GDAL/`osgeo`**, `cartopy`, `cfgrib`)
that is most reliably installed with **conda**. Two paths follow: install a
**released wheel** on top of a conda environment (use the package), or install
**from source** (develop / run the tests).

### A · Install a released wheel from GitHub

The wheels are committed to the repository under [`dist/`](dist/). First create a
conda environment that provides the geospatial libraries, then install the wheel:

```bat
conda env create -f environment.yml
conda activate tethys_rasters
pip install https://github.com/FORESIGHT-ULisboa/Tethys-meteoraster/releases/download/v2.4/meteoraster-2.4-py3-none-any.whl
```

> The wheel declares its pure-Python dependencies, but **GDAL is intentionally
> not** a pip dependency (it is a system/conda package). The conda environment
> above provides it. If you only need the point/plotting features and already
> have GDAL, you can `pip install` the wheel into any environment.

### B · Install from source (development)

#### 1 · Create the conda environment

```bat
conda env create -f environment.yml
conda activate tethys_rasters
```

#### 2 · Install the package (editable) with the dev extras

From the repository root:

```bat
pip install -e ".[dev]"
```

That adds `pytest`, `pytest-cov`, `build` and `jupyterlab` on top of the runtime
dependencies.

---

## Quick start

```python
import numpy as np
import pandas as pd
from meteoraster import MeteoRaster

# data axes: [production_datetime, ensemble_member, leadtime, lat, lon]
data = np.random.rand(2, 3, 2, 4, 5)

mr = MeteoRaster(
    data=data,
    latitudes=np.linspace(40, 30, 4),       # 1-D is auto-meshed to 2-D
    longitudes=np.linspace(-10, 10, 5),
    production_datetime=pd.to_datetime(["2023-01-01", "2023-01-02"]).values,
    leadtimes=np.array([pd.Timedelta(days=1), pd.Timedelta(days=2)]),
    units="mm",
    variable="precip",
)
```

### Read a real dataset

The readers live in `meteoraster.utils` and return a ready-to-use `MeteoRaster`:

```python
from meteoraster.utils import readERA5Land_monthly, read_C3S

era5 = readERA5Land_monthly("era5land_tp.grib", "tp")   # total precipitation
c3s  = read_C3S("seasonal_t2m.grib", "t2m")             # seasonal 2 m temperature
```

### Extract a point time series

```python
# DataFrame: index = production_datetime, columns = MultiIndex(leadtime, ensemble_member)
ts = mr.get_values_from_latlon(lat=35.0, lon=0.0)
```

### Aggregate over KML zones (e.g. catchments)

```python
# agg: DataFrame indexed by production date, columns MultiIndex(zone, leadtime, ensemble_member)
# centroids: DataFrame of zone centroids (x, y)
agg, centroids = mr.get_values_from_KML("zones.kml", nameField="zone_id")
```

### Crop, save and reload

```python
sub = mr.get_cropped(from_lat=32, to_lat=38, from_lon=-2, to_lon=7,
                     from_prod_date=pd.Timestamp("2023-01-02"))

sub.save("forecast.nc")                 # compressed NetCDF (h5netcdf)
reloaded = MeteoRaster.load("forecast.nc")
```

### Resample in time

```python
import numpy as np

mr.resample_timestep("MS", fun=np.nanmean)   # to month start, ensemble-wise

# Or resample a production×leadtime DataFrame with the standalone helper:
from meteoraster import Resample
resampled = Resample.resample(df, timestep_frequency="1D", resamplingType="sum")
```

### Plot

```python
ax, cbar = mr.plot_mean(central_longitude=0, coastline=True, borders=True,
                        colorbar=True, cmap="viridis")
mr.add_shapefile(ax, "basin.shp")
mr.plot_seasonal(lat=35.0, lon=0.0)
mr.plot_availability()
```

---

## Project structure

```
Tethys-meteoraster/
├── meteoraster/
│   ├── __init__.py            # exports MeteoRaster, Resample
│   ├── meteoraster.py         # MeteoRaster core class
│   ├── resample.py            # Resample (time-step resampling of DataFrames)
│   └── utils.py               # ERA5 / ERA5-Land / GFS / C3S / CORDEX readers
├── tests/                     # pytest suite + fixtures (conftest.py)
├── images/foresight.png       # README header
├── dist/                      # built wheels + sdists (committed)
├── environment.yml            # conda environment (tethys_rasters)
├── pyproject.toml             # packaging + pytest configuration
├── AGENTS.md                  # canonical guidance for humans & AI agents
├── CLAUDE.md                  # → points to AGENTS.md
├── .github/copilot-instructions.md  # → points to AGENTS.md
├── LICENSE                    # Apache 2.0
└── README.md
```

---

## Running the tests

```bat
conda activate tethys_rasters
pip install -e ".[dev]"
pytest -ra
```

The suite covers construction, cropping, point/KML extraction, NetCDF
round-trips, completeness checks, resampling and the deprecated-alias warnings.
Tests that need real GRIB/NetCDF inputs (the format readers) are out of scope and
not included.

---

## Building a distribution (wheel + sdist)

The package builds with the standard [PEP 517](https://peps.python.org/pep-0517/)
toolchain:

```bat
conda activate tethys_rasters
pip install -e ".[build]"
python -m build
```

This produces both artifacts in `dist/`:

```
dist/
├── meteoraster-2.4-py3-none-any.whl
└── meteoraster-2.4.tar.gz
```

Install the wheel anywhere (provided GDAL is available):

```bat
pip install dist/meteoraster-2.4-py3-none-any.whl
```

Notes:
- The version lives in [pyproject.toml](pyproject.toml) (`project.version`) and is
  mirrored in `MeteoRaster.VERSION` — bump both together.
- The GitHub wheel URL in [Installation](#a--install-a-released-wheel-from-github)
  resolves once the new `dist/` artifacts are **committed and pushed** (and the
  `v2.4` tag created).

---

## API reference

### `MeteoRaster`

**Construction & conversion**

| Member | Description |
|---|---|
| `MeteoRaster(data, latitudes, longitudes, production_datetime, leadtimes, units='unknown', variable='unknown', …)` | Build from a 5-D array (or a dict with the same keys). 1-D lat/lon are auto-meshed to 2-D; longitudes are wrapped to [−180, 180] and latitudes flipped if needed. |
| `copy()` | Deep copy. |
| `to_xarray()` | Convert to a labelled `xarray.DataArray`. |

**Subsetting, merging & resampling**

| Member | Description |
|---|---|
| `get_cropped(from_prod_date, to_prod_date, from_lat, to_lat, from_lon, to_lon, from_leadtime, to_leadtime)` | Crop in time and space; returns a new `MeteoRaster`. |
| `join(meteoRaster, strickt=False, trim=False)` | Concatenate another raster along production dates (aligns ensembles/leadtimes). |
| `trim()` | Drop leading/trailing all-NaN production dates. |
| `adjust_leadtimes(period='months')` | Align leadtimes to relative periods (experimental). |
| `resample_timestep(rule, fun=np.mean)` | Resample production dates with pandas rules (experimental). |

**Extraction & aggregation**

| Member | Description |
|---|---|
| `get_values_from_latlon(lat, lon)` | Time series from the nearest pixel as a `DataFrame`. |
| `get_values_from_KML(kml, nameField=None, …)` | Area-weighted aggregate over KML polygons → `(agg, centroids)`. |
| `get_quantiles_from_KML(...)` | Spatial quantiles over KML zones — **under review, not functional**. |
| `get_values_from_latlon_by_event(production_date_dataframe)` | *(static)* Reindex a production-date frame by event date. |

**Quality & diagnostics**

| Member | Description |
|---|---|
| `is_complete(full_ensemble=True, space_completeness=False)` | Whether every production×leadtime slice has finite data. |
| `get_complete_index(full_ensemble=True, space_completeness=False)` | Boolean `DataFrame` (production × leadtime) of completeness. |
| `get_missing()` | Fraction of missing pixels per leadtime/ensemble member. |
| `get_completeness(file)` | *(classmethod)* Read the `complete` flag from a saved file. |

**I/O**

| Member | Description |
|---|---|
| `save(file, complevel=1, complete=None)` | Write compressed NetCDF via `h5netcdf`. |
| `load(file, verbose=None)` | *(classmethod)* Reload a NetCDF written by `save`. |

**Visualisation**

| Member | Description |
|---|---|
| `create_plot(central_longitude, …)` | Create a `cartopy` axes. |
| `plot_mean(ax=None, coastline=False, borders=False, colorbar=True, cmap='viridis', …)` | Map of the ensemble/temporal mean → `(ax, cbar)`. |
| `plot_mean_projected`, `plot_coordinates`, `plot_seasonal`, `plot_availability` | Projected mean, grid coordinates, seasonal cycle, data availability. |
| `add_shapefile(ax, shapefile_path, …)` / `add_shape(ax, path, …)` | Overlay shapefile geometries/boundaries. |

> Deprecated camelCase aliases (`getCropped`, `getValuesFromKML`,
> `getDataFromLatLon`, `getQuantilesFromKML`, `resampleTimeStep`) still work but
> emit a `DeprecationWarning`; use the snake_case names above.

### `Resample`

| Member | Description |
|---|---|
| `Resample.resample(data, timestep_frequency, resamplingType, date_from=None, date_to=None, print_func=None)` | *(classmethod)* Resample a production×leadtime `DataFrame`. `resamplingType` ∈ {`'sum'`, `'mean'`, `'linear'`, `'max'`}. |

---

## License
See [LICENSE](LICENSE).
