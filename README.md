<p align="center">
  <img src="images/foresight.png" alt="FORESIGHT" width="320">
</p>

# Tethys meteoraster

<p align="center">
  <a href="https://pypi.org/project/meteoraster/"><img src="https://img.shields.io/pypi/v/meteoraster.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/meteoraster/"><img src="https://img.shields.io/pypi/pyversions/meteoraster.svg" alt="Supported Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

**`meteoraster`** is a Python library, developed by [FORESIGHT — Forecasting and
Optimization for Resilient Environmental Systems through Investigation with
Groundbreaking Hydrological Tools](https://foresight.tecnico.ulisboa.pt/), for
**handling distributed ensemble and probabilistic meteorological forecasts**.

Its core data structure, `MeteoRaster`, holds a single **5-D** array indexed by
`[production_datetime, ensemble_member, leadtime, latitude, longitude]`, and
wraps it with everything you need to turn raw reanalysis/forecast files into
report-ready time series, maps and aggregated statistics:

- read ERA5-Land GRIB (monthly means and hourly, with de-accumulation) into a common model,
- crop, join, trim, resample and check the completeness of the data,
- extract point time series or **catchment/zone aggregates from KML polygons**
  (area-weighted, missing-data aware), as means or as exact weighted quantiles,
- persist to compressed NetCDF (`h5netcdf`) and reload losslessly — with a read
  path that survives an install on a different xarray,
- **stream a file too large to hold in memory**, one production at a time,
- and plot means, seasonal cycles, coordinates and availability with `cartopy`.

Only `numpy`, `scipy`, `pandas`, `xarray`, `h5netcdf` and `h5py` are needed to read,
write, subset and aggregate. GDAL, matplotlib, seaborn and cartopy are imported the
moment you use them, so `import meteoraster` works in a slim container that has
none of them.

---

## Features

| Category | What you get |
|---|---|
| **Data model** | `MeteoRaster` — 5-D `[production_datetime, ensemble_member, leadtime, lat, lon]`; 2-D lat/lon grids; automatic Greenwich (0–360 → −180–180) fix and latitude flip |
| **Format readers** | `meteoraster.utils` — `readERA5Land_monthly`, `read_ERA5Land_hourly` (GRIB via `cfgrib`) |
| **Subsetting / merging** | `get_cropped`, `join`, `trim`, `adjust_leadtimes` |
| **Resampling** | `resample_timestep` (pandas rules) and the standalone `Resample` class |
| **Extraction** | `get_values_from_latlon`, `get_values_from_KML`, `get_quantiles_from_KML`, `get_values_from_latlon_by_event` |
| **Spatial aggregation** | `meteoraster.zonal` — sparse coverage by exact fractional overlap, area-weighted means and quantiles, missing-data aware per zone |
| **Quality / diagnostics** | `is_complete`, `get_complete_index`, `get_missing`, `get_completeness` |
| **I/O** | `save` / `load` (compressed NetCDF via `h5netcdf`), `to_xarray`, and `meteoraster.io` for header-only and streaming reads |
| **Visualisation** | `plot_mean`, `plot_mean_projected`, `plot_seasonal`, `plot_coordinates`, `plot_availability`, `add_shapefile`, `add_shape` |

---

## Installation

The **core** — reading, writing, subsetting, streaming and aggregating — needs only
`numpy`, `scipy`, `pandas`, `xarray`, `h5netcdf` and `h5py`, all pip-installable:

```bat
pip install meteoraster
```

The **geospatial and plotting extras** (**GDAL/`osgeo`**, `cartopy`, `cfgrib`) are
most reliably installed with **conda**, and are imported only when used. Zonal
aggregation from KML or PostGIS geometry needs GDAL; the map plots need cartopy.
`import meteoraster` works without either — a missing one raises at the call site
with the command to install it.

Two paths follow: install the **released package from PyPI** on top of a conda
environment (use the package), or install **from source** (develop / run the tests).

### A · Install from PyPI

Create a conda environment that provides the geospatial libraries, then install
[`meteoraster`](https://pypi.org/project/meteoraster/) into it:

```bat
conda env create -f environment.yml
conda activate tethys_rasters
pip install meteoraster
```

Upgrading later needs no URL — `pip install --upgrade meteoraster` always picks up
the newest release.

> The package declares its pure-Python dependencies, but **GDAL is intentionally
> not** a pip dependency (it is a system/conda package). The conda environment
> above provides it. If you do not need zonal aggregation from vector geometry,
> `pip install meteoraster` into any environment is enough — including a slim
> container image, where `import meteoraster`, `save`, `load` and the streaming
> readers all work with no GDAL, cartopy, matplotlib or seaborn present.

Every release is also attached to the
[GitHub releases page](https://github.com/FORESIGHT-ULisboa/Tethys-meteoraster/releases),
which is useful to pin an exact build without going through PyPI:

```bat
pip install https://github.com/FORESIGHT-ULisboa/Tethys-meteoraster/releases/download/v3.0/meteoraster-3.0-py3-none-any.whl
```

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

That adds `pytest`, `pytest-cov`, `build`, `twine` and `jupyterlab` on top of the
runtime dependencies.

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

The ERA5-Land readers live in `meteoraster.utils` and return a ready-to-use
`MeteoRaster`:

```python
from meteoraster.utils import readERA5Land_monthly, read_ERA5Land_hourly

# monthly means: tp comes back as mm/month, t2m as degC
era5 = readERA5Land_monthly("era5land_monthly.grib", "tp")

# hourly: accumulations are de-accumulated to per-hour increments (mm/hr).
# A CDS request omits the last hour of its final day - it is delivered with the
# next period, so pass that file to fill it in.
hourly = read_ERA5Land_hourly("era5land_2025_11.grib", "tp",
                              file_next_year="era5land_2025_12.grib")
```

These need `cfgrib` and the ecCodes binary. On Windows ecCodes lives in
`<env>/Library/bin` and is only found through `PATH`, so **activate** the conda
environment rather than calling its `python.exe` directly — otherwise `import cfgrib`
raises `RuntimeError: Cannot find the ecCodes library`.

Anything already converted by Tethys needs no reader — `MeteoRaster.load` reads it
directly.

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

Spatial **quantiles** over the same zones, area-weighted so a cell half inside a
zone carries half the probability mass:

```python
agg, centroids = mr.get_quantiles_from_KML("zones.kml", nameField="zone_id",
                                           quantiles=[0.1, 0.5, 0.9])

# Pool an hourly forecast into one distribution per day. `pool` names the axis
# explicitly because "time" is the leadtime axis for a forecast and the production
# axis for an observation raster - the event time is production + leadtime.
agg, _ = mr.get_quantiles_from_KML("zones.kml", nameField="zone_id",
                                   pool=("space", "leadtime"), resampling="1D")
```

The coverage is reusable, and the lower-level API needs no `MeteoRaster` at all —
useful for caching it, or for feeding geometry straight out of PostGIS:

```python
from meteoraster import zonal

coverage = zonal.coverage_from_kml("zones.kml", mr.latitudes, mr.longitudes,
                                   name_field="zone_id")
means = zonal.apply_coverage(coverage, mr.data)      # [..., y, x] -> [..., zone]

coverage.save("coverage.npz")                        # pickle-free
coverage = zonal.Coverage.load("coverage.npz")
assert coverage.matches(mr.latitudes, mr.longitudes) # fingerprints the coordinates

# WKB bytes or WKT work too, so PostGIS geometry needs no intermediate file
coverage = zonal.coverage_from_geometries(mr.latitudes, mr.longitudes,
                                          [feature.geom.wkb], ["basin"])
```

### Crop, save and reload

```python
sub = mr.get_cropped(from_lat=32, to_lat=38, from_lon=-2, to_lon=7,
                     from_prod_date=pd.Timestamp("2023-01-02"))

sub.save("forecast.nc")                 # compressed NetCDF (h5netcdf)
reloaded = MeteoRaster.load("forecast.nc")
```

### Read a file that does not fit in memory

`load()` holds the whole cube. For a large archive, read the metadata first and
then stream one production at a time — peak memory is one block whatever the file
size (measured 126 MB against 467 MB for `load()` on a 352 MB cube):

```python
from meteoraster import io as mrio, zonal

header = mrio.read_header("archive.nct")        # coordinates, shape, units; no values
print(header.shape, header.n_leadtimes, header.complete)

coverage = zonal.coverage_from_kml("zones.kml", header.latitudes, header.longitudes,
                                   name_field="zone_id")

for production, block in mrio.iter_productions("archive.nct", reduce_members="mean"):
    series = zonal.apply_coverage(coverage, block)   # [leadtime, zone]
```

`load()` also takes a subset directly, and casting on the way in genuinely halves
peak memory for a float64 file rather than casting after it is already resident:

```python
mr = MeteoRaster.load("archive.nct", leadtimes=slice(0, 6), dtype="float32")
```

Reads try `h5netcdf`, then `netcdf4`, then `scipy`, and pin the decoding so a file
written by another install returns the same dtypes. If HDF5 locking fails on a
network share or a bind mount, set `HDF5_USE_FILE_LOCKING=FALSE` in the
environment; the library will not set it for you.

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
│   ├── io.py                  # portable NetCDF reads: header-only and streaming
│   ├── zonal.py               # sparse coverage, area-weighted means and quantiles
│   ├── resample.py            # Resample (time-step resampling of DataFrames)
│   ├── _lazy.py               # on-demand GDAL / matplotlib / seaborn / cartopy
│   └── utils.py               # ERA5-Land GRIB readers
├── tests/                     # pytest suite + fixtures (conftest.py)
│   └── resources/             # real GRIB/NetCDF inputs (see PROVENANCE.md)
├── tools/make_fixtures.py     # regenerates the subsetted test fixtures
├── images/foresight.png       # README header
├── environment.yml            # conda environment (tethys_rasters)
├── pyproject.toml             # packaging + pytest configuration
├── AGENTS.md                  # canonical guidance for humans & AI agents
├── CLAUDE.md                  # → points to AGENTS.md
├── .github/copilot-instructions.md  # → points to AGENTS.md
├── .github/workflows/release.yml    # build + publish to PyPI on tag push
├── .github/scripts/check_version.py # release guard: version declared consistently
├── LICENSE                    # Apache 2.0
├── NOTICE                     # attribution required by Apache 2.0
└── README.md
```

---

## Running the tests

```bat
conda activate tethys_rasters
pip install -e ".[dev]"
pytest -ra
```

The suite covers construction, cropping, point/KML extraction, NetCDF round-trips,
completeness checks, resampling and the deprecated-alias warnings. Since v3.0 it also
runs against **real inputs** in `tests/resources` — ERA5-Land GRIB for the readers,
converted Tethys forecasts (ECMWF SEAS5, IPMA, NOAA GFS) for the monthly adaptation and
the resampling units contract, and a real 26-catchment KML layer for zonal aggregation.
Those files are subsets; see
[tests/resources/PROVENANCE.md](tests/resources/PROVENANCE.md) and
`tools/make_fixtures.py`.

Tests skip — by name, never error — when a resource is missing or when an optional
dependency is absent, so a partial clone and a slim install both stay green. **Activate**
the conda environment rather than calling its `python.exe` directly, or the GRIB tests
skip on a `PATH` that hides ecCodes.

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
├── meteoraster-3.0-py3-none-any.whl
└── meteoraster-3.0.tar.gz
```

Install the wheel anywhere (provided GDAL is available):

```bat
pip install dist/meteoraster-3.0-py3-none-any.whl
```

Notes:
- The version lives in [pyproject.toml](pyproject.toml) (`project.version`) and is
  mirrored in `MeteoRaster.VERSION` — bump both together.
  [.github/scripts/check_version.py](.github/scripts/check_version.py) enforces
  that at release time.
- Building by hand is only needed to inspect an artifact locally. Actual releases
  are produced by CI — see [Releasing](#releasing) below.

---

## Releasing

Releases are automated by
[.github/workflows/release.yml](.github/workflows/release.yml). To cut version
`X.Y`:

1. Bump the version in **both** [pyproject.toml](pyproject.toml) and
   `MeteoRaster.VERSION`, then commit.
2. Tag and push:

   ```bat
   git tag vX.Y
   git push origin vX.Y
   ```

The workflow then verifies the two version declarations agree with the tag, builds
the wheel and sdist, runs `twine check --strict`, publishes to
[PyPI](https://pypi.org/project/meteoraster/), and attaches both artifacts to the
GitHub release.

To rehearse without touching PyPI, run the workflow manually from the **Actions**
tab and leave the target as `testpypi`; it builds and uploads to
[TestPyPI](https://test.pypi.org/project/meteoraster/) with no tag required.

> **PyPI uploads are immutable** — a version number can never be reused, even
> after deletion. Get the version right before pushing the tag.

### One-time setup (maintainers)

Publishing is authenticated with
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC):
GitHub mints a short-lived credential at publish time that PyPI accepts only for
this repository, this workflow file and this environment. **No API token is
created, stored or pasted anywhere.** Every value below is a public fact about
the repository — none of it is a secret, which is precisely the point.

**On PyPI** — once per index (repeat on [TestPyPI](https://test.pypi.org/) if you
want the rehearsal path):

1. *Your account → Publishing → Add a pending publisher* → GitHub, and fill in:

   | Field | Value |
   |---|---|
   | PyPI project name | `meteoraster` |
   | Owner | `FORESIGHT-ULisboa` |
   | Repository | `Tethys-meteoraster` |
   | Workflow name | `release.yml` |
   | Environment | `pypi` (on TestPyPI: `testpypi`) |

   The environment name must match the `environment:` key of the corresponding
   job in [release.yml](.github/workflows/release.yml), or PyPI rejects the
   exchange. A *pending* publisher is what you register before the project exists
   on an index; it turns into an ordinary one on the first successful upload.

**On GitHub** — once:

2. *Settings → Environments* → create `pypi` (and `testpypi`).
3. Add a **required reviewer** to `pypi`. Every publish then pauses for an
   explicit approval, so a tag push alone cannot ship a release.

Things worth keeping that way:

- **Do not add a PyPI API token to repository secrets.** Trusted Publishing
  exists to remove that long-lived credential; a token in secrets is usable by
  any workflow in the repository and stays valid until someone revokes it.
  Nothing here needs one.
- The publisher trusts the workflow **by filename**, so anyone able to change
  [release.yml](.github/workflows/release.yml) or push a `v*` tag can trigger a
  publish. Protect `main` and restrict who may push tags; the environment
  reviewer from step 3 is the backstop.
- Only the two publish jobs request `id-token: write`; the build job is
  `contents: read` and the workflow default is `permissions: {}`. Keep that split
  when editing it — the job that produces the artifacts never holds the identity
  that can upload them.

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
| `get_quantiles_from_KML(kml, nameField=None, quantiles=…, pool=('space',), resampling=None, …)` | Area-weighted spatial quantiles over KML zones → `(agg, centroids)`. `pool` names the axes folded into the sample; `resampling` is the block size within a pooled time axis. |
| `get_values_from_latlon_by_event(production_date_dataframe)` | *(static)* Reindex a production-date frame by event date. |

Since **v3.0** `get_values_from_KML` runs on `meteoraster.zonal` (below). Its numbers
changed for any zone touching the edge of the grid: the legacy builder gave the
outermost row and column no weight and took its validity threshold from the first
zone. `ensemble_member` labels are now 0-based, matching the other extractors.

**`meteoraster.zonal`** — needs no `MeteoRaster`; takes plain coordinate arrays.

| Member | Description |
|---|---|
| `coverage_from_kml(kml, latitudes, longitudes, name_field=None, …)` | Build a `Coverage` from KML Placemark polygons. Namespace-agnostic, and it does not touch the input file. |
| `coverage_from_geometries(latitudes, longitudes, geometries, names=None, …)` | Same from `ogr.Geometry`, WKB bytes or WKT — so PostGIS geometry feeds straight in. |
| `apply_coverage(coverage, values, min_coverage=0.8)` | Area-weighted mean per zone of any array whose last two axes are (y, x). Renormalises for missing data and applies the threshold per zone and per slice. |
| `quantiles_from_coverage(coverage, values, quantiles, pool_axes=(), …)` | Area-weighted spatial quantiles per zone. |
| `weighted_quantiles(values, weights, quantiles, method='hazen')` | The quantile primitive. Reduces exactly to `np.quantile` for equal weights. |
| `Coverage.save/load`, `Coverage.matches`, `grid_fingerprint` | Persist a coverage to a pickle-free `.npz` and check it against a grid; the fingerprint hashes the coordinates, so a shifted grid of the same shape is a miss. |
| `cell_corners`, `cell_areas`, `cell_bounds` | The grid geometry, extrapolated half a cell at the boundary so no cell is skipped. |

**`meteoraster.io`** — the portable read path.

| Member | Description |
|---|---|
| `read_header(file)` | `Header` with coordinates, shape, dtype, units and the `complete` flag; reads no values. |
| `iter_productions(file, productions=None, leadtimes=None, reduce_members=None, dtype=None)` | Yield `(production_datetime, block)`; peak memory is one block. |
| `read_cube(file, …)` | Fill a whole or partial cube production by production. |
| `open_raster(file, engine=None, **kwargs)` | Context manager over `xr.open_dataset` with engine fallback and pinned decoding. |
| `read_completeness(file)` | The `complete` flag alone; `None` when the file has none. |
| `available_engines()` | Which of `h5netcdf`, `netcdf4`, `scipy` this install can use. |

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
| `save(file, complevel=1, complete=None, tmp_dir=None, mode=0o644, engine=None)` | Write compressed NetCDF via `h5netcdf`. Staged in the destination directory and swapped in with `os.replace`, so nothing goes through the system temp dir and the swap is atomic. |
| `load(file, verbose=None, productions=None, leadtimes=None, dtype=None, engine=None)` | *(classmethod)* Reload a NetCDF written by `save`, optionally only part of it. |
| `read_header(file)` / `iter_productions(file, …)` | *(classmethods)* Metadata-only and streaming reads; see `meteoraster.io`. |

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

Licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).
