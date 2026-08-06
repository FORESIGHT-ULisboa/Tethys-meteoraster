# AGENTS.md

Canonical guidance for humans and AI coding agents working on **Tethys-meteoraster**.
`CLAUDE.md` and `.github/copilot-instructions.md` point here.

## What this project is

`meteoraster` is a Python library from [FORESIGHT / IST](https://foresight.tecnico.ulisboa.pt/)
for handling **distributed ensemble and probabilistic meteorological forecasts**.
Everything revolves around the `MeteoRaster` class, which wraps a single **5-D**
NumPy array and the metadata needed to read, subset, aggregate, save and plot it.

## Data model

`MeteoRaster.data` is a 5-D `numpy.ndarray` with axes, in order:

| Axis | Name | Meaning |
|---|---|---|
| 0 | `production_datetime` | when the forecast was issued (`pd.Timestamp`/`datetime64`) |
| 1 | `ensemble_member` | ensemble member index (`MeteoRaster.ENSEMBLEMEMBERpOSITION = 1`) |
| 2 | `leadtime` | forecast horizon (`pd.Timedelta` or `pd.DateOffset`) |
| 3 | `latitude` (`y`) | spatial |
| 4 | `longitude` (`x`) | spatial |

Coordinate handling, applied automatically in `__init__`:
- `latitudes` / `longitudes` are stored as **2-D meshgrids**; 1-D inputs are meshed.
- Longitudes are wrapped from `[0, 360]` to `[-180, 180]` (`_fixStartAtGreenwich`).
- Latitudes are flipped to descending order if needed (`_flipLatitude`).

Deterministic and single-member datasets just use size-1 ensemble/leadtime axes.

## Package layout

| Path | Contents |
|---|---|
| [meteoraster/meteoraster.py](meteoraster/meteoraster.py) | `MeteoRaster` — the core class (construction, cropping, extraction, KML aggregation, I/O, plotting) |
| [meteoraster/resample.py](meteoraster/resample.py) | `Resample` — time-step resampling of production×leadtime `DataFrame`s |
| [meteoraster/utils.py](meteoraster/utils.py) | Format readers: ERA5, ERA5-Land, GFS, C3S, CORDEX |
| [meteoraster/__init__.py](meteoraster/__init__.py) | Exports `MeteoRaster`, `Resample` (note: `utils` is imported as a submodule) |
| [tests/](tests/) | pytest suite + fixtures ([tests/conftest.py](tests/conftest.py)) |

## Public API (the parts to know)

**`MeteoRaster`** — construct from a 5-D array (or a dict with keys `data`,
`latitudes`, `longitudes`, `production_datetime`, `leadtimes`). Key methods:

- Subset/merge: `get_cropped`, `join`, `trim`, `adjust_leadtimes`, `resample_timestep`
- Extract: `get_values_from_latlon`, `get_values_from_KML`, `get_quantiles_from_KML`,
  `get_values_from_latlon_by_event` (static)
- Quality: `is_complete`, `get_complete_index`, `get_missing`, `get_completeness` (classmethod)
- I/O: `save`, `load` (classmethod), `to_xarray`, `copy`
- Plot: `create_plot`, `plot_mean`, `plot_mean_projected`, `plot_seasonal`,
  `plot_coordinates`, `plot_availability`, `add_shapefile`, `add_shape`

**`Resample.resample(data, timestep_frequency, resamplingType, …)`** (classmethod)
resamples a `DataFrame` whose columns carry a `leadtime` level. `resamplingType` ∈
{`'sum'`, `'mean'`, `'linear'`, `'max'`}.

**Readers in `meteoraster.utils`** return a ready `MeteoRaster`:
`readERA5_monthly`, `readERA5Land_monthly`, `read_ERA5Land_hourly`, `read_GFS`,
`read_C3S`, `readCORDEX_monthly`.

## Conventions

- **snake_case is the live API.** The camelCase methods (`getCropped`,
  `getValuesFromKML`, `getDataFromLatLon`, `getQuantilesFromKML`,
  `resampleTimeStep`) are thin **deprecated shims** that emit `DeprecationWarning`
  and delegate. Do not add new camelCase methods; prefer/extend snake_case.
- **NumPy is pinned `<2`.** Use `np.nan` (never `np.NaN`) and `np.prod` (never
  `np.product`) so the code stays forward-compatible.
- **NetCDF engine is `h5netcdf`** (`MeteoRaster.ENGINE`). `save`/`load` require it;
  it is declared in both `environment.yml` and `pyproject.toml`.
- **GDAL / `osgeo` is a conda/system dependency**, deliberately *not* a pip
  dependency. The KML helpers (`get_values_from_KML`, `__coverageMatrixFromKML`)
  need it. Install via `environment.yml`.
- Version lives in **two** places that must be bumped together:
  `pyproject.toml` (`project.version`) and `MeteoRaster.VERSION`
  ([meteoraster.py](meteoraster/meteoraster.py)).
  [.github/scripts/check_version.py](.github/scripts/check_version.py) enforces this
  (plus tag agreement) before anything is published.
- **Licensed Apache-2.0.** The license is declared as a PEP 639 SPDX expression
  (`license = "Apache-2.0"` plus `license-files`), which is why `build-system.requires`
  pins `setuptools>=77` and why there is **no** `License :: OSI Approved :: ...`
  classifier — PyPI rejects metadata carrying both.

## Build / test / release

```bat
conda env create -f environment.yml      # or: conda env update -f environment.yml --prune
conda activate tethys_rasters
pip install -e ".[dev]"
pytest -ra                                # run the suite
python -m build                           # wheel + sdist into dist/
```

`meteoraster` is published on **PyPI** (`pip install meteoraster`). Releases are cut
by [.github/workflows/release.yml](.github/workflows/release.yml) on a `v*` tag push:
it verifies the version declarations agree with the tag, builds, runs
`twine check --strict`, publishes to PyPI, and attaches both artifacts to the GitHub
release. A manual run from the Actions tab targets TestPyPI by default, for rehearsal.

Releasing is therefore: bump the version in **both** places, commit, `git tag vX.Y`,
`git push origin vX.Y`. Nothing is uploaded by hand. **PyPI versions are immutable** —
a number can never be reused, even after deletion — so a failing version guard is a
hard stop, not a nuisance.

Historical wheels/sdists stay **committed** under [dist/](dist/) so previously
documented URLs keep resolving. New releases no longer depend on that, and committing
build output is optional going forward.

### One-time PyPI setup (Trusted Publishing)

Authentication is OIDC; no API tokens exist in the repository or in secrets. The
publisher is registered once per index. Since the project is not on the index yet,
it is registered as a **pending publisher**:

- PyPI → *Publishing* → *Add a pending publisher* → GitHub
- Owner `FORESIGHT-ULisboa`, repository `Tethys-meteoraster`
- Workflow filename `release.yml`
- Environment `pypi` (on TestPyPI: `testpypi`) — must match the `environment:` key in
  the workflow job, or the OIDC exchange is rejected

Create GitHub environments with those names under *Settings → Environments*. Adding a
required reviewer there makes every publish an explicit approval gate.

## Known rough edges

- `__groupByQuantile` (reached via `get_quantiles_from_KML`) contains a hard
  `raise Exception('This must be reviewed.')` — that path is **non-functional**
  and should be treated as unfinished, not a regression.
- `resample_timestep` raises `TypeError` at its final diagnostic line: it calls
  `Timestamp.strftime('%Y-%m-%d %H:%M:%S', self.verbose)` (an extra positional
  arg). Dropping the stray `self.verbose` would fix it — left untouched here as
  it is outside the deprecation-cleanup scope and untested.
- [resample.py](meteoraster/resample.py) uses pandas idioms that are deprecated
  or removed on pandas ≥2 — `DataFrame.groupby(..., axis=1)`,
  `resample(..., axis=1)`, the old `stack()` behaviour, and `closed=` on
  `timedelta_range`/`date_range` (the last actually raises on pandas ≥2). The
  module is largely untested; touch it only with new tests and against the pinned
  pandas version.
- The KML reader removes the XML namespace and repairs a non-bare `<kml …>` header
  (`__correctKMLHeader`); test KML fixtures use a bare `<kml>` root.
- Format readers in `utils.py` need real GRIB/NetCDF inputs, so they are not
  covered by the automated tests.
