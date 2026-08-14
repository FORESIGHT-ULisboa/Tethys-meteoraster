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
| [meteoraster/io.py](meteoraster/io.py) | Portable NetCDF access: `open_raster`, `read_header`, `iter_productions`, `read_cube` |
| [meteoraster/zonal.py](meteoraster/zonal.py) | Zonal aggregation: cell corners, `Coverage`, `apply_coverage`, `weighted_quantiles` |
| [meteoraster/_lazy.py](meteoraster/_lazy.py) | On-demand imports for GDAL/matplotlib/seaborn/cartopy |
| [meteoraster/resample.py](meteoraster/resample.py) | `Resample` — time-step resampling of production×leadtime `DataFrame`s |
| [meteoraster/utils.py](meteoraster/utils.py) | ERA5-Land GRIB readers (monthly means, hourly) |
| [meteoraster/__init__.py](meteoraster/__init__.py) | Exports `MeteoRaster`, `Resample` (note: `utils` is imported as a submodule) |
| [tests/](tests/) | pytest suite + fixtures ([tests/conftest.py](tests/conftest.py)) |
| [notebooks/](notebooks/) | seven worked examples, committed **with outputs**, run against `tests/resources` — see [notebooks/README.md](notebooks/README.md) |

`io.py` and `zonal.py` are deliberately independent of `MeteoRaster` and of each
other: both take plain arrays, so a caller can stream one production block off disk
and aggregate it without ever building a cube.

## Public API (the parts to know)

**`MeteoRaster`** — construct from a 5-D array (or a dict with keys `data`,
`latitudes`, `longitudes`, `production_datetime`, `leadtimes`). Key methods:

- Subset/merge: `get_cropped`, `join`, `trim`, `adjust_leadtimes`, `resample_timestep`
- Extract: `get_values_from_latlon`, `get_values_from_KML`, `get_quantiles_from_KML`,
  `get_values_from_latlon_by_event` (static)
- Quality: `is_complete`, `get_complete_index`, `get_missing`, `get_completeness` (classmethod)
- I/O: `save`, `load` (classmethod), `to_xarray`, `copy`, `read_header` /
  `iter_productions` (classmethods delegating to `meteoraster.io`)
- Plot: `create_plot`, `plot_mean`, `plot_mean_projected`, `plot_seasonal`,
  `plot_coordinates`, `plot_availability`, `add_shapefile`, `add_shape`

**`meteoraster.io`** — every read goes through `open_raster`, which tries the read
engines in turn and pins the decoding choices that drift between xarray releases.
`read_header(file)` returns a `Header` (coordinates, shape, dtype, units,
`complete`) without reading values; `iter_productions(file, …)` yields
`(production_datetime, block)` with peak memory of one block; `read_cube` fills a
whole or partial cube production by production. `Header.latitudes` and every yielded
block are north-first, matching `MeteoRaster`.

**`meteoraster.zonal`** — `coverage_from_geometries` / `coverage_from_kml` build a
sparse `Coverage` (per-zone cell indices and weights) by exact fractional overlap;
`apply_coverage(coverage, values)` takes any array whose last two axes are (y, x)
and returns the area-weighted mean per zone, renormalising for missing data and
testing the coverage threshold **per zone and per slice**. `Coverage.save`/`load`
persist to a pickle-free `.npz`, and `grid_fingerprint` is the cache key — it hashes
the coordinates, not just the shape, so a grid that keeps its shape but shifts its
geolocation is a miss. `weighted_quantiles` / `quantiles_from_coverage` give exact
area-weighted quantiles. Geometries may be `ogr.Geometry`, WKB bytes or WKT, so
PostGIS output feeds straight in.

**`Resample.resample(data, timestep_frequency, resamplingType, …)`** (classmethod)
resamples a `DataFrame` whose columns carry a `leadtime` level. `resamplingType` ∈
{`'sum'`, `'mean'`, `'linear'`, `'max'`}. Two data models share the method:

- **one leadtime column → a time series**, aggregated *along production time*;
- **several leadtime columns → forecasts**, where production times are never mixed
  with each other (only leadtimes are aggregated) and a production stamp between two
  issues reuses the last forecast with its leadtimes shifted.

Leadtimes **need not be evenly spaced** (v3.0): each column spans the distance to the
next one, the last reusing the preceding span, so `1h … 6h, 9h … 18h, 24h … 36h` works
and `sum` still conserves mass where the spacing changes. One production row is enough
for a forecast; only the time-series path needs two. The output covers
`horizon // timestep` whole steps, so a target step that does not divide the horizon
truncates the tail — check mass conservation against the fine-grid result aggregated in
blocks, not against the input total.

Both paths implement the same contract, and picking the wrong `resamplingType` is the
easy way to get silently wrong units:

| type | quantity | up-sample | down-sample | units |
|---|---|---|---|---|
| `sum` | **extensive** — a total per step (mm, J/m²) | splits the total | adds | value rescales with the step: mm stays mm but the accumulation period changes (mm/3h vs mm/day) |
| `mean` | **intensive** — interval average (°C, m/s, W/m², mm/h) | replicates | time-weighted average | preserved |
| `linear` | **intensive** — instantaneous value at the stamp | interpolates | samples at the stamp | preserved |
| `max` | **intensive** — extreme | replicates | maximum | preserved |

So the same physical variable needs `sum` when stored as an accumulation per step and
`mean` when stored as a rate. `sum` conserves mass in both directions. Every value is
taken to describe `[stamp, stamp + step)` — a **forward** convention, so sources that
stamp accumulations at the *end* of their interval (ERA5) must be shifted by the
caller. Any missing input inside an output window yields `NaN`.

**Readers in `meteoraster.utils`** return a ready `MeteoRaster`. Since v3.0 there are
two, and they work: `readERA5Land_monthly` (CDS monthly means; `tp` → `mm/month`,
`t2m` → `C`) and `read_ERA5Land_hourly` (hourly, de-accumulating the cumulative
variables to per-hour increments). `readERA5_monthly`, `read_GFS`, `read_C3S` and
`readCORDEX_monthly` were removed — all six were dead code, and converted Tethys output
needs no reader at all (`MeteoRaster.load`).

Two things the hourly reader encodes, because CDS's layout is not obvious. A request
for one period also returns the *previous* period's last hour, as that date carrying
only its last step, and omits the last hour of its own final day, which is delivered
with the next period — pass it as `file_next_year` to fill in. And the stamping differs
by variable: an accumulation at step *s* describes `[t, t+1h)` and belongs at
`date + (s − s₀)`, while an instantaneous field is valid at cfgrib's `valid_time`,
`date + s`, an hour later.

**GRIB needs the ecCodes binary, and on Windows that means an *activated* environment.**
`findlibs` looks for `{prefix}/lib/libeccodes.dll`, which conda-forge never creates; the
DLL is at `<env>/Library/bin/eccodes.dll` and is only found through `PATH`. Calling
`<env>/python.exe` directly raises `RuntimeError: Cannot find the ecCodes library` even
though the package is installed — use `conda run -n tethys_rasters` or activate. Note
also that this is a `RuntimeError`, not an `ImportError`, so `pytest.importorskip`
cannot guard it; `tests/conftest.py` has a `grib_required` marker that probes it safely.

## Conventions

- **snake_case is the live API.** The camelCase methods (`getCropped`,
  `getValuesFromKML`, `getDataFromLatLon`, `getQuantilesFromKML`,
  `resampleTimeStep`) are thin **deprecated shims** that emit `DeprecationWarning`
  and delegate. Do not add new camelCase methods; prefer/extend snake_case.
- **NumPy is `>=1.26`, unpinned above.** The `<2` cap was lifted in v2.6 and CI
  covers both 1.26 and 2.x. Use `np.nan` (never `np.NaN`) and `np.prod` (never
  `np.product`).
- **`h5netcdf` is the *write* engine** (`MeteoRaster.ENGINE`). Reading goes through
  `io.open_raster`, which falls back `h5netcdf → netcdf4 → scipy`, so a file written
  by another install still opens. Never hard-code an engine on a read path.
  `h5py` is declared explicitly: `h5netcdf >=1.6` has pluggable backends and no
  longer depends on it, so a bare install would otherwise raise at open time.
- **Heavy dependencies are imported on demand**, through
  [meteoraster/_lazy.py](meteoraster/_lazy.py): `osgeo`, `matplotlib`, `seaborn`,
  `cartopy`. `import meteoraster` must keep working without any of them — the
  `minimal` CI job enforces it. Do not add them at module scope, and do not put a
  `cartopy` object in a default argument value (that is evaluated at
  class-definition time); use a `None` sentinel resolved in the body.
- **GDAL / `osgeo` is a conda/system dependency**, deliberately *not* a pip
  dependency. The zonal helpers need it. Install via `environment.yml`.
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

**Notebooks.** [notebooks/](notebooks/) is committed **with its outputs**, so it goes stale
silently if the API changes. Re-execute from the repository root with

```bat
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

which fails on the first cell error and so doubles as a coarse smoke test of the public API
against the real fixtures. They read `tests/resources` through a relative path and put the
repository root on `sys.path` when `meteoraster` is not installed, so a plain clone runs
them. They are not in the sdist or the wheel (`MANIFEST.in` ships only `tests/*.py`, and
`packages.find` only matches `meteoraster*`). Two constraints if you edit them: display
leadtimes through `pd.to_timedelta` — the raw numpy `timedelta64` renders as
`10800000000000 nanoseconds` — and keep absolute paths out of the outputs, which is what the
`warnings.formatwarning` override in each preamble is for.

[.github/workflows/tests.yml](.github/workflows/tests.yml) runs the suite on push
and pull request, in two jobs that catch different things:

- **conda** — the full environment, including GDAL, so the zonal aggregation runs.
- **minimal** — `pip install --no-deps -e .` plus only numpy/scipy/pandas/xarray/
  h5netcdf/h5py, over numpy 1.26 and 2.x. It asserts that `osgeo`, `cartopy`,
  `matplotlib` and `seaborn` are *absent* and that importing, saving, loading and
  streaming still work. This is the slim-container contract; it is what would have
  caught `from osgeo import ogr` at module scope.

Tests needing GDAL are guarded with `importorskip`, so the suite is green (with
skips) on an install without it rather than hard-erroring in `conftest.py`.

**Real-data fixtures.** Since v3.0 the suite reads real files from
[tests/resources](tests/resources) — ERA5-Land GRIB for the readers, converted Tethys
forecasts (ECMWF SEAS5, IPMA, NOAA GFS) for the monthly adaptation and the resampling
units contract, and `catchments.kml` (the Vesdre basin as 26 sub-catchments) for zonal
aggregation. Three rules hold there:

- Everything goes through `conftest.require_resource(name)`, which **skips naming the
  missing file**. A partial clone must skip, never error — and the sdist ships no
  resources at all (`MANIFEST.in` is only `recursive-include tests *.py`), so PyPI is
  unaffected by their size.
- GRIB tests carry `conftest.grib_required`, not `importorskip` — see the ecCodes note
  above for why.
- The committed files are **subsets**; the full downloads were 109 MB and committed git
  blobs are permanent. [tools/make_fixtures.py](tools/make_fixtures.py) rebuilds them
  and [tests/resources/PROVENANCE.md](tests/resources/PROVENANCE.md) records what each
  one is and what the subsetting kept. GRIB subsetting drops whole messages and copies
  the survivors byte for byte, so nothing is re-encoded. If you regenerate a fixture,
  update `PROVENANCE.md` in the same commit.

Two things to know before writing a test against them:

- **Geometry.** `catchments.kml` covers only the fixtures whose grid includes Belgium —
  ERA5-Land (0.1°, 1–5 cells per zone), NOAA GFS (0.25°, 1–4) and ECMWF SEAS5 (1°, 1–2).
  The **ipma** fixtures are over Portugal and do not overlap it. On the coarse grids most
  zones fall *inside* a single cell, so they return identical values — pinned as expected
  behaviour in `test_kml_catchments.py`, and the reason to resample onto a finer grid
  before aggregating. If you widen a `.nct` crop, keep 50.7 N inside it.
- **The c3s values are ~1000× too large for their `mm/month` label** (global member-mean
  median 56 283). It is how the file was produced upstream: `tprate` arrives in
  kg m⁻² s⁻¹ ≡ mm/s and was multiplied by 1000 anyway — the same bug the deleted
  `read_C3S` had. No test asserts anything about that fixture's magnitude, and none
  should; see `PROVENANCE.md`.

`meteoraster` is published on **PyPI** (`pip install meteoraster`). Releases are cut
by [.github/workflows/release.yml](.github/workflows/release.yml) on a `v*` tag push:
it **runs the full test suite** (it calls `tests.yml` as a reusable workflow),
verifies the version declarations agree with the tag, builds, runs
`twine check --strict`, publishes to PyPI, and attaches both artifacts to the GitHub
release. A manual run from the Actions tab targets TestPyPI by default, for rehearsal.

Every publish job is `needs: [test, build]`, so a red suite cannot reach either
index. Keep it that way — before this, `build` only checked the version and the
metadata, and a tag push could ship code whose tests were failing.

Releasing is therefore: bump the version in **both** places, commit, `git tag vX.Y`,
`git push origin vX.Y`. Nothing is uploaded by hand. **PyPI versions are immutable** —
a number can never be reused, even after deletion — so a failing version guard is a
hard stop, not a nuisance.

Build output is **not** committed: `dist/` (local builds) and `dist-release/` (what CI
builds and publishes) are both gitignored. Wheels for v2.0–v2.4 were committed under
`dist/` historically; they remain attached to their GitHub releases, and the artifacts
that only ever lived in `dist/` (the 0.1/0.2 wheels and the pre-2.5 sdists) are still
reachable through the tags, e.g. `raw/v2.4/dist/meteoraster-2.4.tar.gz`.

### One-time PyPI setup (Trusted Publishing)

Authentication is OIDC; **no API tokens exist in the repository or in secrets, and
none should be added** — see [README.md](README.md#one-time-setup-maintainers) for the
registration steps and the reasoning. Two constraints matter when touching the
workflow:

- The environment names (`pypi` / `testpypi`) are part of what PyPI verifies. Renaming
  a job's `environment:` key without re-registering the publisher breaks the OIDC
  exchange.
- `id-token: write` belongs only to the publish jobs. The build job stays
  `contents: read`, so the job that produces artifacts never holds the identity that
  can upload them.

## Known rough edges

- **`get_values_from_KML` was rewired onto `meteoraster.zonal` in v3.0** and is no
  longer a rough edge. What changed, since it changes returned numbers: the legacy
  builder (`__coverageMatrixFromKML`, now deleted) clipped its candidate window to
  `0 < i < n-1`, so **the outermost row and column of the grid never received
  coverage** — and since weights were renormalised by the area actually found, the
  numbers looked plausible with the border silently missing (measured: 1.0 on a 3×3
  grid whose true area-weighted mean is 9.0). Its coverage/validity threshold was also
  taken from **zone 0** and applied to every zone. `ensemble_member` labels were
  1-based there and are now 0-based, matching `get_values_from_latlon` and
  `get_quantiles_from_KML`. `elementwise=` is accepted, ignored and warns.
  Deleting that builder also removed the last scipy `LinearNDInterpolator` from a
  library code path (it ran 8 scalar calls per candidate cell) and the hard-coded
  **2 degree** envelope padding.
- [resample.py](meteoraster/resample.py) was rewritten in **`Resample.VERSION` 2.0**
  (uneven leadtime spans and single-production forecasts followed in **2.1**) and is no
  longer a rough edge: the pandas-2/3 casualties are gone
  (`groupby(axis=1)`, `resample(axis=1)`, `closed=`) and
  [tests/test_resample_semantics.py](tests/test_resample_semantics.py) covers the
  up/down × sum/mean/linear/max matrix, mass conservation, `NaN` propagation, the
  spacing-change cases and the former crash cases. Two consequences to know:
  - **Time-series up-sampling changed numbers.** `sum` used to emit a sliding total
    over the whole production step (a 1D→6H total came out 4× too large) and `mean`
    divided the average by the ratio. Both now match the forecast path. Down-sampling
    results are byte-identical to v1.01.

  The internal grid is the exact `gcd` of the production, leadtime and target steps —
  not `min(..., 1H, ...)` — which is why non-integer ratios are no longer truncated
  and why an all-daily frame no longer inflates to an hourly grid (measured 1.31 s →
  0.9 ms on 365×10). A tz-aware index crossing DST has 24 h and 23 h spacings, so the
  `gcd` still lands on 1 h there by itself; do not reintroduce a hard-coded floor.
  Calendar steps are rejected with a named error rather than parsed: `pd.Timedelta`
  reads `'MS'` as **1 millisecond**, which used to turn a monthly index into a 97 GiB
  allocation. The check is case-sensitive, since `'ms'` legitimately means
  milliseconds.
- **A `timestep_frequency` is a fixed duration, so the output grid is built by
  arithmetic, not `date_range`.** `pd.Timedelta('1D')` resolves to a calendar `<Day>`
  on pandas 2 but to `<24 * Hours>` on pandas 3, which silently moved the output grid
  of a tz-aware index across a DST switch. `Resample._output_index` sidesteps that and
  behaves the same on both. Consequence: for a tz-aware index, stamps stop landing on
  local midnight after a switch (and a 24 h forecast then cannot fill a 24 h window
  offset by an hour, so those rows are `NaN`). Resample in UTC when that matters —
  a naive index and its UTC equivalent give identical results.
- **`adjust_leadtimes` and `resample_timestep` are still marked experimental** and warn
  on every call. They work — v3.0 fixed `adjust_leadtimes`, which measured its horizon
  with `.dt.days` and so raised `AttributeError` whenever the leadtimes were already
  `pd.DateOffset`s, i.e. on exactly the monthly data it exists for — but neither has the
  test coverage `Resample` does. Note they are *different code paths* from `Resample`
  and take pandas **calendar** rules (`'MS'`), which `Resample` rejects by name. Easy to
  conflate.
- A broken BLAS/LAPACK build (a `defaults`/MKL mix is the usual cause) **crashes
  the interpreter** rather than raising, the first time `numpy.linalg` or scipy's
  `LinearNDInterpolator` is evaluated — no traceback, and it takes the whole test
  session with it. The `conda` CI job checks `np.linalg.inv` up front so this fails
  loudly. If the suite dies with `Windows fatal exception: code 0xc06d007f`, that
  is this, not the library.
