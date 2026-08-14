# Test fixtures

Real inputs for the tests that cannot be written synthetically: the ERA5-Land readers
in `meteoraster.utils`, the monthly `pd.DateOffset` adaptation, time-step resampling of
forecasts whose leadtime spacing changes, and zonal aggregation over a real catchment
layer.

None of this ships to PyPI — `MANIFEST.in` is only `recursive-include tests *.py`, so
the sdist carries the suite but no resources. Tests skip by name when a file is absent
(`conftest.require_resource`), so a clone without them still passes.

**These are subsets, not the originals.** The full downloads were 109 MB; committed git
blobs are permanent, so they were cut to ~18 MB while keeping every property a test
depends on. Rebuild them with:

```bat
python tools/make_fixtures.py --source <dir with the full downloads>
```

GRIB subsetting drops whole messages and copies the survivors byte for byte, so values,
grid and packing are exactly as downloaded. The `.nct` files were cropped through
`MeteoRaster.get_cropped` and re-saved.

The originals are not kept in this repository. Details below are as measured from the
files; the CDS request dictionaries were not recorded at download time.

---

## `era5land_monthly_tp_and_t2m.zip` — 9.5 MB, unchanged

| | |
|---|---|
| Source | Copernicus CDS, [`reanalysis-era5-land-monthly-means`](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means) |
| Retrieved | 2026-08-14 |
| Contents | 2 GRIB messages: `tp` (`stepType` **`avgas`**) and `2t` (`stepType` `avgid`) |
| Period | 2026-03-01, one month, `step` 24 h |
| Grid | global, 3600 × 1801 at 0.1° (90 → −90 N, −180 → 179.9 E) |

Used by `readERA5Land_monthly`. Two variables with *different* `stepType`s in one file
is why the reader filters on `shortName` — filtering on `stepType` is what broke, since
the value hard-coded before v3.0 was `avgad`. A single month also collapses cfgrib's
`time` dimension to a scalar, which the reader has to expand back.

Kept whole: 2 messages of a global grid leave nothing to drop, and re-encoding a
sub-area is not worth the risk at this size.

## `era5land_tp_belgium.zip` — 1.1 MB, 72 of 720 messages

| | |
|---|---|
| Source | Copernicus CDS, [`reanalysis-era5-land`](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) hourly |
| Retrieved | 2026-08-14 |
| Variable | `tp`, total precipitation, `stepType` `accum` (accumulated since 00 UTC) |
| Area | 54 / −2 / 47 / 12 (N/W/S/E), 141 × 71 at 0.1° |
| Original period | 2025-10-31 to 2025-11-30 |

Subset to a contiguous four-day window that keeps **both** month-boundary patterns the
reader handles:

| dataDate | steps kept | what it is |
|---|---|---|
| 2025-10-31 | 24 | last hour of the *previous* month, as CDS delivers it |
| 2025-11-01 | 1–24 | full day |
| 2025-11-02 | 1–24 | full day |
| 2025-11-03 | 1–**23** | last hour withheld — see below |

## `era5land_tp_belgium_next.zip` — 0.2 MB, 25 messages

Same download, the *following* period. Holds 2025-11-03 `step 24` (the hour withheld
above) plus 2025-11-04 `steps 1–24`.

The split is deliberate: it makes `read_ERA5Land_hourly(..., file_next_year=...)`
testable. A CDS request omits the last hour of its own last day, which arrives with the
next period whose first date *is* this period's last day, carrying `step 24` only.

## `era5land_t2m_belgium.zip` — 1.0 MB, 72 of 720 messages

Same download and area as `era5land_tp_belgium.zip`, variable `2t` with `stepType`
`instant`, and the same four-day subset. Instantaneous rather than accumulated, so it
pins the other half of the stamping contract: cfgrib's `valid_time` is
`date + step`, an hour later than an accumulation's `date + (step − step[0])`.

---

## `tethys_c3s_ecmwf51_tprate_2026.07.nct` — 0.2 MB, cropped

| | |
|---|---|
| Source | Tethys, from Copernicus CDS `seasonal-monthly-single-levels`, ECMWF SEAS5 (system 51) |
| Production | 2026-07-01 |
| Original | `(1, 51, 6, 180, 360)`, global 1°, `mm/month`, `complete=True` |
| Subset | `(1, 8, 6, 33, 45)` — 8 members, lat 20–53 N, lon 20 W–25 E |

The only fixture with `pd.DateOffset` leadtimes (`months=1…6`) and a real ensemble
axis. Drives `adjust_leadtimes(period='months')`, the `DateOffset` save/load round trip,
`io.iter_productions(reduce_members='mean')`, and the assertion that `Resample` refuses
calendar leadtimes.

The northern edge is at 53 N so the grid covers `catchments.kml`; on a 1° grid the whole
Vesdre basin needs only the row of cells centred at 50.5 N.

> ⚠ **The values are ~1000× too large for the `mm/month` they are labelled.** Measured:
> global member-mean median 56 283, max 1 467 041 — i.e. 56 m of rain in a month. Divided
> by 1000 both look right (median 56 mm/month, max 1467 mm/month for the wettest tropical
> cell). This is how the file was produced upstream — `tprate` is delivered in
> kg m⁻² s⁻¹ ≡ mm/s, and the now-deleted `read_C3S` in `meteoraster.utils` had the same
> `*= 1000` in it. **No test asserts anything about the magnitude of this fixture**; the
> assertions are structural and relational (leadtime labels, member reduction, zonal means
> inside the range of the cells they average). Fix it upstream before using this file for
> anything quantitative.

## `tethys_ipma_t2m_20260806.nct` — 0.5 MB, cropped
## `tethys_ipma_tp_20260814.nct` — 0.2 MB, cropped

| | |
|---|---|
| Source | Tethys, from IPMA forecasts |
| Production | 2026-08-06 (`t2m`, `C`, 2 members, `complete=False`) and 2026-08-14 (`tp`, `mm`, 1 member) |
| Original grid | 188 × 371, lat 28–46.7 N, lon 37 W–0 |
| Subset | 41 × 36, lat 37–41 N, lon 9.5 W–6 W (mainland Portugal) |
| Leadtimes | 120, from 1 h to 240 h, spacing changing **1 h → 3 h → 6 h** |

The variable-step case. Spacing that changes *within* one forecast is what `Resample`
gained in v3.0; before that it rejected these outright, and a single production row
was rejected too. Cropping is spatial only, so the leadtime spacing is untouched —
`test_real_rasters.test_ipma_leadtimes_really_do_change_spacing` guards that.

## `catchments.kml` — 0.3 MB, unchanged

| | |
|---|---|
| Source | QGIS export, layer `catchments__vesdre_26_zones` |
| Contents | **26** nested sub-catchments of the **Vesdre** basin (Belgium), named by an `id` field |
| Structure | namespaced KML 2.2, one `<MultiGeometry>`-wrapped `<Polygon>` per Placemark, no inner rings, 7940 vertices |
| Extent | lon 5.6361–6.2557 E, lat 50.4322–50.6772 N |

The only *real* geometry fixture — the KMLs in `conftest.py` are generated on the fly onto
a 3×3 grid whose area weights are computable on paper, which is what makes their
assertions exact. This one covers what they cannot: a real export's structure, 26 zones
at once, and catchments far smaller than a grid cell.

Which fixtures it overlaps, and how many cells each zone lands on — the point being that
the same layer behaves very differently by resolution:

| Grid | Resolution | Cells per zone | Zones inside one cell |
|---|---|---|---|
| ERA5-Land (both hourly, and the global monthly) | 0.1° | 1–5 | 4 of 26 |
| `tethys_NOAA_GFS_*` | 0.25° | 1–4 | 12 of 26 |
| `tethys_c3s_*` | 1° | 1–2 | 20 of 26 |

The two **ipma** fixtures are over Portugal and do not overlap it at all.

## `tethys_NOAA_GFS_0.25_2026.02.02.nct` — 1.8 MB, unchanged
## `tethys_NOAA_GFS_0.25_2026.05.18.nct` — 3.2 MB, unchanged

| | |
|---|---|
| Source | Tethys, from NOAA GFS 0.25° |
| Productions | 7 daily issues from 2026-02-02 and from 2026-05-18 |
| Shape | `(7, 1, 129, 29, 57)`, lat 47–54 N, lon 2 W–12 E |
| Leadtimes | 129, uniform 3 h, 0 to 16 days |
| Units | `mm/3h` (February, `complete=False`) and `C` (May) |

The only multi-production fixture, so it drives the forecast path: production times are
never mixed, and a stamp between two issues reuses the last forecast with its leadtimes
shifted. Kept whole — 29 × 57 is already small, and the 7 productions are the point.
The `C` one also backs the real-data plotting test that replaced the GLEAM fixture.
