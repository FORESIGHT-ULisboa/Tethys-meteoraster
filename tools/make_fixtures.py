'''
Regenerates the subsetted test fixtures in tests/resources from full downloads.

The originals are 109 MB; committed git blobs are permanent, so the fixtures that
ship are cut down to ~20 MB while keeping every property a test depends on. Run this
against the full files to rebuild them:

    python tools/make_fixtures.py --source <dir with the full downloads> [--out tests/resources]

Needs the conda environment ACTIVATED (not just its python.exe called): on Windows the
ecCodes DLL sits in <env>/Library/bin and findlibs only finds it through PATH.

GRIB subsetting drops whole messages and copies the survivors byte for byte - nothing
is re-encoded, so values, grid and packing are identical to the download. What is kept
is chosen to preserve both month-boundary patterns that read_ERA5Land_hourly handles:

    era5land_tp_belgium        Oct 31 step 24 | Nov 1-2 steps 1-24 | Nov 3 steps 1-23
    era5land_tp_belgium_next   Nov 3 step 24  | Nov 4 steps 1-24

Withholding Nov 3 step 24 from the main file and putting it in the "next" file is what
makes the file_next_year borrow testable: the next period's first date is this period's
last day, carrying step 24 only.

era5land_monthly_tp_and_t2m is copied unchanged - it is 2 messages of a global 0.1 deg
grid, so dropping messages cannot help and a geographic re-encode is not worth the risk.
catchments.kml is copied too: it is not derived from anything, so there is nothing to
subset, but copying it means a run reproduces the whole of tests/resources.
'''

from __future__ import annotations

import argparse
import shutil
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# GRIB
# --------------------------------------------------------------------------- #

def iter_messages(buffer):
    '''
    Yields each GRIB message in `buffer` as bytes.

    Message boundaries come from section 0 (total length at bytes 8-16 for edition 2,
    4-7 for edition 1), so no decoder is needed to split the stream.
    '''

    position = 0
    size = len(buffer)
    while True:
        position = buffer.find(b'GRIB', position)
        if position < 0:
            return

        edition = buffer[position + 7]
        if edition == 2:
            length = struct.unpack('>Q', buffer[position + 8:position + 16])[0]
        elif edition == 1:
            length = int.from_bytes(buffer[position + 4:position + 7], 'big')
        else:
            position += 4
            continue

        if length <= 0 or position + length > size:
            position += 4
            continue

        yield buffer[position:position + length]
        position += length


def message_keys(message, keys=('dataDate', 'step', 'shortName')):
    import eccodes as ec

    handle = ec.codes_new_from_message(message)
    try:
        return {key: ec.codes_get(handle, key) for key in keys}
    finally:
        ec.codes_release(handle)


def filter_grib_zip(source, target, keep, member='data.grib'):
    '''
    Writes `target` as a zip holding the messages of `source` for which
    keep(dataDate, step) is true. Survivors are copied verbatim.
    '''

    with zipfile.ZipFile(source) as archive:
        buffer = archive.read(archive.infolist()[0].filename)

    selected = []
    for message in iter_messages(buffer):
        info = message_keys(message)
        if keep(info['dataDate'], info['step']):
            selected.append(message)

    if not selected:
        raise SystemExit('%s: the filter kept no messages' % source.name)

    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, b''.join(selected))

    print('  %-42s %4u messages  %6.1f MB' % (target.name, len(selected),
                                              target.stat().st_size / 1e6))


#: (dataDate, step) pairs kept in each hourly fixture. See the module docstring.
HOURLY_MAIN = {(20251031, 24)}
HOURLY_MAIN |= {(20251101, s) for s in range(1, 25)}
HOURLY_MAIN |= {(20251102, s) for s in range(1, 25)}
HOURLY_MAIN |= {(20251103, s) for s in range(1, 24)}      # last hour withheld

HOURLY_NEXT = {(20251103, 24)}                            # the withheld hour
HOURLY_NEXT |= {(20251104, s) for s in range(1, 25)}


# --------------------------------------------------------------------------- #
# .nct
# --------------------------------------------------------------------------- #

#: name -> (members to keep or None, crop kwargs for MeteoRaster.get_cropped)
NCT_SUBSETS = {
    # 51 members on a global grid is far more than the DateOffset/ensemble coverage
    # needs; 8 members over western Europe and North Africa keeps both. The northern
    # edge has to clear 50.7 N: catchments.kml is the Vesdre basin (Belgium), and on
    # this 1 deg grid it needs the row of cells centred at 50.5.
    'tethys_c3s_ecmwf51_tprate_2026.07.nct': (
        8, dict(from_lat=20, to_lat=53, from_lon=-20, to_lon=25)),
    # The point is the mixed 1h/3h/6h leadtime spacing, which a spatial crop keeps.
    'tethys_ipma_t2m_20260806.nct': (
        None, dict(from_lat=37, to_lat=41, from_lon=-9.5, to_lon=-6)),
    'tethys_ipma_tp_20260814.nct': (
        None, dict(from_lat=37, to_lat=41, from_lon=-9.5, to_lon=-6)),
}

#: Already small (29x57), and the multi-production forecast path needs them whole.
NCT_VERBATIM = ('tethys_NOAA_GFS_0.25_2026.02.02.nct',
                'tethys_NOAA_GFS_0.25_2026.05.18.nct')

#: Not derived from anything, so there is nothing to subset - copied so that a run
#: against a source directory reproduces the whole of tests/resources.
VERBATIM = ('era5land_monthly_tp_and_t2m.zip', 'catchments.kml')


def subset_nct(source, target, members, crop):
    from meteoraster import MeteoRaster

    raster = MeteoRaster.load(source, verbose=False)
    if members is not None:
        raster.data = np.ascontiguousarray(raster.data[:, :members])
    cropped = raster.get_cropped(**crop)
    cropped.save(target, complete=raster.is_complete())

    print('  %-42s %-22s %6.1f MB' % (target.name, cropped.data.shape,
                                      target.stat().st_size / 1e6))


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='directory holding the full-size downloads')
    parser.add_argument('--out', type=Path, default=ROOT / 'tests' / 'resources')
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)

    print('GRIB (messages dropped, survivors copied verbatim):')
    tp_source = arguments.source / 'era5land_tp_belgium.zip'
    filter_grib_zip(tp_source, arguments.out / 'era5land_tp_belgium.zip',
                    lambda date, step: (date, step) in HOURLY_MAIN)
    filter_grib_zip(tp_source, arguments.out / 'era5land_tp_belgium_next.zip',
                    lambda date, step: (date, step) in HOURLY_NEXT)
    filter_grib_zip(arguments.source / 'era5land_t2m_belgium.zip',
                    arguments.out / 'era5land_t2m_belgium.zip',
                    lambda date, step: (date, step) in HOURLY_MAIN)

    print('copied unchanged:')
    for name in VERBATIM:
        target = arguments.out / name
        if (arguments.source / name).resolve() != target.resolve():
            shutil.copyfile(arguments.source / name, target)
        print('  %-42s %6.1f MB' % (name, target.stat().st_size / 1e6))

    print('.nct (cropped):')
    for name, (members, crop) in NCT_SUBSETS.items():
        subset_nct(arguments.source / name, arguments.out / name, members, crop)

    print('.nct (copied unchanged):')
    for name in NCT_VERBATIM:
        target = arguments.out / name
        if (arguments.source / name).resolve() != target.resolve():
            shutil.copyfile(arguments.source / name, target)
        print('  %-42s %6.1f MB' % (name, target.stat().st_size / 1e6))

    total = sum(path.stat().st_size for path in arguments.out.iterdir() if path.is_file())
    print('total: %.1f MB' % (total / 1e6))


if __name__ == '__main__':
    main()
