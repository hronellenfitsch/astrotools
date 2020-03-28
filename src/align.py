#!/usr/bin/env python3

import io
import subprocess
import argparse
import os.path
import datetime
import tempfile
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta, TimezoneInfo
from astropy.wcs.utils import proj_plane_pixel_scales

from glob import glob

from analyze import *
from capture import *

def utc_offset():
    # return the UTC offset of the current time in hours
    offset = datetime.datetime.utcnow().astimezone().utcoffset()
    return offset.total_seconds()/3600

def expose_and_solve(args, i, dir, here, ra_guess=None, dec_guess=None):
    """ Take one exposure and put it into dir/i.ext
    Then use the information in the location here to
    plate solve and extract sky coordinates
    """
    fname = take_exposure(i, dir)
    # fname = f"tmp/{i}.jpg" #DEBUG for local data
    plate_solve(fname, ra=ra_guess, dec=dec_guess).communicate()

    # use EXIF data to estimate when this image was taken
    mid_time = approximate_exposure_time(fname, args.utc_offset, here)

    # Ra-Dec coordinates
    radec, pix_scale = plate_solve_data(fname)

    # Cartesian for fitting
    coords = radec_to_cartesian(radec, mid_time, here)

    return {"fname": fname, "mid_time": mid_time, "coords": coords, "pixel_scale": pix_scale,
    "radec": radec}

def estimate_axis(args, here, mid_times, ys, pix_scale):
    """ Estimate the mount axis and print the guidance message
    """
    dts = [(t2 - t1).sec for t1, t2 in zip(mid_times[:-1], mid_times[1:])]
    res = find_m(here, ys, dts)
    std_errors = fit_errors(res)

    print("Optimization: ", res.message)
    print_guidance(here, res, args.pixel_threshold, pix_scale*u.arcsec/u.pixel, std_errors, ys[-1])

    return dts

def polar_align(args, tmpdir):
    """ Perform polar alignment estimation using the given args.
    """
    here = EarthLocation(lat=args.lat*u.deg, lon=args.lon*u.deg, height=args.height*u.m)
    fnames = []
    mid_times = []
    dts = []
    ys = []
    pixel_scales = []

    print("Reference exposure...")
    exposure = expose_and_solve(args, 0, tmpdir, here)
    fnames.append(exposure["fname"])
    mid_times.append(exposure["mid_time"])
    ys.append(exposure["coords"])
    pixel_scales.append(exposure["pixel_scale"])

    # analysis loop
    for i in range(1, args.N):
        print(f"Waiting {args.wait} seconds...")
        time.sleep(args.wait)

        # does not seem to work well
        # ra_guess = exposure["radec"].ra.to(u.deg) # guess from last exposure for faster plate solve
        # dec_guess = exposure["radec"].dec.to(u.deg)
        ra_guess, dec_guess = None, None

        # take exposure
        exposure = expose_and_solve(args, i, tmpdir, here,
            ra_guess=ra_guess, dec_guess=dec_guess)
        fnames.append(exposure["fname"])
        mid_times.append(exposure["mid_time"])
        ys.append(exposure["coords"])
        pixel_scales.append(exposure["pixel_scale"])

        # fit using current data
        cur_pix_scale = np.mean(pixel_scales)
        dts = estimate_axis(args, here, mid_times, ys, cur_pix_scale)

        if args.estimate_drift:
            print("Estimating drift...")
            Ys = [read_luma(fname) for fname in fnames]
            print_empirical_drift(Ys, dts, cur_pix_scale*u.arcsec/u.pixel, args.pixel_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("N", type=int, help="number of exposures to take")

    parser.add_argument("--lat", default=42.368144, help="Observer latitude in degrees")
    parser.add_argument("--lon", default=-71.081818, help="Observer longitude in degrees")
    parser.add_argument("--utc-offset", default=utc_offset(), help="Observer offset from UTC. Per default taken from the system clock")
    parser.add_argument("--height", default=9, help="Observer height above sea level in meters")
    parser.add_argument("-p", "--pixel-threshold", default=2, type=float, help="max pixel shift allowed during exposures")
    parser.add_argument("--estimate-drift", action="store_true", help="estimate empirical drift between alignment exposures using image overlaps")
    parser.add_argument("--L", default=4, type=float, help="approximate length-wise image scale in degrees")
    parser.add_argument("--H", default=3, type=float, help="approximate height-wise image scale in degrees")
    parser.add_argument("--wait", type=float, default=10, help="Time in seconds to wait between exposures")

    args = parser.parse_args()

    # we will work with a temporary directory that gets cleaned up automatically
    with tempfile.TemporaryDirectory() as tmpdir:
        polar_align(args, tmpdir)
