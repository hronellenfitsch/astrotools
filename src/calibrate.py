#!/usr/bin/env python3

import rawpy
import imageio

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from astropy.io import fits

import argparse
import os.path

def weighted_var(values, weights):
    """
    Return the weighted variance.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average)**2, weights=weights)
    return variance

def optimize_dark_numerical(light, dark):
        # try numerical optimization
        fun = lambda alpha: weighted_var(np.round(np.clip((light - alpha*dark).flatten(), 0, np.inf)),
                                                dark.flatten())
        r = minimize_scalar(fun, [0, 2])

        if r.success:
                print(f"Successful optimization.")
        else:
                print("Optimization not successful.")
                print(r)
        return r.x

def optimize_dark_fast(light, dark):
        """ Find optimal dark frame scaling by minimizing
        global dark current-weighted image variance.
        """
        light, dark = light.flatten(), dark.flatten()

        light_avg = np.average(light, weights=dark)
        dark_avg = np.average(dark, weights=dark)

        num = np.average((light - light_avg)*(dark - dark_avg), weights=dark)
        den = np.average((dark - dark_avg)**2, weights=dark)

        return num/den

def calibrate_light(fname, dark, bias=None, slow=False):
        # load RAW file
        print(f"Analyzing RAW file {fname}")
        with rawpy.imread(fname) as raw:
                light = np.array(raw.raw_image, dtype=int)

        # calibrate with bias if applicable
        if bias is not None:
                print("Calibrating with bias...")
                light = np.clip(light - bias, 0, np.inf)

        # find optimal dark frame factor
        print("Optimizing dark frame...")
        if slow:
                alpha = optimize_dark_numerical(light, dark)
        else:
                alpha = optimize_dark_fast(light, dark)

        print(f"Dark factor α = {r.x:.4}")
        return np.clip(light - alpha*dark, 0, np.inf)

def save_fits(fname, img):
        """ Save original raw fname as calibrated FITS
        """
        path, ext = os.path.splitext(fname)

        fname_cal = path + ".cal.fts"

        print(f"Saving calibrated FITS {fname_cal}")

        # save as 16-bit unsigned integer
        hdu = fits.PrimaryHDU(np.array(img, dtype=np.uint16))
        hdu.writeto(fname_cal, overwrite=True)

if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument("files", help="file names of raw images to analyze", nargs="+")
        parser.add_argument("--master-dark", "-d", required=True, help="master dark .tif")
        parser.add_argument("--calibrate-dark", action="store_true", help="Calibrate the master dark by subtracting the master bias. Master darks from DeepSkyStacker are already calibrated if created with (master) bias frames")
        parser.add_argument("--master-bias", "-b", default=None, help="master bias .tif")
        parser.add_argument("--dont-save", action="store_true", help="don't save calibrated FITS")
        parser.add_argument("--slow", action="store_true", help="Use slow but potentially more accurate numerical optimization.")

        args = parser.parse_args()

        # load calibration files as int64
        dark = np.array(imageio.imread(args.master_dark), dtype=int)

        if args.master_bias is not None:
                bias = np.array(imageio.imread(args.master_bias), dtype=int)

                # calibrate master dark frame if desired
                if args.calibrate_dark:
                        print("Calibrating master dark frame...")
                        dark = np.clip(dark - bias, 0, np.inf)
        else:
                bias = None

        # loop over files and calibrate them
        for fname in args.files:
                light_c = calibrate_light(fname, dark, bias, slow_mode=args.slow)
                save_fits(fname, light_c)
                print()
