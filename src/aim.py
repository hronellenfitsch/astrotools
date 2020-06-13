#!/usr/bin/env python3

import sys
import time
import os
import os.path
import subprocess
import tempfile
import argparse

from glob import glob

from capture import take_exposure, plate_solve

def annotate(fname, target=None, brightness=1.0):
    path, ext = os.path.splitext(fname)

    wcsfile = path + ".wcs"
    outfile = path + "_ann.jpg"

    cmd_args = []

    if target is not None:
        cmd_args += ["-T"] + target

    return subprocess.Popen(["plotann.py",
    wcsfile, fname, "--grid-size=1", "--scale=0.2"] + cmd_args + [outfile])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--aim-file", default="", type=str, help="Aim this file instead of taking exposure")
    parser.add_argument("--L", default=3.0, type=float, help="approximate length-wise image scale in degrees")
    parser.add_argument("--H", default=4.0, type=float, help="approximate height-wise image scale in degrees")
    parser.add_argument("--brightness", default=2.0, type=float,
        help="Adjust brightness of the exposure by this factor to be able to see fainter stars")
    parser.add_argument("-T", "--target", default=None, nargs=3, help="target name and ra-dec to plot.")

    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.aim_file == "":
            fname = take_exposure(0, tmpdir)
        else:
            fname = args.aim_file

        process = plate_solve(fname, L=args.L, H=args.H)
        process.communicate()

        process = annotate(fname, target=args.target, brightness=args.brightness)
        process.communicate()

        # display image
        path, ext = os.path.splitext(fname)
        aimed_fname = path + "_ann.jpg"
        subprocess.run(["xdg-open", aimed_fname])

        time.sleep(50)
