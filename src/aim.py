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

def annotate(fname):
    path, ext = os.path.splitext(fname)

    wcsfile = path + ".wcs"
    outfile = path + "_ann.jpg"

    return subprocess.Popen(["plotann.py",
    wcsfile, fname, "--grid-size=1", "--scale=0.2", outfile])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--aim-file", default="", type=str, help="Aim this file instead of taking exposure")
    parser.add_argument("--L", default=3.0, type=float, help="approximate length-wise image scale in degrees")
    parser.add_argument("--H", default=7.0, type=float, help="approximate height-wise image scale in degrees")

    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.aim_file == "":
            fname = take_exposure(0, tmpdir)
        else:
            fname = args.aim_file

        process = plate_solve(fname, L=args.L, H=args.H)
        process.communicate()

        process = annotate(fname)
        process.communicate()

        # display image
        path, ext = os.path.splitext(fname)
        aimed_fname = path + "_ann.jpg"
        subprocess.run(["xdg-open", aimed_fname])
