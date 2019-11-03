#!/usr/bin/env python3

import sys
import time
import os
import os.path
import subprocess
from glob import glob

import argparse

import gphoto2 as gp

def take_exposure(i, dir):
    print(f"Exposure {i+1}...")
    camera = gp.Camera()
    camera.init()
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)

    _, ext = os.path.splitext(file_path.name)
    target = os.path.join(dir, f'{i}{ext}')

    camera_file = camera.file_get(
        file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)

    camera_file.save(target)
    camera.exit()

    return target

def plate_solve(fname):
    path, ext = os.path.splitext(fname)

    wcsfile = path + ".wcs"

    dirn = os.path.dirname(fname)
    print("Plate-solving...")

    return subprocess.Popen(["solve-field", "-z2", "-Nnone", "-L3", "-H7",
    "--match", "none", "--rdls", "none", "--corr", "none", "--solved", "none", "--index-xyls", "none",
    "--overwrite", "--no-plots", f"-D{dirn}", fname])

def annotate(fname):
    path, ext = os.path.splitext(fname)

    wcsfile = path + ".wcs"
    outfile = path + "_ann.jpg"

    return subprocess.Popen(["plotann.py",
    wcsfile, fname, "--grid-size=1", "--scale=0.2", outfile])

def cleanup(dir):
    subprocess.run(["rm"] + glob(os.path.join(dir, "*")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default="aim_tmp/", help="Directory for temporary files")
    parser.add_argument("--aim-file", default="", type=str, help="Aim this file instead of taking exposure")

    args = parser.parse_args()

    print("Cleaning working directory")
    os.makedirs(args.dir, exist_ok=True)
    cleanup(args.dir)

    if args.aim_file == "":
        fname = take_exposure(0, args.dir)
    else:
        fname = args.aim_file

    process = plate_solve(fname)
    process.communicate()

    process = annotate(fname)
    process.communicate()

    # display image
    path, ext = os.path.splitext(fname)
    aimed_fname = path + "_ann.jpg"
    subprocess.run(["xdg-open", aimed_fname])
