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
    print(wcsfile)

    dirn = os.path.dirname(fname)
    print("Plate-solving in the background")
    return subprocess.Popen(["solve-field", "--overwrite", "-z2", "-L3", "-H7",
        "-Nnone", "--match", "none", "--rdls", "none", "--corr", "none", "--solved", "none", "--index-xyls", "none",
         "-p", f"-D{dirn}", fname])

def cleanup(dir):
    subprocess.run(["rm"] + glob(os.path.join(dir, "*")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("N", type=int, help="number of exposures to take")

    parser.add_argument("--dir", default="tmp/", help="Directory for temporary files")
    parser.add_argument("--wait", type=float, default=15, help="Time in seconds to wait between exposures")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup temporary file directory before taking exposures")

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    if args.cleanup:
        print("Cleaning working directory")
        cleanup(args.dir)

    print(f"Taking {args.N} exposures.")

    for i in range(args.N):
        fname = take_exposure(i, args.dir)
        process = plate_solve(fname)

        if i != args.N - 1:
            time.sleep(args.wait)

    process.communicate()
