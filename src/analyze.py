#!/usr/bin/env python3

import io
import subprocess
import argparse
import os.path

from glob import glob

import rawpy
import imageio
import gphoto2
from skimage.feature import register_translation

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta, TimezoneInfo
from astropy.wcs.utils import proj_plane_pixel_scales

import exifread

import numpy as np
import scipy as sp

from scipy.optimize import least_squares

def read_luma(fname):
    try:
        with rawpy.imread(fname) as raw:
            rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
    except:
        rgb = imageio.imread(fname)

    # something like a luminance
    Y = (rgb[:,:,0] * 0.3) + (rgb[:,:,1] * 0.59) + (rgb[:,:,2] * 0.11)

    # crop
    N, M = Y.shape
    return Y[int(N/3):int(2*N/3), int(M/3):int(2*M/3)]

# find exact time of exposure
def exif_to_astropy(t):
    return t[0:4] + "-" + t[5:7] + "-" + t[8:]

def approximate_exposure_time(fname, utc_offset, loc):
    # find time of exposure
    with open(fname, 'rb') as f:
        tags = exifread.process_file(f)

    start_time = exif_to_astropy(tags["EXIF DateTimeOriginal"].printable)
    exp_time = tags["EXIF ExposureTime"]

    num, den = exp_time.values[0].num, exp_time.values[0].den
    exp_time = num/den

    utc_offset = utc_offset*u.hour

    s_time = Time(start_time, location=loc, scale='utc')
    e_time = TimeDelta(exp_time*u.s)

    # approximate time of exposure in UTC
    return s_time + 0.5*e_time - utc_offset

def plate_solve_data(fname):
    # save intermediate file
    # with open("tmp/tmp.png", "wb") as f:
    #     imageio.imwrite(f, Y, "png")

    path, ext = os.path.splitext(fname)

    wcsfile = path + ".wcs"
    if not os.path.exists(wcsfile):
        # do plate solve
        dirn = os.path.dirname(fname)
        subprocess.run(["solve-field", "--overwrite", "-L3", "-H7", "-z2",
        "-Nnone", "--match", "none", "--rdls", "none", "--corr", "none", "--solved", "none", "--index-xyls", "none",
        "-p", f"-D{dirn}", fname])

    # Read world coordinate system
    w = WCS(wcsfile)

    # n, m = Y.shape
    try:
        with rawpy.imread(fname) as raw:
            sz = raw.sizes
            n, m = sz.height, sz.width
    except:
        n, m, _ = imageio.imread(fname).shape

    # field center coordinates, offset=0
    radec = w.all_pix2world(m/2, n/2, 0)

    ra = float(radec[0])
    dec = float(radec[1])

    print(f"Detected center of image (Ra, Dec) = {ra}, {dec}")
    print(f"Pixel scale: {pixel_scale(w):.4} arcsec/px")

    return SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

def pixel_scale(w):
    arcsec_per_pix = (proj_plane_pixel_scales(w)[0]*(u.degree/u.pixel)).to(u.arcsec/u.pixel)
    return arcsec_per_pix.value

def plate_solve(Y):
    """ Plate solve image using astrometry.net and return the celestial
    coordinates of the center
    """
    # save intermediate file
    with open("tmp/tmp.png", "wb") as f:
        imageio.imwrite(f, Y, "png")

    subprocess.run(["solve-field", "--overwrite", "-p",
    "-Nnone", "--match", "none", "--rdls", "none", "--corr", "none", "--solved", "none",
    "-Dtmp", "tmp/tmp.png"])

    # Read world coordinate system
    w = WCS("tmp/tmp.wcs")

    n, m = Y.shape

    # field center coordinates, offset=0
    radec = w.wcs_pix2world(m/2, n/2, 0)

    ra = float(radec[0])
    dec = float(radec[1])

    return SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

def altaz_to_cartesian(alt, az):
    """ Converts alt-az coordinates into right-handed Cartesian coordinates
    where the x-axis points North, the y-axis points West, and the z-axis points
    to the Zenith
    """
    return np.array([
        np.cos(alt)*np.cos(az),
        -np.cos(alt)*np.sin(az),
        np.sin(alt)
    ])

def cartesian_to_altaz(x):
    """ Converts local Cartesian coordinates to Alt-az,
    inverting altaz_to_cartesian
    """
    x, y, z = x
    return np.arcsin(z), np.arctan2(-y, x)

def radec_to_cartesian(c, time, loc):
    """ Converts ra-dec SkyCoord to Local Cartesian
    """
    altaz_pos = c.transform_to(AltAz(obstime=time, location=loc))

    alt = altaz_pos.alt.radian
    az = altaz_pos.az.radian

    return altaz_to_cartesian(alt, az)

# rotate
def K_mat(e):
    """ Rotation matrix generator
    """
    ex, ey, ez = e

    return np.array([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])

def R_mat(e, θ):
    """ Rotation matrix by angle θ around axis e
    """
    K = K_mat(e)
    return np.eye(3) + np.sin(θ)*K + (1 - np.cos(θ))*np.dot(K, K)

# try to infer the axis m
def find_m(ys, dts, resample=False):
    """ Fit the mount axis m to the data
    """
    T = 86164.0905 # sidereal day
    Ω = 2*np.pi/T

    if resample:
        """ Bootstrap resampling
        """
        sample_idxs = np.random.choice(len(ys) - 1, len(ys) - 1)
    else:
        sample_idxs = np.arange(len(ys) - 1)

    def residuals(coords):
        alt, az = coords
        m = altaz_to_cartesian(alt, az)

        # rotate y's
        y_rots = []
        for y, dt in zip(ys[0:-1], dts):
            R = R_mat(m, -Ω*dt)
            y_rots.append(np.dot(R, y))

        return np.concatenate([y1 - y_rot for i, (y1, y_rot) in enumerate(zip(ys[1:], y_rots))
                              if i in sample_idxs])

    return least_squares(residuals, np.array([here.lat.radian, 0]))

def fit_errors(res):
    # estimate standard error of (alt, az) fit
    jac = res.jac
    H = jac.T.dot(jac) # Hessian of the cost function

    n, p = jac.shape
    cov = 2*res.cost/(n - p)*np.linalg.inv(H) # estimate for the covariance matrix. factor 2 is because data points are constrained to lie on the unit sphere

    # standard error estimates for alt and az
    return np.sqrt(np.diag(cov))

def print_guidance(here, res, pixel_threshold, pixel_scale):
    n_ax = altaz_to_cartesian(here.lat.radian, 0) # true polar axis
    m_ax = altaz_to_cartesian(res.x[0], res.x[1]) # estimated mount axis

    print(f"Estimated axis: [alt, az] = {res.x*180/np.pi}")

    angular_mismatch = np.arccos(np.dot(m_ax, n_ax))*180/np.pi

    alt_mismatch = (res.x[0] - here.lat.radian)*180/np.pi
    az_mismatch = res.x[1]*180/np.pi

    print(f"Angle between mount and polar axis: {angular_mismatch:.3} deg")
    print(f"Altitude error: {alt_mismatch:.3} +- {std_errors[0]*180/np.pi:.3} deg")
    print(f"Azimuth error: {az_mismatch:.3} +- {std_errors[1]*180/np.pi:.3} deg")

    print()
    if alt_mismatch > 0:
        print("Move mount DOWN.")
    else:
        print("Move mount UP.")

    if az_mismatch < 0:
        print("Move mount EAST.")
    else:
        print("Move mount WEST.")

    print("")

    if np.abs(angular_mismatch) < 1e-6:
        print("WARNING: Angular mismatch is tiny-- you are probably too well aligned for the given time between exposures. Increase --wait-time!")

    # estimate drift using fitted mount axis
    T = 86164.0905 # sidereal day
    Ω = 2*np.pi/T

    y_last = ys[-1]

    # estimate angular drift velocity
    v_px = Ω*np.linalg.norm(np.cross(m_ax - n_ax, y_last))
    pixels_est = lambda t: ((v_px*u.radian).to(u.arcsec)/pixel_scale).value*t

    max_time = pixel_threshold/v_px*pixel_scale.to(u.radian/u.pixel).value

    print(f"Estimated drift velocity: {v_px/pixel_scale.to(u.radian/u.pixel).value:.2} px/sec")
    print(f"Estimated maximum exposure time for a drift of {pixel_threshold} px: {round(max_time, 1)} sec")

def print_empirical_drift(Ys, dts):
    # calculate empirical pixel drift velocity
    empirical_drift = [register_translation(Y1, Y2, 150)[0] for Y1, Y2 in zip(Ys[:-1], Ys[1:])]
    measured_drift = [np.linalg.norm(d/dt) for d, dt in zip(empirical_drift, dts)]

    drift_mean = np.mean(measured_drift)
    drift_std = np.std(measured_drift, ddof=1)
    print(measured_drift)
    print(f"Measured drift velocity: {drift_mean:.2} +- {drift_std:.1} px/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("files", help="file names of EXIF-tagged raw images to analyze", nargs="+")

    parser.add_argument("--lat", default=42.368172, help="Observer latitude in degrees")
    parser.add_argument("--lon", default=-71.081802, help="Observer longitude in degrees")
    parser.add_argument("--utc-offset", default=-5, help="Observer offset from UTC")
    parser.add_argument("--height", default=9, help="Observer height above sea level in meters")
    parser.add_argument("-p", "--pixel-threshold", default=2, type=float, help="max pixel shift allowed during exposures")
    parser.add_argument("--estimate-drift", action="store_true", help="estimate drift between alignment exposures")
    parser.add_argument("--pixel-scale", default=4.02, type=float, help="pixel scale of exposures in px/arcsec")

    args = parser.parse_args()

    # Do analysis
    here = EarthLocation(lat=args.lat*u.deg, lon=args.lon*u.deg, height=args.height*u.m)

    # average time when exposures were taken
    mid_times = [approximate_exposure_time(fname, args.utc_offset, here) for fname in args.files]

    # time between exposures
    dts = [(t2 - t1).sec for t1, t2 in zip(mid_times[:-1], mid_times[1:])]

    # cropped luminance from raws
    # Ys = [read_luma(fname) for fname in args.files]

    # plate-solved coordinates of center of exposures
    cs = [plate_solve_data(fname) for fname in args.files]
    ys = [radec_to_cartesian(c, mid_time, here) for c, mid_time in zip(cs, mid_times)]

    # Fit position of mount axis
    res = find_m(ys, dts)
    std_errors = fit_errors(res)

    print(res.message)

    print_guidance(here, res, args.pixel_threshold, args.pixel_scale*u.arcsec/u.pixel)

    if args.estimate_drift:
        print("Estimating drift...")
        Ys = [read_luma(fname) for fname in args.files]
        print_empirical_drift(Ys, dts)
