# astrotools

This is a set of Python 3 scripts that I use to aid in low focal-length
DSLR astrophotography with my Linux laptop.

The project was started because from my backyard, Polaris is obscured by
a building, and I wanted a means of polar aligning my mount that was more
convenient than drift alignment.
The result of this are the two scripts capture.py and analyze.py,
which use libgphoto to take pictures through a DSLR attached to the computer
and pointed at the sky.
These pictures are then plate-solved to determine the exact pointing-position in
the sky, and a simple rotational model is fitted to the data to obtain
the exact alignment of the mount axis in a local alt-azimuthal frame.
This location is then compared to the theoretical polar axis given by the
local latitude, and a recommendation for adjusting the mount axis is given.

While the mount model is very simple and does not take into account complicating
factors, I have been happy with the results, at least up to a focal length of 200mm
and with exposures of 2-3 minutes. I have not tried to push the system to its limits.
Your mileage may vary. I am happy about feedback that would allow me to improve the model.

## Requirements
  * Python >= 3.6
  * astropy
  * rawpy
  * imageio
  * exifread
  * local install of astrometry.net (in particular, the solve-field and plotann.py tools must be callable)
  * libgphoto2 and python bindings

## Polar alignment with capture.py and analyze.py
Make sure your mount is roughly polar aligned.
Attach your camera via USB and set the desired exposure length. I have found 10s to work well.
You may also want to shoot in JPEG instead of RAW to speed up
the plate-solving process.

Point you camera at a region of the sky close to the celestial equator
and somewhere halfway between the horizon and the meridian, such that
errors in altitude and azimuth alignment are about equal in magnitude.

To obtain, e.g., 3 measurement exposures with 20 seconds of wait
time between them, run

    ./capture.py --wait 20 3

The resulting files will be put in the ./tmp folder and automatically
plate-solved.
Make sure that your camera's time is set correctly, since EXIF data will
be used to convert the plate-solved coordinates into a local frame.

Next, we analyze them and fit the mount model. For this, run

    ./analyze.py --lat your_latitude --lon your_longitude --height your_height --utc-offset your_offset --pixel-scale your_pixel_scale tmp/\*.jpg

For instance, during normal time in the EST time zone, the utc_offset would
be -5. For my camera with a 200mm lens, the pixel scale is approx
4.02 px/arcsec. MIT's Killian court is about 9m above sea level.
So an example input might look like

    ./analyze.py --lat 42.358991 --lon -71.091619 --height 9 --utc-offset -5 --pixel-scale 4.02

The script will analyze the files and tell you its estimate of the mount position,
alignment error, and estimated pixel drift. After making adjustments to the mount, run
the two scripts again until satisfied.

## Aiming your camera using aim.py
After successful polar alignment you want to aim your camera at your target.
To make your that your pointing is really correct, you can use aim.py
to plate-solve a camera image and display an overlay with any deep-sky
objects in the field of view. Again, this is just a convenience script which uses
astrometry.net's solve-field and plotann.py internally.

Make sure your camera is plugged in and desired exposure settings are set.
Again, 10s seems to work well.
Then, run

    ./aim.py

to take an exposure, plate-solve, and display an overlay.

## Mathematical model

The mathematical model used for fitting is described in the file MathematicalModel.ipynb
