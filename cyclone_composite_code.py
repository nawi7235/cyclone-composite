# ---------------------------------------
# Cyclone-Centered Composite Generator
# ---------------------------------------
#
# This "psuedo" script constructs a mean composite of cyclone-centered meridional moist static energy (MSE_v)
# from ERA5 reanalysis data. Using this script, you can also interpolate other meteorological fields
# (i.e. geopotential, relative vorticity, etc). The main steps are:
#
# 1. Load pre-processed cyclone track data.
# 2. For each cyclone event:
#    a. Identify the corresponding meteorological field(s).
#    b. Create a cyclone-centered Cartesian grid (a synoptic-scale distance from the center).
#    c. Convert Cartesian offsets to lat/lon using great-circle distances.
#    d. Interpolate the meteorological field onto the cyclone-relative grid (linear in this script).
#    e. Accumulate these fields into a composite.
# 3. Average over all selected cyclones to produce the final composite field.
#
# The script uses pyproj for spherical geometry and scipy.griddata for interpolation.
#
# Author: Nadiyah Williams
# Last updated: 2025-07-18

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
from pyproj import Geod
from scipy.interpolate import griddata

# Load cyclone track data (from preprocessed CSV)
cyclone_data = pd.read_csv('/path/to/cyclone/data', index_col=0)

# Define Cartesian grid offsets (used to center fields around cyclone center)
dist_km = 1000
spacing_km = 25
x = np.arange(-dist_km, dist_km + spacing_km, spacing_km)
y = np.arange(-dist_km, dist_km + spacing_km, spacing_km)
xx, yy = np.meshgrid(x, y)

# Geod object for spherical distance calculations
geod = Geod(ellps="WGS84")

def offset_to_latlon(lat0, lon0, x_km, y_km):
    """
    Convert cyclone-centered x/y offsets (in km) into absolute lat/lon positions.
    """
    azimuth = np.rad2deg(np.arctan2(x_km, y_km))  # angle from cyclone center
    distance = np.sqrt(x_km**2 + y_km**2) * 1000  # convert to meters
    lon_new, lat_new, _ = geod.fwd(lon0*np.ones_like(x_km), lat0*np.ones_like(y_km), azimuth, distance)
    return lat_new, lon_new, azimuth

def interpolate_to_cyclone_center_pyresample(heat_field, lats_1d, lons_1d, lat0, lon0):
    """
    Interpolates a latitude-longitude gridded field to a cyclone-centered Cartesian grid.
    Uses scipy's griddata (linear interpolation) for simplicity.
    """
    # Step 1: Get cyclone-relative lat/lon grid
    lat_grid, lon_grid, azimuth = offset_to_latlon(lat0, lon0, xx, yy)

    # Step 2: Prepare interpolation inputs
    lon2d, lat2d = np.meshgrid(lons_1d, lats_1d)
    points = np.column_stack((lat2d.ravel(), lon2d.ravel()))
    values = heat_field.ravel()
    interp_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))

    # Step 3: Interpolate and reshape to grid
    interp_values = griddata(points, values, interp_points, method='linear', fill_value=np.nan)
    return interp_values.reshape(xx.shape), azimuth

# Define the path to AHT files (already interpolated fields centered on ERA5 grid)
path_to_field = '/path/to/metfield'

# Initialize composite grid
composite = np.zeros_like(xx)
heat_vals = []
count = 0

# Loop through selected cyclones (example slice for debugging)
for _, row in cyclone_data.iterrows():
    time_str = row['Time']
    if pd.isna(time_str):
        print("Skipping row due to missing time value.")
        continue

    time_str = str(time_str)
    hour = int(time_str[11:13])
    filename = f"AHT{time_str[:4]}{time_str[5:7]}{time_str[8:10]}{hour}Z.nc"
    file_path = os.path.join(path_to_field, filename)

    # Load sample ERA5 moist static energy field
    ahtfile = xr.open_dataset(file_path)
    mse = ahtfile['MSE_v'].values

    # Recenter longitudes to [-180, 180]
    lon_era5 = ((ahtfile['lon'].values + 180) % 360) - 180

    lat0, lon0 = row['Lat'], row['Lon']

    # Interpolate MSE field to cyclone-centered Cartesian grid
    heat_interp, azimuth = interpolate_to_cyclone_center_pyresample(
        mse, ahtfile['lat'].values, lon_era5, lat0, lon0)

    # Add to composite
    composite += heat_interp.astype(np.float64)
    heat_vals.append(heat_interp)
    count += 1

# Compute final average composite
composite_avg = composite / count
