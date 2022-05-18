#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radar handling script

@author: gabriel
"""

import numpy as np
import xarray as xr

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt

import os
import warnings
import datetime
from siphon.radarserver import RadarServer

def raw_to_masked_float(var, data):
    # Values come back signed. If the _Unsigned attribute is set, we need to convert
    # from the range [-127, 128] to [0, 255].
    if var._Unsigned:
        data = data & 255

    # Mask missing points
    data = np.ma.array(data, mask=data==0)

    # Convert to float using the scale and offset
    return data * var.scale_factor + var.add_offset

def polar_to_cartesian(az, rng):
    az_rad = np.deg2rad(az)[:, None]
    x = rng * np.sin(az_rad)
    y = rng * np.cos(az_rad)
    return x, y

def data_query(date, lon, lat):
    # Open instance of a RadarServer to access data
    rs = RadarServer('http://tds-nexrad.scigw.unidata.ucar.edu/thredds/radarServer/nexrad/level2/S3/')
    # Fetch the server query
    query = rs.query()
    # Query the point of interest at the specified time
    query.lonlat_point(lon, lat).time_range(date, 
                                            date + datetime.timedelta(hours=1))
    # Pull list of datasets that match the specifications
    cat = rs.get_catalog(query)
    station_name = cat.datasets[0].name[0:4]
    data = cat.datasets[0].remote_access()
    
    # Pull out the data of interest
    sweep = 0
    ref_var = data.variables['Reflectivity_HI']
    ref_data = ref_var[sweep]
    rng = data.variables['distanceR_HI'][:]
    az = data.variables['azimuthR_HI'][sweep]
    ref = raw_to_masked_float(ref_var, ref_data)
    x, y = polar_to_cartesian(az, rng)
    
    # Correction for NWS Hanford
    
    
    return [x, y, ref], station_name

def plotting(data, center_nws, center_event, date, station):
    
    x, y, ref = data
    
    tol = 0.5
    center_lat, center_lon = center_nws[0], center_nws[1]
    # Define projection
    proj_ortho = ccrs.Orthographic(central_latitude=center_lat, 
                                   central_longitude=center_lon)
    proj_pc = ccrs.PlateCarree()
    # Define bounds
    vmin, vmax = -20, 60

    # Figure formatting
    fig, ax = plt.subplots(dpi=300, subplot_kw={'projection': proj_ortho})
    ax.set_extent([center_lon-tol, center_lon+tol, center_lat-tol, center_lat+tol])
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot data
    im = ax.pcolormesh(x, y, ref, vmin=vmin, vmax=vmax, cmap='Spectral_r', zorder=0)
    colorbar = fig.colorbar(im)
    colorbar.set_label('Equivalent reflectivity [dBZ]', rotation=270, labelpad=20)
    # Replace the extent with the proper location
    center_lat, center_lon = center_event[0], center_event[1]
    ax.set_extent([center_lon-tol, center_lon+tol, center_lat-tol, center_lat+tol])
    # Plot metadata
    ax.set_title(date.strftime('%Y-%m-%d %H:%M:%S'), fontsize=10)
    fig.suptitle('{0} radar reflectivity'.format(station),
                 y=1, x=0.54)
    
if __name__ == '__main__':
    
    coords = {'Creek Fire': [37.201, -119.272],
              'NWS Hanford': [36.314, -119.632]}
    
    coords_event = coords['Creek Fire']
    coords_nws = coords['NWS Hanford']
    
    # Initialize variables
    lon_event, lat_event = coords_event[1], coords_event[0]
    lon_nws, lat_nws = coords_nws[1], coords_nws[0]
    
    dates = [datetime.datetime(2020, 9, 5, 0),
             datetime.datetime(2020, 9, 5, 12),
             datetime.datetime(2020, 9, 5, 15),
             datetime.datetime(2020, 9, 5, 18),
             datetime.datetime(2020, 9, 5, 21)]
    
    for date in dates:
        # Retrieve data
        data, station_name = data_query(date, lon_nws, lat_nws)   
        
        # Plot the data
        plotting(data, [lat_nws, lon_nws], [lat_event, lon_event], date, station_name)