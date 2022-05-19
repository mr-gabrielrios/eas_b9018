#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radar handling script

See Ryan May's great tutorial on this: https://nbviewer.org/gist/dopplershift/356f2e14832e9b676207

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

def dist_to_coords(nws_coords, dx, dy):
    lat = nws_coords[0] + (180/np.pi)*(dy/6378137)
    lon = nws_coords[1] + (180/np.pi)*(dx/6378137)/np.cos(nws_coords[0] * np.pi/180)
    
    return [lat, lon]

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
    ref = raw_to_masked_float(ref_var, ref_data).filled(np.nan)
    x, y = polar_to_cartesian(az, rng)
    # Convert distance data to coordinate data
    x_, y_ = x.ravel(), y.ravel()
    lons, lats = [None]*len(x_), [None]*len(y_)
    for i in range(0, len(x_)):
        lons[i], lats[i] = dist_to_coords([lat, lon], x_[i], y_[i])
        
    # Build xArray Dataset
    ds = xr.Dataset(
        data_vars={
            'reflectivity': (['x', 'y'], ref.data)
            },
        coords={
            'lon': (['x', 'y'], np.reshape(lons, ref.data.shape)),
            'lat': (['x', 'y'], np.reshape(lats, ref.data.shape))
        })
    ds = ds.assign_coords(t=date)
    ds = ds.expand_dims('t')
    
    return ds, ref, station_name

def plotting(data, center_nws, center_event, date, station, aux=None, norm=False):
    
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
    
    param = 'reflectivity' if not norm else 'normalized_reflectivity'
    
    if aux:
        lat = aux['lat']
        lon = aux['lon']
        ref = data[param]
        vmin, vmax = -2, 2
    else:
        lat = data['lat']
        lon = data['lon']
        ref = data[param]

    # Plot data
    im = ax.pcolormesh(lat, lon, ref, 
                       vmin=vmin, vmax=vmax, cmap='Spectral_r', zorder=0,
                       transform=ccrs.PlateCarree())
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
             datetime.datetime(2020, 9, 6, 0)]
    
    hours = [0, 18, 21]
    
    datasets = []
    
    for date in dates:
        # Retrieve data
        for hour in hours:
            date = datetime.datetime(date.year, date.month, date.day, hour)
            ds, ref, station_name = data_query(date, lon_nws, lat_nws)   
            datasets.append(ds)
        
    data = xr.concat(datasets, dim='t')