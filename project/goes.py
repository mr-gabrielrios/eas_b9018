#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOES handling script

@author: gabriel
"""

''' Imports '''
# Visualization
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.pyplot as plt
# Numerical processingv
import numpy as np
import xarray as xr
# Miscellaneous
import datetime
import os


def basemap(center, tol=0.5, sat_img=False):
    # Get map center point
    center_lat, center_lon = center[0], center[1]
    # Define projection
    proj_ortho = ccrs.Orthographic(central_latitude=center_lat, central_longitude=center_lon)
    proj_pc = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(subplot_kw={'projection': proj_ortho})
    ax.set_extent([center_lon-tol, center_lon+tol, center_lat-tol, center_lat+tol])
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # Import satellite data
    if sat_img:
        sat_img = cimgt.GoogleTiles(style='satellite')
        ax.add_image(sat_img, 7)
        
    return fig, ax

def goes_pull(year, month, day, satellite='G17', product=None, band=None):
    
    from goes2go.data import goes_nearesttime

    from datetime import datetime, timedelta
    import pandas as pd

    hours = [0, 3, 9, 15, 18]
    
    img = []
    img_dates = []
    for hour in hours:
        img_date = datetime(year, month, day, hour)
        g = goes_nearesttime(datetime(year, month, day, hour),
                             satellite=satellite,
                             product=product,
                             return_as='xarray',
                             bands=band)
        img.append(g)
        img_dates.append(img_date)
    
    if product == 'GLM':
        return xr.merge(img)
    else:
        return xr.concat(img, dim='t')
    
    
def reprocess(data, dataset_name=None, args=None):
    
    ''' Reprocess coordinate data. '''
    
    import grid_processes as gp
    
    long_name = data.attrs['title']
    
    # GOES-R projection info
    proj_info = data['goes_imager_projection']
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height + proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis    
    lambda_0 = lon_origin * np.pi / 180  # Longitude of origin converted to radians
    
    size_x, size_y = len(data.x), len(data.y)
    
    # GOES-R grid info
    if args:
        [central_lat, central_lon, bound_sz] = args[:]
        [x0, x1, y1, y0] = gp.grid_grab(central_lon, central_lat, bound_sz, H, r_pol, r_eq, lambda_0, size_x, size_y)
        print([x0, x1, y1, y0])
        lat_rad_1d = data['x'][x0:x1]
        lon_rad_1d = data['y'][y0:y1]
        data = data.isel(y=slice(y0, y1), x=slice(x0, x1))
    else:
        lat_rad_1d = data['x'][:]
        lon_rad_1d = data['y'][:]
        
    # Create meshgrid
    lat_rad, lon_rad = np.meshgrid(lat_rad_1d, lon_rad_1d)  # x and y (reference PUG, Section 4.2.8.1)   
    
    # Latitude/longitude projection calculation from satellite radian angle vectors (reference PUG, Section 4.2.8.1)
    a = np.power(np.sin(lat_rad), 2) + np.power(np.cos(lat_rad), 2) * (
                np.power(np.cos(lon_rad), 2) + np.power(np.sin(lon_rad), 2) * np.power(r_eq, 2) / np.power(r_pol, 2))
    b = (-2) * H * np.cos(lat_rad) * np.cos(lon_rad)
    c = np.power(H, 2) - np.power(r_eq, 2)
    r_s = ((-b) - np.sqrt(np.power(b, 2) - 4 * a * c)) / (2 * a)  # distance from satellite to surface

    s_x = r_s * np.cos(lat_rad) * np.cos(lon_rad)
    s_y = (-1 * r_s) * np.sin(lat_rad)
    s_z = r_s * np.cos(lat_rad) * np.sin(lon_rad)    

    # Transform radian latitude and longitude values to degrees
    lat_deg = (180 / np.pi) * np.arctan(
        (np.power(r_eq, 2) / np.power(r_pol, 2)) * s_z / (np.sqrt((np.power((H - s_x), 2)) + np.power(s_y, 2))))
    lon_deg = (180 / np.pi) * (lambda_0 - np.arctan(s_y / (H - s_x)))
    
    
    data['lat'] = (['x', 'y'], lat_deg.T)
    data['lon'] = (['x', 'y'], lon_deg.T)
    
    return data
   
def mapper(data, param, single_time=None, center=None, extent=None, sat_img=False):
    
    if not center:
        center = [np.nanmean([data['lat'].min().values, data['lat'].max().values]),
                  np.nanmean([data['lon'].min().values, data['lon'].max().values])]
    
    # Get extrema
    vmin, vmax = data[param].min().values, data[param].max().values
    # Normalize
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Pick timestep
    if single_time:
        data = data.isel(t=0)
    else:
        for t in range(0, len(data.t)):
            data_ = data.isel(t=t)
            
            fig, ax = basemap(center, sat_img=sat_img)
            im = ax.pcolormesh(data_['lon'], data_['lat'], data_[param].values.T, 
                               transform=ccrs.PlateCarree(), 
                               vmin=vmin, vmax=vmax, cmap='Greys_r')
            colorbar = fig.colorbar(im)
            
            ax.set_title('{0}'.format(data_['time_coverage_start'].values), 
                         fontsize=10)
            fig.suptitle('{0}'.format(data[param].attrs['long_name']),
                         y=1, x=0.54)
            
            plt.gca()
    
if __name__ == '__main__':
    
    coords = {'Creek Fire': [37.201, -119.272]}
    
    ''' Aerosol/Smoke '''
    data = goes_pull(2020, 9, 5, satellite='G17', product='GLM')
    data = reprocess(data, dataset_name=list(data.variables)[0], args=[37.201, -119.272, 5])
    # Plotting
    mapper(data, param=list(data.variables)[0])
    
    ''' Aerosol/Smoke '''
    # data = goes_pull(2020, 9, 5, satellite='G17', product='ABI-L2-ADPC')
    # data = reprocess(data, dataset_name='Smoke', args=[37.201, -119.272, 5])
    # # Plotting
    # mapper(data, param='Smoke')
        
    print('Script completed.')