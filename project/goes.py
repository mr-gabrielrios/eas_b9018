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

    hours = [0, 18, 21]
    
    img = []
    img_dates = []
    for hour in hours:
        img_date = datetime(year, month, day, hour)
        g = goes_nearesttime(datetime(year, month, day, hour),
                             satellite=satellite,
                             product=product,
                             return_as='xarray',
                             bands=band)
        
        # Hold temporary xArray Datasets
        g_concat = []
            
        if 'GLM' in product:
            group_vars = [var for var in g.coords if 'group' in var or 'flash' in var]
            g = g.drop_vars(group_vars)
            g = g.drop_dims(['number_of_groups', 'number_of_flashes'])
            
            for i in range(0, len(g.number_of_events)):
                working_data = g.isel(number_of_events=i)
                ds = xr.Dataset(
                    data_vars={'energy': (['x', 'y'],[[working_data['event_energy'].values]])},
                    coords={'lat': (['x', 'y'], [[working_data['event_lat'].values]]),
                            'lon': (['x', 'y'], [[working_data['event_lon'].values]]),
                            't': working_data['time_coverage_start'].values})
                ds = ds.expand_dims('t')
                ds = ds.assign_coords(time_coverage_start=
                                      working_data['time_coverage_start'])
                ds.attrs['title'] = g.attrs['title']
                ds.attrs['long_name'] = 'GLM L2+ Lightning Detection: event radiant energy'
                ds.attrs['units'] = 'J'
                g_concat.append(ds)
                
            g = xr.concat(g_concat, dim='t')
            
        img.append(g)
        img_dates.append(img_date)
    
    return xr.concat(img, dim='t')
    
    
def reprocess(data, dataset_name=None, args=None):
    
    ''' Reprocess coordinate data. '''
    
    import grid_processes as gp
    
    long_name = data.attrs['title']
    
    # GOES-R projection info
    try:
        proj_info = data['goes_imager_projection']
        lon_origin = proj_info.longitude_of_projection_origin
        H = proj_info.perspective_point_height + proj_info.semi_major_axis
        r_eq = proj_info.semi_major_axis
        r_pol = proj_info.semi_minor_axis  
    except:
        # Values taken from the PUG
        lon_origin = -137
        H = 35786023 + 6378137
        r_eq = 6378137.0
        r_pol = 6356752.31414
    lambda_0 = lon_origin * np.pi / 180  # Longitude of origin converted to radians
    
    size_x, size_y = len(data.x), len(data.y)
    
    # GOES-R grid info
    if args:
        [central_lat, central_lon, bound_sz] = args[:]
        print(data)
        [x0, x1, y1, y0] = gp.grid_grab(central_lon, central_lat, bound_sz, H, r_pol, r_eq, lambda_0, size_x, size_y)
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
   
def mapper(data, param, single_time=None, center=None, tol=None, extent=None, sat_img=False):
    
    if not center:
        center = [np.nanmean([data['lat'].min().values, data['lat'].max().values]),
                  np.nanmean([data['lon'].min().values, data['lon'].max().values])]
    
    # Pick timestep
    if single_time:
        
        try:
            fig, ax = basemap(center, sat_img=sat_img)
            im = ax.scatter(data['lon'], data['lat'], c=data[param].values, 
                               transform=ccrs.PlateCarree(), cmap='viridis', s=5)
            colorbar = fig.colorbar(im)
            
            ax.set_title('{0}'.format(data['time_coverage_start'].values[0]), 
                         fontsize=10)
            fig.suptitle('{0}'.format(data.attrs['long_name']),
                         y=1, x=0.54)
            
            plt.gca()
        except:
            pass
    else:
        # Get extrema
        vmin, vmax = data[param].min().values, data[param].max().values
        # Normalize
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for t in range(0, len(data.t)):
            
            data_ = data.isel(t=t)
            
            # Mask missing data
            lon = xr.where(np.isnan(data_['lon']), 0, data_['lon'])
            lat = xr.where(np.isnan(data_['lat']), 0, data_['lat'])
            c = xr.where(np.isnan(data_['lat']) | np.isnan(data_['lon']), 
                         0, data_[param])
            
            if param == 'Rad':
                cmap = 'Greys_r'
            elif param == 'RRQPE':
                cmap = 'Greens'
            else:
                cmap = 'viridis'
            
            if tol:
                fig, ax = basemap(center, sat_img=sat_img, tol=tol)
            else:
                fig, ax = basemap(center, sat_img=sat_img)
            if param == 'energy':
                im = ax.scatter(data_['lon'], data_['lat'], c=data_[param].values, 
                                   transform=ccrs.PlateCarree(), 
                                   vmin=vmin, vmax=vmax, cmap=cmap)
            else:
                im = ax.pcolormesh(lon, lat, c, 
                                   transform=ccrs.PlateCarree(), 
                                   vmin=vmin, vmax=vmax, cmap=cmap)
            colorbar = fig.colorbar(im)
            colorbar.set_label('{0}'.format(data_[param].attrs['units']),
                               rotation=270, labelpad=15)
            
            ax.set_title('{0}'.format(data_['time_coverage_start'].values), 
                         fontsize=10)
            fig.suptitle('{0}'.format(data_[param].attrs['long_name']),
                         y=1, x=0.54)
            
            plt.gca()


def aggregator(dates, coords, product, band=None):
    
    ''' Aggregate data into single Dataset for a collection of given dates. '''
    
    # Initialize list to hold all data
    dataset = []
    # Iterate over all dates
    for date in dates:
        # Grab data
        data = goes_pull(date.year, date.month, date.day, 
                         satellite='G17', product=product, band=band)
        # Apply geospatial coordinates
        data = reprocess(data, dataset_name='Rad', args=[coords[0], coords[1], 5])
        dataset.append(data)
    # Concatenate everything
    dataset = xr.concat(dataset, dim='t')
    return dataset

def anom(data, param, center):
    
    for hour, hourly_data in data.groupby('t.hour'):
        # Find mean and standard across all days at the chosen hour
        mean, std = hourly_data[param].mean(dim='t'), hourly_data[param].std(dim='t')
        # Iterate over each daily instance of this chosen hour
        for i in range(0, len(hourly_data.t)):
            # Find standardized anomaly
            std_anom = (hourly_data[param].isel(t=i) - mean)/std
            # Normalize
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2)
        
            fig, ax = basemap(center)
            im = ax.pcolormesh(hourly_data['lon'].isel(t=0), 
                               hourly_data['lat'].isel(t=0), 
                               std_anom.values.T, 
                               transform=ccrs.PlateCarree(), 
                               norm=norm, cmap='RdBu_r')
            
            ax.set_title(hourly_data['time_coverage_start'].isel(t=i).values)
            fig.colorbar(im)
            plt.gca()
    
    
if __name__ == '__main__':
    
    rerun = False
    
    if not rerun:
        coords = {'Creek Fire': [37.201, -119.272],
                  'Mendocino Fire': [39.243, -123.103]}
        dates = {'Creek Fire': [datetime.datetime(2019, 9, 4),
                                datetime.datetime(2019, 9, 5),
                                datetime.datetime(2019, 9, 6),
                                datetime.datetime(2019, 9, 7),
                                datetime.datetime(2019, 9, 8),
                                datetime.datetime(2019, 9, 9),
                                datetime.datetime(2019, 9, 10),
                                datetime.datetime(2020, 9, 4),
                                datetime.datetime(2020, 9, 5),
                                datetime.datetime(2020, 9, 6),
                                datetime.datetime(2020, 9, 7),
                                datetime.datetime(2020, 9, 8),
                                datetime.datetime(2020, 9, 9),
                                datetime.datetime(2020, 9, 10),
                                datetime.datetime(2021, 9, 4),
                                datetime.datetime(2021, 9, 5),
                                datetime.datetime(2021, 9, 6),
                                datetime.datetime(2021, 9, 7),
                                datetime.datetime(2021, 9, 8),
                                datetime.datetime(2021, 9, 9),
                                datetime.datetime(2021, 9, 10)],
                 'Mendocino Fire': [datetime.datetime(2018, 7, 27),
                                datetime.datetime(2018, 7, 28),
                                datetime.datetime(2018, 7, 29),
                                datetime.datetime(2018, 7, 30),
                                datetime.datetime(2018, 7, 31),
                                datetime.datetime(2018, 8, 1),
                                datetime.datetime(2018, 8, 2),
                                datetime.datetime(2019, 7, 27),
                                datetime.datetime(2019, 7, 28),
                                datetime.datetime(2019, 7, 29),
                                datetime.datetime(2019, 7, 30),
                                datetime.datetime(2019, 7, 31),
                                datetime.datetime(2020, 8, 1),
                                datetime.datetime(2020, 8, 2),
                                datetime.datetime(2020, 7, 27),
                                datetime.datetime(2020, 7, 28),
                                datetime.datetime(2020, 7, 29),
                                datetime.datetime(2020, 7, 30),
                                datetime.datetime(2020, 7, 31),
                                datetime.datetime(2020, 8, 1),
                                datetime.datetime(2020, 8, 2)]}
        
        case_study = 'Creek Fire'
        
        # data = aggregator(dates[case_study], coords[case_study], 'ABI-L1b-RadC', band=11)
        
        ''' Lightning '''
        # data = goes_pull(2020, 9, 7, satellite='G17', product='GLM')
        # # Plotting
        # mapper(data, single_time=True, center=coords['Creek Fire'], param='energy')
        
        ''' Band 2 Radiances '''
        data = goes_pull(2020, 9, 6, satellite='G17', product='ABI-L1b-RadC', band=2)
        data = reprocess(data, dataset_name='Smoke', args=[37.201, -119.272, 5])
        # Plotting
        mapper(data, param='Rad')
        
        ''' Aerosol/Smoke '''
        # data = goes_pull(2020, 9, 5, satellite='G17', product='ABI-L2-ADPC')
        # data = reprocess(data, dataset_name='Smoke', args=[37.201, -119.272, 5])
        # # Plotting
        # mapper(data, param='Smoke')
        
        ''' Precipitation '''
        # data = goes_pull(2020, 9, 7, satellite='G17', product='ABI-L2-RRQPEF')
        # data = reprocess(data, dataset_name=list(data.variables)[0], args=[37.201, -119.272, 5])
        # # Plotting
        # mapper(data, param=list(data.variables)[0], center=coords[case_study], tol=0.5)
        
        ''' Reflectance, Band 2 '''
        # dataset = []
        # for date in dates[case_study]:
        #     data = goes_pull(date.year, date.month, date.day, 
        #                      satellite='G17', product='ABI-L1b-RadC', band=2)
        #     data = reprocess(data, dataset_name='Rad', args=[coords[case_study][0], 
        #                                                      coords[case_study][1], 
        #                                                      3])
        #     dataset.append(data)
        #     # # Plotting
        #     # mapper(data, param='Rad')
        # dataset = xr.concat(dataset, dim='t')
        
    print('Script completed.')  