### Objective
# Translate coordinate inputs to grid indices to minimize data loaded, typically by data_processing.py

import numpy as np

### Latitude and longitude to ABI grid elements
# Note: CONUS has a longitudinal extent of 0.140 rad and a latitudinal extent of 0.084 rad
#       CONUS has a grid space of 1500 x 2500, with a resolution of 56u rad
#       Reference PUG-L2+-Vol5, Table 4.2.3-3, for constant values
def coord_to_grid(lon_deg, lat_deg, H, r_pol, r_eq, lambda_0, size_x, size_y):
    # Reference PUG-L2+-Vol5, Table 4.2.8, for constant values
    # https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf
    ecc = np.sqrt((r_eq**2-r_pol**2)/r_eq**2) # 1st Earth eccentricity
    
    if size_x == 2500:
        goes_conus_res = 56e-6 # Resolution for 2-km-resolution CONUS images
    elif size_x == 10000:
        goes_conus_res = 14e-6 # Resolution for 0.5-km-resolution CONUS images
    elif size_x == 5424:
        goes_conus_res = 56e-6 # Full-disk resolution
    else:
        goes_conus_res = 280e-6 # Resolution for 10-km-resolution CONUS images
    
    # For full disk imagery
    if size_x == 5424:
        goes_lon = [-0.151872, 0.151872] # E/W extent of GOES-Wast CONUS images (rad)
        goes_lat = [-0.151872, 0.151872] # N/S extent of GOES-Wast CONUS images (rad)
    # For GOES-17
    else:
        goes_lon = [-0.07, 0.07] # E/W extent of GOES-Wast CONUS images (rad)
        goes_lat = [0.044268005, 0.128212] # N/S extent of GOES-Wast CONUS images (rad)
        # For GOES-16
        # goes_lon = [-0.101332, 0.038612] # E/W extent of GOES-East CONUS images (rad)
        # goes_lat = [0.044268005, 0.128212] # N/S extent of GOES-East CONUS images (rad)
    
    lon, lat = [lon_deg*np.pi/180, lat_deg*np.pi/180] # Convert from degrees to radians
    phi_c = np.arctan(np.tan(lat)*(r_pol**2/r_eq**2)) # Geocentric latitude
    r_c = r_pol/np.sqrt(1-(ecc**2)*(np.cos(phi_c)**2)) # Geocentric distance to Earth's surface
    [s_x, s_y, s_z] = [H - r_c*np.cos(phi_c)*np.cos(lon-lambda_0),
                       -r_c*np.cos(phi_c)*np.sin(lon-lambda_0),
                       r_c*np.sin(phi_c)]
    
    # Longitude (x), latitude (y)
    [x, y] = [np.arcsin(-s_y/np.sqrt(s_x**2+s_y**2+s_z**2)), np.arctan(s_z/s_x)]
    
    # Find the index by determining space from the top-left corner (grid origin)
    y_idx = (goes_lat[1] - y)/goes_conus_res
    x_idx = (x - goes_lon[0])/goes_conus_res
    
    return x_idx, y_idx 

### Defines grid elements as a function of longitude, latitude, and bound/box size
def grid_grab(central_lon, central_lat, bound_sz, H, r_pol, r_eq, lambda_0, size_x, size_y):
    bound_box = [central_lon - bound_sz,
                 central_lon + bound_sz,
                 central_lat - bound_sz,
                 central_lat + bound_sz]
    [x0, y0] = coord_to_grid(bound_box[0], bound_box[2], H, r_pol, r_eq, lambda_0, size_x, size_y) # Select SW box corner
    [x1, y1] = coord_to_grid(bound_box[1], bound_box[3], H, r_pol, r_eq, lambda_0, size_x, size_y) # Select NE box corner
    # Use floor and ceiling functions to fully grab the appropriate grid points
    [x0, y0] = [np.floor(x0), np.ceil(y0)]
    [x1, y1] = [np.ceil(x1), np.floor(y1)]
    return int(x0), int(x1), int(y0), int(y1)