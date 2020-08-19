from netCDF4 import Dataset, num2date
import numpy as np
from datetime import timedelta
import pandas as pd
from scipy.ndimage import label, convolve


def convert_mass_points_to_corner_points(col_mass, lons, lats):
    """
    Converts mass points grid to corner point grid.

    Args:
        col_mass (3d numpy array): 3 dimensional column mass array on mass points grid of size l x m x n.
        lons (2d numpy array): 2 dimensional regular grid longitude array of size m x n.
        lats (2d numpy array): 2 dimensional regular grid latitude array of size m x n.

    Returns:
        col_mass (2d numpy array): 3 dimensional column mass array on corner points grid of size l x m+1 x n+1.
        lons (2d numpy array): 2 dimensional regular grid longitude array of size m+1 x n+1.
        lats (2d numpy array): 2 dimensional regular grid latitude array of size m+1 x n+1.
    """
    # Get longitude/latitude grid resolutions
    dlon = np.round(lons[0, 1] - lons[0, 0], 2)
    dlat = np.round(lats[1, 0] - lats[0, 0], 2)

    # Extend lonmin/latmin start by half a grid point
    lonmin = np.round(lons[0, 0] - dlon/2., 2)
    latmin = np.round(lats[0, 0] - dlat/2., 2)

    # Extend lonmax/latmin by half a grid point
    lonmax = np.round(lons[-1, -1] + dlon/2., 2)
    latmax = np.round(lats[-1, -1] + dlat/2., 2)

    # Generate new longitude/latitude grids in 1d
    lons_1d = np.arange(lonmin, lonmax + dlon, dlon)
    lats_1d = np.arange(latmin, latmax + dlat, dlat)

    # Meshgrids to get new long/lat grids as corner points
    lons_crnr, lats_crnr = np.meshgrid(lons_1d, lats_1d)

    # Create array for column mass at corner points
    col_mass_crnr = np.zeros((col_mass.shape[0], col_mass.shape[1]+1, col_mass.shape[2]+1))

    # Define 2 x 2 averaging kernal matrix
    k = (1. / 4.) * np.ones((2, 2))

    # For each time-step extend mass point grid by computing the average for 2 x 2 grid-boxes
    # Note: This requires adding a buffer row and column of zeros on the left and top of the domain.
    for i in range(col_mass.shape[0]):
        col_mass_crnr[i, 1:, 1:] = col_mass[i, :, :]
        col_mass_crnr[i, :, :] = convolve(col_mass_crnr[i, :, :], k)
    return col_mass_crnr, lons_crnr, lats_crnr


def read_satellite_data(path, fn, species_flag):
    """
    Reads in satellite retrieval file.

    Args:
        path (string): Path to satellite retrieval netcdf file.
        fn (string): Filename of satellite retrieval file. Must be netcdf.
        species_flag (string): A string indicating which specie to validate. Valid options are 'ash' or 'so2'.

    Returns:
        col_mass (3d numpy array): Column mass loadings in g/m^2.
        lons (2d numpy array): longitude grid.
        lats (2d numpy array): latitude grid.
        datetime_arr (1d numpy array): datetime object array.
    """
    rootgrp = Dataset(path + fn, 'r', format="NETCDF4")
    if species_flag == 'ash':
        col_mass = rootgrp['mass_loading'][:, ::-1, :]
        lons = rootgrp['longitude'][:, :]
        lats = rootgrp['latitude'][::-1, :]
    elif species_flag == 'so2':
        col_mass = rootgrp['mass_loading'][:, ::-1, 1:]
        lons = rootgrp['longitude'][:, 1:]
        lats = rootgrp['latitude'][::-1, 1:]
    else:
        raise Exception('Invalid species! Valid options are ash or so2.')
    # Create datetime array from netcdf file
    datetime_arr = np.array([num2date(dt, units=rootgrp['time'].units, calendar=rootgrp['time'].calendar)
                             for dt in rootgrp['time'][:]])
    rootgrp.close()

    # Convert mass points to corner points so that satellite and model data are on the same grid for validation metrics.
    col_mass_crnr, lons_crnr, lats_crnr = convert_mass_points_to_corner_points(col_mass, lons, lats)

    return col_mass_crnr, lons_crnr, lats_crnr, datetime_arr


def read_model_data(path, fn, species_flag):
    """
    Reads in FALL3D-8.0 output file.

    Args:
        path (string): Path to FALL3D output netcdf file.
        fn (string): Filename of FALL3D output file. Must be netcdf.
    Returns:
        col_mass (3d numpy array): Column mass loadings in g/m^2.
        lons (2d numpy array): longitude grid.
        lats (2d numpy array): latitude grid.
        datetime_arr (datetime object): datetime object containing model time steps.
    """
    rootgrp = Dataset(path + fn, 'r', format="NETCDF4")
    if species_flag == 'ash':
        col_mass = rootgrp['tephra_col_mass_pm'][:, 1, :, :]
    elif species_flag == 'so2':
        col_mass = rootgrp['SO2_col_mass'][:, :, :]
    else:
        raise Exception('Invalid species! Valid options are ash or so2.')
    lons_1d = rootgrp['lon'][:]
    lats_1d = rootgrp['lat'][:]
    datetime_arr = np.array([num2date(dt, units=rootgrp['time'].units,
                                      calendar=rootgrp['time'].calendar).replace(minute=0, second=0)
                             for dt in rootgrp['time'][:]])
    rootgrp.close()
    lons, lats = np.meshgrid(lons_1d, lats_1d)
    return col_mass, lons, lats, datetime_arr


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculates the great circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lon1 (float): Longitude of first point in decimal degrees.
        lat1 (float): Latitude of first point in decimal degrees.
        lon2 (float): Longitude of second point in decimal degrees.
        lat2 (float): Latitude of second point in decimal degrees.

    Returns:
        distance (float): Great circle distance in km.
    """
    # Set constants
    DEG2RAD = np.pi/180.
    EARTH_RADIUS = 6378.  # [km]
    # Apply Haversine formula
    dlon = lon2*DEG2RAD - lon1*DEG2RAD
    dlat = lat2*DEG2RAD - lat1*DEG2RAD
    distance = 2. * np.arcsin(np.sqrt(np.sin(dlat/2)**2 + np.cos(lat1*DEG2RAD) *
                                      np.cos(lat2*DEG2RAD) * np.sin(dlon/2)**2)) * EARTH_RADIUS
    return distance


def filter_objects(objects, object_num, object_threshold):
    """
    Filters set of objects based on a size limit.

    Args:
        objects (2d int array): An array of objects labelled by integers (e.g. 1, 2, 3 ... object_num).
        object_num (int): Total number of objects in object array.
        object_threshold (int): Maximum size of object. Object of size less than object_threshold will be filtered out.

    Returns:
        objects_filtered (2d int array): New object array after filtering.
        object_num_filtered (int): New total number of objects after filtering.
    """
    objects_modified = np.copy(objects)

    # 1. Zero-out all small objects
    for i in range(object_num):
        if len(objects[objects == float(i + 1)]) < object_threshold:
            objects_modified[objects == float(i + 1)] = 0

    # 2. Convert to binary
    objects_modified[objects_modified > 0] = 1

    # 3. Re-label objects
    objects_filtered, object_num_filtered = label(objects_modified)
    return objects_filtered, object_num_filtered


def calc_fms(A, B, area):
    """
    Computes the Figure of Merit in Space (FMS) validation metric.

    Args:
        A (2d int array): A binary array of size m x n.
        B (2d int array): A binary array of size m x n.
        area (2d float array): A float array of size m x n containing the spatial area of each grid-box [m^2].

    Returns:
        fms (float): Figure of Merit in Space (valid range from 0 to 1).
    """
    # find where A = B = 1. (i.e. intersection)
    intersect = np.where((A == 1.) & (A == B))
    intersect_area = np.nansum(area[intersect])
    # find union of A and B
    union_binary = np.zeros_like(A)
    union_binary[A == 1.] = 1.
    union_binary[B == 1.] = 1.
    union = np.where(union_binary == 1.)
    union_area = np.nansum(area[union])
    # compute FMS
    fms = (intersect_area / union_area)
    return fms


def compute_validation_metrics(col_mass_sat, lons_sat, lats_sat, datetime_sat,
                               col_mass_mod, lons_mod, lats_mod, datetime_mod,
                               start_time, end_time, species_flag, output_path):
    """
    Computes and write validation metrics (SAL and FMS) to file

    Args:
        col_mass_sat (2d numpy array): Satellite column mass loadings in g/m^2.
        lons_sat (2d numpy array): Satellite longitude grid.
        lats_sat (2d numpy array): Satellite latitude grid.
        datetime_sat (datetime object): Datetime object containing satellite time steps.
        col_mass_mod (2d numpy array): Model column mass loadings in g/m^2.
        lons_mod (2d numpy array): Model longitude grid.
        lats_mod (2d numpy array): Model latitude grid.
        datetime_mod (datetime object): Datetime object containing model time steps.
        start_time (datetime object): Datetime object specifying the start time of the validation period.
        end_time (datetime object): Datetime object specifying the end time of the validation period.
        species_flag (string): A string indicating which specie to validate. Valid options are 'ash' or 'so2'.
        output_path (string): Output path for validation metric text file.

    Returns:
        None. A file will be written to the output_path.
    """

    dt_minutes = 60.
    time_diff = end_time - start_time
    time_minutes = time_diff.total_seconds()/60.
    num = int(time_minutes/dt_minutes) + 1

    # Setup empty lists for writing out data
    S_list = []
    A_list = []
    L_list = []
    SAL_list = []
    FMS_list = []
    datetime_list = []
    yyyymmdd_list = []
    HHMM_list = []
    hours_list = []
    object_num_obs_list = []
    object_num_mod_list = []
    center_of_mass_lon_obs_list = []
    center_of_mass_lat_obs_list = []
    center_of_mass_lon_mod_list = []
    center_of_mass_lat_mod_list = []

    for i in range(num):
        current_time = start_time + timedelta(minutes=i * dt_minutes)

        # Select time step to compare satellite to model
        qq_sat = np.where(datetime_sat == current_time)[0][0]
        qq_mod = np.where(datetime_mod == current_time)[0][0]
        cm_sat = np.nan_to_num(col_mass_sat[qq_sat, :, :])   # nan_to_num converts NaNs to zeros (for edge of sat view)
        cm_mod = col_mass_mod[qq_mod, :, :]

        # Count total number of objects in observation and model data
        if species_flag == 'ash':
            mass_loading_threshold = 0.2  # mass loading threshold for ash (g/m^2)
        elif species_flag == 'so2':
            mass_loading_threshold = 5.  # mass loading threshold for so2 (DU)
        else:
            raise Exception('Invalid species! Valid options are ash or so2.')
        cm_sat[cm_sat < mass_loading_threshold] = 0.
        cm_mod[cm_mod < mass_loading_threshold] = 0.

        # Convert mass loading arrays to binary arrays
        cm_sat_binary = cm_sat.copy()
        cm_mod_binary = cm_mod.copy()
        cm_sat_binary[cm_sat_binary > 0.] = 1.
        cm_mod_binary[cm_mod_binary > 0.] = 1.

        # Count objects
        labeled_array_obs, num_objects_obs = label(cm_sat_binary)
        labeled_array_mod, num_objects_mod = label(cm_mod_binary)

        # Filter out small objects (i.e. remove objects smaller than size 16)
        labeled_array_obs, num_objects_obs = filter_objects(labeled_array_obs, num_objects_obs, 16)
        labeled_array_mod, num_objects_mod = filter_objects(labeled_array_mod, num_objects_mod, 16)

        # Compute area of each lat/lon grid box (note this assumes spherical Earth)
        Re = 6.378e6  # [m]
        dlon_deg = 0.1  # [deg]
        dlat_deg = 0.1  # [deg]
        dlon_rad = dlon_deg * np.pi / 180.  # [rad]
        dlat_rad = dlat_deg * np.pi / 180.  # [rad]
        area_2d = (Re ** 2.) * np.cos(lats_mod * np.pi / 180.) * dlon_rad * dlat_rad  # [m^2]

        # Calculate Figure of Merit in Space (FMS)
        FMS = calc_fms(cm_sat_binary, cm_mod_binary, area_2d)

        # Compute SAL
        # 1. Convert g/m^2 to area-integrated ash mass [g] in each column
        R_xy_obs = cm_sat.copy() * area_2d
        R_xy_mod = cm_mod.copy() * area_2d

        # 2. Calculate total mass of each object and normalise by maximum total mass for each object.
        # Do for observation fields
        R_n_obs_arr = np.zeros(num_objects_obs)
        R_n_obs_max_arr = np.zeros(num_objects_obs)
        for i in range(num_objects_obs):
            R_n_obs_arr[i] = np.nansum(R_xy_obs[labeled_array_obs == float(i + 1)])
            R_n_obs_max_arr[i] = np.nanmax(R_xy_obs[labeled_array_obs == float(i + 1)])
        V_n_obs_arr = R_n_obs_arr / R_n_obs_max_arr

        # Do for model fields
        R_n_mod_arr = np.zeros(num_objects_mod)
        R_n_mod_max_arr = np.zeros(num_objects_mod)
        for i in range(num_objects_mod):
            R_n_mod_arr[i] = np.nansum(R_xy_mod[labeled_array_mod == float(i + 1)])
            R_n_mod_max_arr[i] = np.nanmax(R_xy_mod[labeled_array_mod == float(i + 1)])
        V_n_mod_arr = R_n_mod_arr / R_n_mod_max_arr

        # 3. Calculate weighted means
        V_obs = np.nansum(R_n_obs_arr * V_n_obs_arr) / np.nansum(R_n_obs_arr)
        V_mod = np.nansum(R_n_mod_arr * V_n_mod_arr) / np.nansum(R_n_mod_arr)

        # Compute S
        S = (V_mod - V_obs) / (0.5 * (V_mod + V_obs))

        # Compute A
        R_avg_obs = np.nansum(R_xy_obs) / np.nansum(area_2d)
        R_avg_mod = np.nansum(R_xy_mod) / np.nansum(area_2d)
        A = (R_avg_mod - R_avg_obs) / (0.5 * (R_avg_mod + R_avg_obs))

        # Compute L1 and L2 (L1+L2=L)
        # Find centre of mass of the satellite observations
        C_obs_x = np.nansum(R_xy_obs * lons_sat) / np.nansum(R_xy_obs)
        C_obs_y = np.nansum(R_xy_obs * lats_sat) / np.nansum(R_xy_obs)

        # Find centre of mass of the model simulations
        C_mod_x = np.nansum(R_xy_mod * lons_mod) / np.nansum(R_xy_mod)
        C_mod_y = np.nansum(R_xy_mod * lats_mod) / np.nansum(R_xy_mod)

        # Find distance betweeen centre of masses
        C_dist = haversine(C_obs_x, C_obs_y, C_mod_x, C_mod_y)

        # Find maximum distance within domain
        D_max = haversine(lons_mod[0, 0], lats_mod[0, 0], lons_mod[-1, -1], lats_mod[-1, -1])

        # Compute L1
        L1 = C_dist / D_max

        # Find center of masses for each object
        C_n_obs_arr = np.zeros_like(R_n_obs_arr)
        C_n_mod_arr = np.zeros_like(R_n_mod_arr)
        C_obs_n_x_arr = np.zeros(num_objects_obs)
        C_obs_n_y_arr = np.zeros(num_objects_obs)
        for i in range(num_objects_obs):
            C_obs_n_x = np.nansum(R_xy_obs[labeled_array_obs == float(i + 1)] *
                                  lons_mod[labeled_array_obs == float(i + 1)]) / \
                        np.nansum(R_xy_obs[labeled_array_obs == float(i + 1)])
            C_obs_n_y = np.nansum(R_xy_obs[labeled_array_obs == float(i + 1)] *
                                  lats_mod[labeled_array_obs == float(i + 1)]) / \
                        np.nansum(R_xy_obs[labeled_array_obs == float(i + 1)])
            C_obs_n_x_arr[i] = C_obs_n_x
            C_obs_n_y_arr[i] = C_obs_n_y
            # Calculate distance between centre of mass and centre of mass of each object
            C_n_obs_arr[i] = haversine(C_obs_x, C_obs_y, C_obs_n_x, C_obs_n_y)

        C_mod_n_x_arr = np.zeros(num_objects_mod)
        C_mod_n_y_arr = np.zeros(num_objects_mod)
        for i in range(num_objects_mod):
            C_mod_n_x = np.nansum(R_xy_mod[labeled_array_mod == float(i + 1)] *
                                  lons_mod[labeled_array_mod == float(i + 1)]) / \
                        np.nansum(R_xy_mod[labeled_array_mod == float(i + 1)])

            C_mod_n_y = np.nansum(R_xy_mod[labeled_array_mod == float(i + 1)] *
                                  lats_mod[labeled_array_mod == float(i + 1)]) / \
                        np.nansum(R_xy_mod[labeled_array_mod == float(i + 1)])
            C_mod_n_x_arr[i] = C_mod_n_x
            C_mod_n_y_arr[i] = C_mod_n_y
            # Calculate distance between centre of mass and centre of mass of each object
            C_n_mod_arr[i] = haversine(C_mod_x, C_mod_y, C_mod_n_x, C_mod_n_y)

        # Compute weighted average distance
        H_obs = np.nansum(R_n_obs_arr * np.abs(C_n_obs_arr)) / np.nansum(R_n_obs_arr)
        H_mod = np.nansum(R_n_mod_arr * np.abs(C_n_mod_arr)) / np.nansum(R_n_mod_arr)

        # Compute L2
        L2 = 2. * (np.abs(H_mod - H_obs) / D_max)

        # Compute L
        L = L1 + L2

        # Compute SAL
        SAL = np.abs(S) + np.abs(A) + L

        # Append data to lists
        S_list.append(S)
        A_list.append(A)
        L_list.append(L)
        SAL_list.append(SAL)
        FMS_list.append(FMS)
        hours_list.append(i)
        datetime_list.append(current_time)
        yyyymmdd_list.append(int(current_time.strftime('%Y%m%d')))
        HHMM_list.append(int(current_time.strftime('%H%M')))
        object_num_obs_list.append(num_objects_obs)
        object_num_mod_list.append(num_objects_mod)
        center_of_mass_lon_obs_list.append(C_obs_x)
        center_of_mass_lat_obs_list.append(C_obs_y)
        center_of_mass_lon_mod_list.append(C_mod_x)
        center_of_mass_lat_mod_list.append(C_mod_y)

    # Write data to file
    print("Writing validation metrics to file...")
    header = 'yyyymmdd, HHMM, S, A, L, SAL, FMS, num_objects_obs, num_objects_mod, ' \
             'clon_obs, clat_obs, ' \
             'clon_mod, clat_mod'
    zippedlist = list(zip(yyyymmdd_list, HHMM_list, S_list, A_list, L_list, SAL_list, FMS_list,
                          object_num_obs_list, object_num_mod_list,
                          center_of_mass_lon_obs_list, center_of_mass_lat_obs_list,
                          center_of_mass_lon_mod_list, center_of_mass_lat_mod_list))
    df = pd.DataFrame(zippedlist, columns=['yyyymmdd', 'HHMM', 'S', 'A', 'L', 'SAL', 'FMS',
                                           'num_objects_obs', 'num_objects_mod',
                                           'clon_obs', 'clat_obs', 'clon_mod', 'clat_mod'])
    np.savetxt(output_path, df.values, fmt=' '.join(['%i %.4i'] + ['%.4f'] * 5 + ['%i %i'] + ['%.4f'] * 4), header=header)
    print("Done.")
    return
