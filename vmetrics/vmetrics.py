from netCDF4 import Dataset, num2date
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy.ndimage import label


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    deg2rad = np.pi/180.
    # haversine formula
    dlon = lon2*deg2rad - lon1*deg2rad
    dlat = lat2*deg2rad - lat1*deg2rad
    a = np.sin(dlat/2)**2 + np.cos(lat1*deg2rad) * np.cos(lat2*deg2rad) * np.sin(dlon/2)**2
    c = 2. * np.arcsin(np.sqrt(a))
    r = 6378.  # Radius of earth in kilometers. Use 3956 for miles
    # returns distance in km
    return c * r


def filter_objects(objects, object_num, object_threshold):
    objects_modified = np.copy(objects)
    j = 0
    # 1. Zero-out all small objects
    for i in range(object_num):
        if len(objects[objects == float(i + 1)]) < object_threshold:
            objects_modified[objects == float(i + 1)] = 0
            j += 1
    # 2. Convert to binary
    objects_modified[objects_modified > 0] = 1
    # 3. Relabel objects
    objects_filtered, object_num_filtered = label(objects_modified)
    return objects_filtered, object_num_filtered


def calc_fms(A, B, area):
    """
    Compute Figure of Merit in Space (FMS)
    :param A: n x m binary array
    :param B: n x m binary array
    :param area: n x m array containing the spatial area of each grid-box [m^2]
    :return: fms: figure of merit in space
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
    fms = (intersect_area / union_area) * 100.
    return fms

# def compute_fms(path_to_satellite_data, path_to_model_data):
#     # Calculate Figure of Merit in Space (FMS) for filtered arrays
#     obs_binary = np.copy(labeled_array_obs)
#     mod_binary = np.copy(labeled_array_mod)
#     obs_binary[obs_binary > 0.] = 1.
#     mod_binary[mod_binary > 0.] = 1.
#     FMS_filtered = calc_fms(obs_binary, mod_binary, area_2d)
#     print("FMS = " + str(round(FMS_filtered, 3)) + " %")
#     # # write FMS data to file
#     # header_fms = 'yyyymmdd, HHMM, FMS, FMSf'
#     # zippedlist_fms = list(zip(yyyymmdd_list, HHMM_list, FMS_list, FMSf_list))
#     # df_fms = pd.DataFrame(zippedlist_fms, columns=['yyyymmdd', 'HHMM', 'FMS', 'FMSf'])
#     # np.savetxt(output_path + 'puyehue_fms.txt', df_fms.values, fmt=' '.join(['%i %.4i'] + ['%.4f'] * 2),
#     #            header=header_fms)
#     return


def compute_sal(path_to_satellite_data, path_to_model_data, output_path='./sal.txt', write_objects=False):
    # path_to_satellite_data = '/home/aprata/radtrans/retrievals/case_studies/2011_puyehue/fall3d_validation/puyehue_2011_ash_retrievals_nn.nc'
    # path_to_model_data = '/home/aprata/radtrans/retrievals/case_studies/2011_puyehue/fall3d_validation/puyehue-2011.data-insertion.nc'
    # read in model data
    rootgrp_mod = Dataset(path_to_model_data, 'r', format="NETCDF4")
    lons_1d = rootgrp_mod['lon'][:]
    lats_1d = rootgrp_mod['lat'][:]
    datetime_mod = np.array([num2date(dt, units=rootgrp_mod['time'].units,
                                      calendar=rootgrp_mod['time'].calendar).replace(minute=0, second=0)
                             for dt in rootgrp_mod['time'][:]])
    col_mass_mod = rootgrp_mod['tephra_col_mass_pm'][:, 1, :, :]
    rootgrp_mod.close()

    # set up grid definition based on model lat/lons
    lons_grid, lats_grid = np.meshgrid(lons_1d, lats_1d)

    # read in satellite data
    rootgrp_sat = Dataset(path_to_satellite_data, 'r', format="NETCDF4")
    col_mass_sat = rootgrp_sat['mass_loading'][:, ::-1, :]
    datetime_sat = np.array([num2date(dt, units='seconds since 2011-06-05 00:00 UTC', calendar='proleptic_gregorian')
                             for dt in rootgrp_sat['time'][:]])
    lons_sat = rootgrp_sat['longitude'][:, :]
    lats_sat = rootgrp_sat['latitude'][::-1, :]
    rootgrp_sat.close()

    start_time = datetime(2011, 6, 5, 15)
    end_time = datetime(2011, 6, 9)

    dt_minutes = 60.
    time_diff = end_time - start_time
    time_minutes = time_diff.total_seconds()/60.
    num = int(time_minutes/dt_minutes) + 1

    SAL_list = []
    S_list = []
    A_list = []
    L_list = []
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
        #print(current_time.strftime('%Y-%m-%d %H%M UTC'))
        # Select time step to compare satellite to model
        qq_sat = np.where(datetime_sat == current_time)[0][0]
        qq_mod = np.where(datetime_mod == current_time)[0][0]
        cm_sat = np.nan_to_num(col_mass_sat[qq_sat, :, :])   # nan_to_num converts NaNs to zeros (for edge of sat view)
        cm_mod = col_mass_mod[qq_mod, :, :]

        # Count total number of objects in observation and model data
        mass_loading_threshold = 0.2  # mass loading threshold (g/m^2)

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
        #print('Number of objects (obs) = ' + str(num_objects_obs))
        #print('Number of objects (mod) = ' + str(num_objects_mod))

        # Filter out small objects (i.e. remove objects smaller than size 16)
        labeled_array_obs, num_objects_obs = filter_objects(labeled_array_obs, num_objects_obs, 16)
        labeled_array_mod, num_objects_mod = filter_objects(labeled_array_mod, num_objects_mod, 16)

        # Compute area of each lat/lon grid box (note this assumes spherical Earth)
        Re = 6.378e6  # [m]
        dlon_grid = 0.1  # [deg]
        dlat_grid = 0.1  # [deg]
        dlon = dlon_grid * np.pi / 180.  # [rad]
        dlat = dlat_grid * np.pi / 180.  # [rad]
        area_2d = (Re ** 2.) * np.cos(lats_grid * np.pi / 180.) * dlon * dlat  # [m^2]

        # Compute SAL
        # 1. Convert g/m^2 to area-integrated ash mass [g] in each column
        R_xy_obs = cm_sat.copy() * area_2d
        R_xy_mod = cm_mod.copy() * area_2d

        # Check that total mass of ash is reasonable:
        #print('Total mass ash for obs [Tg] = ', str(round(np.nansum(R_xy_obs)/1e12, 3)))
        #print('Total mass ash for mod [Tg] = ', str(round(np.nansum(R_xy_mod)/1e12, 3)))

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
        C_obs_x = np.nansum(R_xy_obs * lons_grid) / np.nansum(R_xy_obs)
        C_obs_y = np.nansum(R_xy_obs * lats_grid) / np.nansum(R_xy_obs)

        # Find centre of mass of the model simulations
        C_mod_x = np.nansum(R_xy_mod * lons_grid) / np.nansum(R_xy_mod)
        C_mod_y = np.nansum(R_xy_mod * lats_grid) / np.nansum(R_xy_mod)

        #print('Center of mass lon, lat (obs) = ' + str(C_obs_x) + ', ' + str(C_obs_y))
        #print('Center of mass lon, lat (mod) = ' + str(C_mod_x) + ', ' + str(C_mod_y))

        # Find distance betweeen centre of masses
        C_dist = haversine(C_obs_x, C_obs_y, C_mod_x, C_mod_y)

        # Find maximum distance within domain
        D_max = haversine(lons_grid[0, 0], lats_grid[0, 0], lons_grid[-1, -1], lats_grid[-1, -1])
        # Compute L1
        L1 = C_dist / D_max

        # Find center of masses for each object
        C_n_obs_arr = np.zeros_like(R_n_obs_arr)
        C_n_mod_arr = np.zeros_like(R_n_mod_arr)
        C_obs_n_x_arr = np.zeros(num_objects_obs)
        C_obs_n_y_arr = np.zeros(num_objects_obs)
        for i in range(num_objects_obs):
            C_obs_n_x = np.nansum(R_xy_obs[labeled_array_obs == float(i + 1)] *
                                  lons_grid[labeled_array_obs == float(i + 1)]) / \
                        np.nansum(R_xy_obs[labeled_array_obs == float(i + 1)])
            C_obs_n_y = np.nansum(R_xy_obs[labeled_array_obs == float(i + 1)] *
                                  lats_grid[labeled_array_obs == float(i + 1)]) / \
                        np.nansum(R_xy_obs[labeled_array_obs == float(i + 1)])
            C_obs_n_x_arr[i] = C_obs_n_x
            C_obs_n_y_arr[i] = C_obs_n_y
            # Calculate distance between centre of mass and centre of mass of each object
            C_n_obs_arr[i] = haversine(C_obs_x, C_obs_y, C_obs_n_x, C_obs_n_y)
            #print(i, C_obs_n_x, C_obs_n_y)

        C_mod_n_x_arr = np.zeros(num_objects_mod)
        C_mod_n_y_arr = np.zeros(num_objects_mod)
        for i in range(num_objects_mod):
            C_mod_n_x = np.nansum(R_xy_mod[labeled_array_mod == float(i + 1)] *
                                  lons_grid[labeled_array_mod == float(i + 1)]) / \
                        np.nansum(R_xy_mod[labeled_array_mod == float(i + 1)])

            C_mod_n_y = np.nansum(R_xy_mod[labeled_array_mod == float(i + 1)] *
                                  lats_grid[labeled_array_mod == float(i + 1)]) / \
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
        #print('S = ' + str(S))
        #print('A = ' + str(A))
        #print('L = ' + str(L))
        #print('SAL = ' + str(SAL))

        # write observation objects to file
        if write_objects:
            # write header for observation fields
            header_obs = 'Author=Andrew Prata (andrew.prata@bsc.es)\n' + \
                         'datetime=' + current_time.strftime('%Y%m%d%H%M') + '\n' + \
                         'S=' + str(round(S, 3)) + '\n' + \
                         'A=' + str(round(A, 3)) + '\n' + \
                         'L=' + str(round(L, 3)) + '\n' + \
                         'SAL=' + str(round(SAL, 3)) + '\n' + \
                         'N_OBS=' + str(round(num_objects_obs, 3)) + '\n' + \
                         'clon_OBS=' + str(round(C_obs_x, 3)) + '\n' + \
                         'clat_OBS=' + str(round(C_obs_y, 3)) + '\n' + \
                         'clon, clat'

            # write header for model fields
            header_mod = 'Author=Andrew Prata (andrew.prata@bsc.es)\n' + \
                         'datetime=' + current_time.strftime('%Y%m%d%H%M') + '\n' + \
                         'S=' + str(round(S, 3)) + '\n' + \
                         'A=' + str(round(A, 3)) + '\n' + \
                         'L=' + str(round(L, 3)) + '\n' + \
                         'SAL=' + str(round(SAL, 3)) + '\n' + \
                         'N_MOD=' + str(round(num_objects_mod, 3)) + '\n' + \
                         'clon_MOD=' + str(round(C_mod_x, 3)) + '\n' + \
                         'clat_MOD=' + str(round(C_mod_y, 3)) + '\n' + \
                         'clon, clat'
            ziplist_obs = list(zip(list(C_obs_n_x_arr), list(C_obs_n_y_arr)))
            ziplist_mod = list(zip(list(C_mod_n_x_arr), list(C_mod_n_y_arr)))
            df_obs = pd.DataFrame(ziplist_obs)
            df_mod = pd.DataFrame(ziplist_mod)
            np.savetxt(output_path + current_time.strftime('%Y%m%d%H%M') + '_obs.txt', df_obs.values, fmt=' '.join(['%.4f'] * 2), header=header_obs)
            np.savetxt(output_path + current_time.strftime('%Y%m%d%H%M') + '_mod.txt', df_mod.values, fmt=' '.join(['%.4f'] * 2), header=header_mod)
        S_list.append(S)
        A_list.append(A)
        L_list.append(L)
        SAL_list.append(SAL)
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
    # write SAL data to file
    header = 'yyyymmdd, HHMM, S, A, L, SAL, num_objects_obs, num_objects_mod, ' \
             'clon_obs, clat_obs, ' \
             'clon_mod, clat_mod'
    zippedlist = list(zip(yyyymmdd_list, HHMM_list, S_list, A_list, L_list, SAL_list,
                          object_num_obs_list, object_num_mod_list,
                          center_of_mass_lon_obs_list, center_of_mass_lat_obs_list,
                          center_of_mass_lon_mod_list, center_of_mass_lat_mod_list))
    df = pd.DataFrame(zippedlist, columns=['yyyymmdd', 'HHMM', 'S', 'A', 'L', 'SAL', 'num_objects_obs', 'num_objects_mod',
                                           'clon_obs', 'clat_obs', 'clon_mod', 'clat_mod'])
    np.savetxt(output_path, df.values, fmt=' '.join(['%i %.4i'] + ['%.4f'] * 4 + ['%i %i'] + ['%.4f'] * 4), header=header)
    return SAL_list

