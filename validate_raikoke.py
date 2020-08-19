from vmetrics import vmetrics as vm
from datetime import datetime


if __name__ == "__main__":
    # Set input paths to satellite retrieval and FALL3D output netcdf files
    path_sat = '/home/aprata/radtrans/retrievals/case_studies/fall3d_tests/'
    fn_sat = 'raikoke-2019.so2-retrievals.nc'
    path_mod = '/home/aprata/radtrans/retrievals/case_studies/fall3d_tests/'
    fn_mod = 'raikoke-2019.res.nc'
    species_flag = 'so2'

    # Set output path for validation metrics textfile
    output_path = '/home/aprata/projects/vmetrics/expected_output/validation_metrics_raikoke.txt'

    # Set validation time for Raikoke 2019 eruption
    start_time = datetime(2019, 6, 22, 18)
    end_time = datetime(2019, 6, 24, 18)

    print("Reading in satellite and model datasets...")
    # Read in satellite data
    col_mass_sat, lons_sat, lats_sat, datetime_sat = vm.read_satellite_data(path_sat, fn_sat, species_flag)

    # Read in model data
    col_mass_mod, lons_mod, lats_mod, datetime_mod = vm.read_model_data(path_mod, fn_mod, species_flag)

    print("Computing validation metrics (i.e. SAL and FMS)...")
    # Compute validation metrics
    vm.compute_validation_metrics(col_mass_sat, lons_sat, lats_sat, datetime_sat,
                                  col_mass_mod, lons_mod, lats_mod, datetime_mod,
                                  start_time, end_time, species_flag, output_path)

    print("Output saved here: " + output_path)
