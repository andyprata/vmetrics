# Validation Metrics
This package was written for validating the FALL3D-8.0 benchmark suite with satellite retrievals and accompanies the GMD paper accessible [here](https://gmd.copernicus.org/preprints/gmd-2020-166/).

The code was written to test two specific case studies:
1. The 2011 eruption of Puyehue-Cordon Caulle (Chile). Used for validating ash simulations.
1. The 2019 eruption of Raikoke (Russia). Used for validating SO2 simulation.

# Installation
To install, clone or download this repository to your local disk.
```
git clone https://github.com/andyprata/vmetrics.git
``` 

Then at the top of the file tree run
```
pip install -e .
```

Pip should automatically find and install all the dependancies. The installation has been tested for Python 3.6 using a conda environment.

# Running the validation script
Once installed you can validate FALL3D ash simulations with
```
python validate.py ./data/puyehue-2011.sat.nc <model_output.nc> ash
```
And validate the FALL3D so2 simulations with
```
python validate.py ./data/raikoke-2019.sat.nc <model_output.nc> so2
```

where <model_output.nc> is the full path (including filename) to the relevant FALL3D netcdf output. After running this script a validation metrics textfile will be written to `./output`.

To verify that the results presented in the GMD paper are the same as those generated by you we suggest using `diff`. For example:
```
diff ./output/validation_metrics_ash.txt ./output/validation_metrics_ash_data_insertion.txt
diff ./output/validation_metrics_so2.txt ./output/validation_metrics_so2_data_insertion.txt
```

where the `_data_insertion.txt` files are the data insertion results presented in the GMD paper. These files are included in this repo for testing and can be found under `output` folder. Validation results for the no data insertion runs (i.e. `_no_data_insertion.txt`) are also included. Expected result is no difference. If you have any issues with the installation or running of this package please contact Andrew Prata (andrew.prata@bsc.es) or Arnau Folch (arnau.folch@bsc.es).

