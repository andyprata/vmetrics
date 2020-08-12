# Validation Metrics
This package was written for validating the FALL3D-8.0 benchmark suite with satellite retrievals and accompanies the GMD paper accessible here:

The code was written to test two specific case studies:
The 2011 eruption of Puyehue-Cordon Caulle (Chile)
The 2019 eruption of Raikoke (Russia)

The 2011 Cordon-Caulle case can be used to validate fine ash mass loading simulations and the 2019 Raikoke case can be used to validate SO2 simulations from FALL3D.

# Installation
To install clone or download this repository to your local disk.
```clone``` 

Then at the top of the file tree run
```pip install -e .```

Pip should auto-matically find and install all the dependancies.
The installation has been tested for Python Version 3.6 using a conda environment e.g.
```conda create --name py36 python=3.6```
```conda activate py36```
```pip install -e .```

# Running the validation script
Once installed you can validate FALL3D simulations like so
```python validate.py <satellite_retrieval.nc> <model_output.nc> <species_name>```

where `<satellite_retrieval.nc` is the full path (including filename) to a satellite retrieval file (included in the repository) and <model_output.nc> is the path (including filename) to the FALL3D netcdf output. After running this script a validation metric text file will be written to `./output`.

To verify that the results presented in the GMD paper we suggest using `diff` on user generated validation text files and the . For example:
```diff validation_metrics_ash.txt validation_metrics_ash_gmd.txt```
```diff validation_metrics_so2.txt validation_metrics_so2_gmd.txt```

Expected result is no difference.

