# Validation Metrics
This package was written for validating the FALL3D-8.0 benchmark suite with satellite retrievals and accompanies the GMD paper accessible [here](https://gmd.copernicus.org/preprints/gmd-2020-166/).

The code was written to test two specific case studies:
1. The 2011 eruption of Puyehue-Cordon Caulle (Chile). Used for validating ash simulations.
1. The 2019 eruption of Raikoke (Russia). Used for validating SO2 simulations.

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
Once installed you can validate FALL3D with the validate.py script, which takes 4 arguments.

To validate the Puyehue-Cordon Caulle 2011 ash simulations do
```
python validate_puyehue.py
```
And to validate the Raikoke 2019 SO2 simulations do
```
python validate_raikoke.py
```

Note the input and output files and paths must be set by the user in the `validate_puyehue.py` and `validate_raikoke.py` files. 

To verify that the results presented in the GMD paper are the same as those generated by the user we suggest using `diff`. For example:
```
diff ./expected_output/validation_metrics_puyehue.txt ./user_output/validation_metrics_puyehue.txt
diff ./expected_output/validation_metrics_raikoke.txt ./user_output/validation_metrics_raikoke
```

These verification files are included in this repository for testing and can be found under `expected_output` folder. Expected result is no difference. If you have any issues with the installation or running of this package please contact Andrew Prata (andrew.prata@bsc.es) or Arnau Folch (arnau.folch@bsc.es).

