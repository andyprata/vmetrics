from vmetrics import vmetrics as vm
import sys

path_to_obs = sys.argv[1]
path_to_mod = sys.argv[2]
species_flag = sys.argv[3]

if __name__ == "__main__":
    output_path = './output/validation_metrics_'+species_flag+'.txt'
    print("Computing validation metrics (i.e. SAL and FMS)...")
    vm.compute_validation_metrics(path_to_obs, path_to_mod, species_flag, output_path=output_path)
    print("Output saved here: " + output_path)
