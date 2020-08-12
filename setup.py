import setuptools

setuptools.setup(
        name = "vmetrics",
        packages = ["vmetrics"],
        version = "1.0",
        author = "Andrew Prata",
        install_requires=["netCDF4", "numpy", "pandas", "scipy"]
)

