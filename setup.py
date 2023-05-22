from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include

setup(
    ext_modules=cythonize(["standardized_mutual_info.pyx"], language_level="3"),
    include_dirs=[get_include()],
)
