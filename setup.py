from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("_fhals_update.pyx")
)
setup(
    ext_modules = cythonize("_rfhals_update.pyx")
)

