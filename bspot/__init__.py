from pybspot import *

from importlib.metadata import PackageNotFoundError, version
try:
    __version__ = version("bspot")
except PackageNotFoundError:
    __version__ = "0+unknown"
