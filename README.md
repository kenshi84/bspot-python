# bspot-python
Python binding to [BSP-OT](https://github.com/baptiste-genest/BSP-OT) (Genest et al., 2025)

# Build wheels for macOS Python 3.12
- Download official Python 3.12 installer for macOS from [python.org](https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg).
```
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pip install cibuildwheel
CIBW_BUILD="cp312-*" /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m cibuildwheel --output-dir wheels
```
