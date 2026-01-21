# [bspot-python](https://github.com/kenshi84/bspot-python)
Python binding to [BSP-OT](https://github.com/baptiste-genest/BSP-OT) (Genest et al., 2025)

# Installation
```bash
pip install bspot
```

# Build wheel locally for Python 3.12 on macOS
- Download official Python 3.12 installer for macOS from [python.org](https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg).
```
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pip install cibuildwheel
CIBW_BUILD="cp312-*" /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m cibuildwheel --output-dir wheels
```
