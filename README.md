# [bspot-python](https://github.com/kenshi84/bspot-python)
Python binding to [BSP-OT](https://github.com/baptiste-genest/BSP-OT) (Genest et al., 2025)

# Installation
```bash
pip install bspot
```

# Example usage
```python
>>> import bspot
>>> import numpy as np
>>> A = np.random.randn(3,10000)
>>> B = np.random.randn(3,10000)
>>> bspot.set_num_threads(8)        # By default, use all available threads
>>> bspot.compute_matching(A, B, orthogonal=True)
array([1586, 7207,  330, ..., 3329, 4056, 3637],
      shape=(10000,), dtype=int32)
```

# Build wheel locally for Python 3.12 on macOS
- Download official Python 3.12 installer for macOS from [python.org](https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg).
```
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pip install cibuildwheel
CIBW_BUILD="cp312-*" /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m cibuildwheel --output-dir wheels
```
