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

# Building (mostly copied from [libigl-python-bindings](https://github.com/libigl/libigl-python-bindings?tab=readme-ov-file#testing-cibuildwheel-locally))

## Testing cibuildwheel locally

Install whichever version of Python from the [official website](https://www.python.org/downloads/) and then run:

    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv venv-official-3.12
    source venv-official-3.12/bin/activate
    python -m pip install cibuildwheel
    CIBW_BUILD="cp312-*" python -m cibuildwheel --output-dir wheels --platform macos

## Downloading all the artifacts

A successful [.github/workflows/wheels.yml](.github/workflows/wheels.yml) run will generate a lot of `.whl` files. To download these all at once, you can use the following command:

    mkdir -p wheels
    cd wheels
    gh run download [runid]

## Uploading to TestPyPI / PyPI

Then these can be uploaded to pypi using:

    python -m twine upload --repository testpypi wheels/*/*.whl wheels/*/*.tar.gz
