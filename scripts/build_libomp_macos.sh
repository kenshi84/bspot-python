#!/usr/bin/env bash
set -euo pipefail

# Build a local libomp with a deployment target compatible with cibuildwheel.
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "build_libomp_macos.sh is only supported on macOS." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MACOSX_DEPLOYMENT_TARGET:=11.0}"
: "${LLVM_RELEASE:=18.1.8}"
: "${LIBOMP_PREFIX:=${ROOT_DIR}/.cibw/libomp}"

LIBOMP_LIB="${LIBOMP_PREFIX}/lib/libomp.dylib"
if [[ -f "${LIBOMP_LIB}" ]]; then
  if command -v otool >/dev/null 2>&1; then
    install_name="$(otool -D "${LIBOMP_LIB}" | sed -n '2p' | xargs)"
    if [[ "${install_name}" == "${LIBOMP_LIB}" ]]; then
      exit 0
    fi
  else
    exit 0
  fi
  rm -rf "${LIBOMP_PREFIX}"
fi

WORK_DIR="${ROOT_DIR}/.cibw/build-libomp"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

TARBALL="llvm-project-${LLVM_RELEASE}.src.tar.xz"
SRC_DIR="llvm-project-${LLVM_RELEASE}.src"
URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_RELEASE}/${TARBALL}"

if [[ ! -f "${TARBALL}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "${TARBALL}" "${URL}"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "${URL}" "${TARBALL}" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
dest = sys.argv[2]
urllib.request.urlretrieve(url, dest)
PY
  else
    echo "Need curl or python3 to download LLVM sources." >&2
    exit 1
  fi
fi

rm -rf "${SRC_DIR}"
tar -xf "${TARBALL}"

cmake -S "${SRC_DIR}/openmp" -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
  -DCMAKE_INSTALL_PREFIX="${LIBOMP_PREFIX}" \
  -DCMAKE_INSTALL_NAME_DIR="${LIBOMP_PREFIX}/lib"

cmake --build build --target install -j

if command -v install_name_tool >/dev/null 2>&1; then
  install_name_tool -id "${LIBOMP_LIB}" "${LIBOMP_LIB}"
fi
