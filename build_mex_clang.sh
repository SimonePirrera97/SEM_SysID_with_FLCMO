#!/bin/zsh
set -euo pipefail

# Direct Apple-silicon build using Command Line Tools; full Xcode is not needed.
SCRIPT_DIR=${0:A:h}
MATLAB_ROOT=${MATLAB_ROOT:-/Applications/MATLAB_R2026a.app}
MATLAB_INCLUDE="$MATLAB_ROOT/extern/include"
MATLAB_LIB="$MATLAB_ROOT/bin/maca64"
SDK_PATH=$(/usr/bin/xcrun --sdk macosx --show-sdk-path)
CC=/Library/Developer/CommandLineTools/usr/bin/clang
COMMON=(-bundle -undefined dynamic_lookup -arch arm64 -mmacosx-version-min=13.3
  -std=c99 -O3 -DNDEBUG -DMATLAB_MEX_FILE -DMX_COMPAT_64
  -isysroot "$SDK_PATH" -I"$MATLAB_INCLUDE" -L"$MATLAB_LIB")

"$CC" "${COMMON[@]}" "$SCRIPT_DIR/nnoe/nnoe_constraints_mex.c" \
  -lmex -lmx -lmat -o "$SCRIPT_DIR/nnoe/nnoe_constraints_mex.mexmaca64"
"$CC" "${COMMON[@]}" "$SCRIPT_DIR/Example4_GrayBoxMagneticLevitation/levitator_constraints_mex.c" \
  -lmex -lmx -lmat -o "$SCRIPT_DIR/Example4_GrayBoxMagneticLevitation/levitator_constraints_mex.mexmaca64"
"$CC" "${COMMON[@]}" -I"$SCRIPT_DIR/native" -DSIDQR_NCB=4 -DSIDQR_USE_MWBLAS \
  "$SCRIPT_DIR/native/sidqr_mex.c" "$SCRIPT_DIR/native/sidqr.c" \
  -lmwblas -lmex -lmx -lmat \
  -o "$SCRIPT_DIR/native/sidqr_mex.mexmaca64"

echo "Built all MEX files in $SCRIPT_DIR"
