#!/bin/bash
set -euo pipefail
LAB=/opt/anygrasp_lab
TC=/opt/graspnet_lab/toolchain
export PATH="$LAB/envs/anygrasp/bin:$TC/bin:/usr/bin:/bin"
export PYTHONPATH=
export CUDA_HOME="$TC"
export CC="$TC/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$TC/bin/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
export OMP_NUM_THREADS=2
export PIP_CACHE_DIR="$LAB/cache/pip"
SITE="$LAB/envs/anygrasp/lib/python3.11/site-packages"
export CPATH="$SITE/nvidia/cusparse/include:$SITE/nvidia/cublas/include:$SITE/nvidia/cusolver/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/curand/include:$SITE/nvidia/cufft/include:$SITE/nvidia/nvtx/include"
mkdir -p "$LAB/tools/include" "$LAB/tools/lib" "$LAB/outputs/deployment_20261011"
cp /usr/include/x86_64-linux-gnu/cblas.h /usr/include/x86_64-linux-gnu/cblas_mangling.h "$LAB/tools/include/"
ln -sf "$SITE/nvidia/cusparse/lib/libcusparse.so.12" "$LAB/tools/lib/libcusparse.so"
export LIBRARY_PATH="$LAB/tools/lib:/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$LAB/tools/lib:$TC/lib:${LD_LIBRARY_PATH:-}"
python -m pip install ninja wheel 'setuptools<80' Pillow tqdm transforms3d grasp-nms
cd "$LAB/vendor/MinkowskiEngine"
python setup.py install --force_cuda --blas=blas --blas_include_dirs="$LAB/tools/include" --blas_library_dirs=/usr/lib/x86_64-linux-gnu > "$LAB/outputs/deployment_20261011/minkowski_build.log" 2>&1
cd "$LAB/vendor/anygrasp_sdk/pointnet2"
python -m pip install --no-build-isolation --no-deps . > "$LAB/outputs/deployment_20261011/pointnet_build.log" 2>&1
cp -a /opt/graspnet_lab/vendor/graspnetAPI/graspnetAPI "$SITE/graspnetAPI"
cd "$LAB"
python -c 'import torch, MinkowskiEngine, pointnet2, graspnetAPI; print(torch.__version__, torch.cuda.is_available(), MinkowskiEngine.__version__)'
