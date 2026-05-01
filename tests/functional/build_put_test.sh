#!/bin/bash
set -euo pipefail

WORK_DIR=/fsx/anshumgo/aws-ofi-nccl/NCCLOFI-1483-20260428-1726
BUILD_DIR=$WORK_DIR/build-baseline
EFADP_DIR=/fsx/anshumgo/efa-dp-direct/efa-dp-direct-20260428-1748/CUDA
CUDA_HOME=/usr/local/cuda-12.8
LIBFABRIC_HOME=/opt/amazon/efa
MPI_HOME=/opt/amazon/openmpi

NVCC="$CUDA_HOME/bin/nvcc"
COMMON_FLAGS="-std=c++17 -rdc=true -arch=sm_90 -Xcompiler -fPIC"
INCLUDES="-I$WORK_DIR/include -I$WORK_DIR/3rd-party -I$WORK_DIR/3rd-party/nccl/cuda/include -I$BUILD_DIR/include -I$EFADP_DIR/src -I$LIBFABRIC_HOME/include -I$MPI_HOME/include"

# Step 1: Compile our test
$NVCC $COMMON_FLAGS $INCLUDES -c -o /tmp/gin_put_test.o $WORK_DIR/tests/functional/gin_put_test.cu

# Step 2: Device-link all rdc objects together
$NVCC $COMMON_FLAGS -dlink \
    /tmp/gin_put_test.o \
    $EFADP_DIR/build/efa_cuda_dp.cu.o \
    -o /tmp/gin_put_test_dlink.o

# Step 3: Final host link
$NVCC \
    /tmp/gin_put_test.o \
    $EFADP_DIR/build/efa_cuda_dp.cu.o \
    $EFADP_DIR/build/efa_cuda_dp.o \
    /tmp/gin_put_test_dlink.o \
    -L$LIBFABRIC_HOME/lib \
    -L$MPI_HOME/lib \
    -lfabric -lmpi -ldl \
    -o $WORK_DIR/tests/functional/gin_put_test

echo "Built: $WORK_DIR/tests/functional/gin_put_test"
