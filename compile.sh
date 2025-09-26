#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
 -DCMAKE_C_COMPILER=gcc \
 -DCMAKE_CXX_COMPILER=g++
make
compiled_file=$(ls student_agent_module*.so | head -n 1)
mv "$compiled_file" student_agent_module.so
cd ..