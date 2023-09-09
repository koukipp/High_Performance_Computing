#!/bin/bash
#export OMP_NUM_THREADS=4
#export LD_LIBRARY_PATH='/opt/intel/compilers_and_libraries_2020/linux/lib/intel64'
export LD_LIBRARY_PATH='/home/john/intel/compilers_and_libraries_2020/linux/lib/intel64'
./seq_main "$@"