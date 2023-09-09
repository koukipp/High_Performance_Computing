+=======================+
|Computer Specifications|
+=======================+

OS : Windows 10 Enterprise 2004
CPU : CPU INTEL CORE I5-6500 3.20GHZ LGA1151 - BOX
RAM : 8GB single channel 3200MHZ

VM USES
-------

OS : Ubuntu 20.04.1 LTS
Cores : 2/4
RAM : 4GB/8GB
Kernel : 5.4.0-51-generic
Compiler : icc (ICC) 19.1.2.254 20200623

+=========+
|Changelog|
+=========+

1_sobel_orig -> 2_sobel_function_inlining
-----------------------------------------

performace boost -O0 : 0.2s

We inlined the function convolution2D leading to a performance boost. This happens because we avoid the overhead(e.g. creation of stack frames) of calling the function multiple times inside the loop.

We also tried inlining the sobel function but there was not a performace boost as the functiion is only called once.

2_sobel_function_inlining-> 3_sobel_loop_interchange
----------------------------------------------------

performace boost -O0 : 1s

We interchanged the loops from j, i to i, j to improve the locality of memory accesses considering the memory locations are now consecutive in memory.

3_sobel_loop_interchange -> 4_sobel_loop_fusion
-----------------------------------------------

performace boost -O0 : 0.5s

We fused the loops in lines 99 and 106 that occured after inlining convolution2D. There is only a very small performance boost caused by avoiding the incrementation overhead of the loop counters.

We also tried fusing the loops in the lines 86 and 106 leading to no measurable improvement.

4_sobel_loop_fusion -> 5_sobel_strength_reduction
-------------------------------------------------

performace boost -O0 : 1.1s

We replaced pow in line 106 with multiplication leading to a performance boost of 1s in line 122 leading to a boost of 0.1s as multiplication is less taxing in execution time compared to pow.

5_sobel_strength_reduction -> 6_sobel_subexpression_elimination
---------------------------------------------------------------

performace boost -O0 : 0.1s

We stored the value of the array in current index in lines 101, 102 and 122 to avoid consecutive accesses in the array and recalculations of the index.

We tried many other subexpression eliminations having no measurable results probably because they were simple calculations while not including memory accesses in the array.

6_sobel_subexpression_elimination -> 7_sobel_loop_unrolling
---------------------------------------------------------------

performace boost -O0 : 0.3s

We completely unrolled the loops in lines 97 and 98(leading in the most significant boost) using two macros for cleaner code and we partially unrolled the loop in line 120 with a step of 8 using another macro.

We also tried unrolling the loops in lines 84, 85 leading to no boost.

Conclusions
-----------

We notice no obvious performance boost between versions with -fast as the compiler has probably no more optimizations to do with the exception of the loop interchange and the strength reduction.

The -fast versions are way faster than the -O0 probably because the compiler uses vectorization to parallelize for loops as seen in the icc reports.

The standard deviation becomes smaller as the code becomes faster probably because less context switches occur due to the fast execution and the execution times become more consistent.
