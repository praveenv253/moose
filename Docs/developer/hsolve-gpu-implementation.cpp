/**
 * \page HSolveGpuImplementation HSolve GPU Implementation
 *
 * \section Intro Introduction
 *
 * This page describes how HSolve has been currently implemented on the GPU
 * and how one might proceed in completing this implementation task.
 *
 * \section currStatus Current status of work
 *
 * Currently, the HSolve passive solver for inverting the Hines matrix has
 * been parallelized, while parallelization of HSolve active (for channels)
 * has been completed, but has not been tested. This work is going on.
 * Implementation of buffering neuron spikes within the GPU is still not
 * complete.
 *
 * The code has been divided into two parts: the major part is linked to the
 * ordinary moose structure and compiles using g++. The second part is the
 * code within cudaLibrary, which contains CUDA code and can only be compiled
 * using nvcc. We do not want to compile all of moose using nvcc because it is
 * not as mature a compiler as g++. The cuda code is therefore compiled into
 * a shared library, the functions of which are then called from moose.
 *
 * The layout of the code in the cudaLibrary folder is roughly as follows:
 *
 * - GpuInterface forms the interface between the cuda code and the non-cuda
 *   C++ code that calls cuda functions from moose.
 *   GpuInterface.h is the only file that is included in the non-cuda segment.
 *   It is therefore essential that this header remain completely free of all
 *   cuda-related keywords. It should not require inclusion of a CUDA header.
 *
 * - GpuInterface has several data structures that are key to understanding
 *   the way information is passed between the CPU and the GPU.
 *   - GpuDataStruct is a structure that contains all the information that
 *     needs to be passed into the GPU.
 *   - GpuInterface is a class that helps construct GpuDataStruct and
 *     manage all calls that need to tbe made to the GPU. It bridges the
 *     cuda and non-cuda code segments.
 *   - GpuLookupTable and the associated structures are essentially
 *     cuda-compatible copies of LookupTable and its associates (from
 *     RateLookup.h). These are also passed into the GPU via
 *     GpuLookupTable::vTable and GpuLookupTable::caTable.
 *
 * - GpuKernels.cpp describes the kernel functions. Note how these are called
 *   from GpuInterface's member functions. This two-level abstraction is
 *   required because kernel launch code can also only be compiled by nvcc.
 *   Most of the code here is a replica of the code that executes in the CPU,
 *   defined in the CPU implementation of HSolvePassive and HSolveActive.
 *
 * - testGpuKernels.cu is currently located in a separate file that is compiled
 *   by nvcc. However, it is called along with the other test cases from
 *   testHSolve.cpp.
 *
 * Additionally, test cases for GpuInterface, for testing setup, have also
 * been written. These are defined in hsolveCuda/testGpuInterface.cpp. Note
 * that this is compiled by g++, because we want to test whether or not the
 * "interface", as seen by the CPU side, works correctly. (The internals have
 * been separately tested in testGpuKernels.cu).
 *
 * \section FutureWork Future Work
 *
 * This section will highlight the possible ways in which the parallelization
 * process can be continued, and chalk out what work is still pending with
 * some hints regarding how one might actually implement the remaining modules.
 *
 */
