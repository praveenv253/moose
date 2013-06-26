/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _GPU_KERNELS_H
#define _GPU_KERNELS_H

#include "GpuInterface.h"

__global__ void updateMatrixKernel(GpuDataStruct);
__global__ void forwardEliminateKernel(GpuDataStruct);
__global__ void backwardSubstituteKernel(GpuDataStruct);

#endif
