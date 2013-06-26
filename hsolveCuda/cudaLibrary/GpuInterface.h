/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _GPU_INTERFACE_H
#define _GPU_INTERFACE_H

#include "../HSolve.h"
#include "../HSolveStruct.h"				// For structure definitions
#include "../HinesMatrix.h"					// For JunctionStruct

struct OperandStruct {
	double *op;
	unsigned int nOps;
};

/* Structure to store data that to be transferred to the GPU */
struct GpuDataStruct {
	double *HS;							// Tridiagonal part of Hines matrix
	double *HJ;							// Off-diagonal elements
	double *V;							// Vm values of compartments
	double *VMid;						// Vm values at mid-time-step
	double *HJCopy;						//
	CompartmentStruct *compartment;		// Array of compartments
	OperandStruct *operand;				// Array of operands
	OperandStruct *backOperand			//
	JunctionStruct *junction;			// Array of junctions

	// Thse do not need to be passed by reference because they are not going
	// to be changed by any kernel.
	unsigned int nCompts;
	unsigned int HJSize;
	unsigned int operandSize;
	unsigned int backOperandSize;
	unsigned int junctionSize;
};

/* 
 * GpuInterface class that allows each Hines Solver to create its own interface
 * with the GPU, so that each Hines solver object only uses as many blocks and
 * threads as it needs.
 */
class GpuInterface {
	protected:
		unsigned int numBlocks_;
		unsigned int numThreads_;
		int stage_;
		GpuDataStruct data_;

	public:
		GpuInterface(HSolve *);
		void gpuUpdateMatrix();
		void gpuForwardEliminate();
		void gpuBackwardSubstitute();
};

#endif
