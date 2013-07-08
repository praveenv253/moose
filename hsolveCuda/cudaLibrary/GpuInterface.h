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

#include "../../basecode/header.h"			// For miscellaneous defns
#include "../HSolveStruct.h"				// For structure definitions
#include "../RateLookup.h"
#include "../HinesMatrix.h"					// For JunctionStruct
#include "../HSolvePassive.h"
#include "../HSolveActive.h"
#include "../HSolve.h"						// For HSolve

/** Structure to store data that is to be transferred to the GPU */
struct GpuDataStruct {
	double *HS;							///< Tridiagonal part of Hines matrix
	double *HJ;							///< Off-diagonal elements
	double *V;							///< Vm values of compartments
	double *VMid;						///< Vm values at mid-time-step
	double *HJCopy;						///<
	double **operand;					///< Array of pointers to operands
	double **backOperand;				///<
	CompartmentStruct *compartment;		///< Array of compartments
	JunctionStruct *junction;			///< Array of junctions

	// Thse do not need to be passed by reference because they are not going
	// to be changed by any kernel.
	unsigned int nCompts;
	unsigned int HJSize;
	unsigned int operandSize;
	unsigned int backOperandSize;
	unsigned int junctionSize;
};

/**
 * GpuInterface class that allows each Hines Solver to create its own interface
 * with the GPU.
 */
class GpuInterface {
	protected:
		unsigned int numBlocks_;
		unsigned int numThreads_;
		int stage_;
		vector< double * > operand_;
		vector< double * > backOperand_;
		GpuDataStruct data_;
		HSolve *hsolve_;

	public:
		GpuInterface( HSolve * );
		void makeOperands( HSolve * );
		void gpuUpdateMatrix();
		void gpuForwardEliminate();
		void gpuBackwardSubstitute();

		// Functions for unit tests
		double getA( unsigned int, unsigned int ) const;
		double getB( unsigned int ) const;
		double getV( unsigned int ) const;
		double getVMid( unsigned int ) const;
};

#endif
