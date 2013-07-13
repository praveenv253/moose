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
#include "../testGpuInterface.h"			// For testing this class

/** Structure to store the GPU version of a lookup table */
struct GpuLookupTable {
	double *table;
	double min;
	double max;
	double dx;
	unsigned int nPts;
	unsigned int nColumns;

	GpuLookupTable() {}
	GpuLookupTable( LookupTable );
};

/**
 * Structure to store the GPU version of a lookup row. This is not really
 * any different from the ordinary LookupRow. I just did not want to include
 * another header file and create complications.
 */
struct GpuLookupRow {
	double *row;
	double fraction;
};

/**
 * Structure to store the GPU version of a lookup column. This can effectively
 * be just a single integer, but I am keeping it the same as the CPU version
 * for better extensibility.
 */
struct GpuLookupColumn {
	unsigned int column;
};

/** Structure to store data that is to be transferred to the GPU */
struct GpuDataStruct {
	/** Data structures for HSolvePassive */
	double *HS;							///< Tridiagonal part of Hines matrix
	double *HJ;							///< Off-diagonal elements
	double *V;							///< Vm values of compartments
	double *VMid;						///< Vm values at mid-time-step
	double *HJCopy;						///<
	double **operand;					///< Array of pointers to operands
	double **backOperand;				///<
	CompartmentStruct *compartment;		///< Array of compartments
	JunctionStruct *junction;			///< Array of junctions

	/** Data structures for HSolveActive */
	ChannelStruct		 *channel;
	int					 *channelCount;
	CurrentStruct		 *current;
	CurrentStruct		 **currentBoundary;
	double				 *state;
	//SpikeGenStruct		 *spikegen;		// TODO
	CaConcStruct		 *caConc;
	double				 *ca;
	double				 *caActivation;
	double				 **caTarget;
	unsigned int		 *caCount;
	// gCaDepend and caDependIndex are only used in setup, I think.

	// Lookup table stuff
	GpuLookupTable		 vTable;
	GpuLookupTable		 caTable;
	GpuLookupColumn		 *column;
	GpuLookupRow		 *caRowCompt;
	GpuLookupRow		 **caRow;
	// Id fields will be looked into later.
	// Will also look into outVm and outCa later.

	/** Sizes of elements for HSolvePassive */
	// Thse do not need to be passed by reference because they are not going
	// to be changed by any kernel.
	unsigned int nCompts;
	unsigned int HJSize;
	unsigned int operandSize;
	unsigned int backOperandSize;
	unsigned int junctionSize;

	/** Sizes of elements for HSolveActive */
	unsigned int nChannels;		///< Number of channels and current elements
	unsigned int stateSize;		///< Number of states across all channels
	unsigned int nCaPools;		///< Number of calcium pools in all compts
	unsigned int caRowComptSize;	/** caRowCompt has a size equal to the max
									 * number of calcium pools out of all
									 * compts. */
};

/**
 * GpuInterface class that allows each Hines Solver to create its own interface
 * with the GPU.
 */
class GpuInterface {
	friend void testGpuInterface();
	friend void testSetupWorking();

	protected:
		unsigned int numBlocks_;
		unsigned int numThreads_;
		int stage_;
		vector< double * > operand_;
		vector< double * > backOperand_;
		GpuDataStruct data_;
		HSolve *hsolve_;

	public:
		// Setup
		GpuInterface( HSolve * );
		void makeOperands( HSolve * );

		// HSolvePassive functions
		void gpuUpdateMatrix();
		void gpuForwardEliminate();
		void gpuBackwardSubstitute();
		void synchronize();

		// HSolveActive functions
		void gpuAdvanceChannels( double );
		void gpuCalculateChannelCurrents();
		void gpuAdvanceCalcium();

#ifdef DO_UNIT_TESTS
		// Functions for unit tests
		double getA( unsigned int, unsigned int ) const;
		double getB( unsigned int ) const;
		double getV( unsigned int ) const;
		double getVMid( unsigned int ) const;
#endif

		/**
		 * Single-step-synchronization function - you can test by syncing
		 * after every time step and un-setup-ing, i.e. copying back all
		 * modified data from the GPU. Useful for debugging.
		 */
		void unsetup();
};

#endif
