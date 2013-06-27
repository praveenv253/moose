/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <vector>
#include "GpuInterface.h"
#include "GpuKernels.h"

/*
 * Check CUDA return value and handle appropriately
 */
#define _(value) {															\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		std::cerr << "Error " << cudaGetErrorString(_m_cudaStat)			\
				  << " at line " << __LINE__ << " in file " << __FILE__		\
				  << std::endl;												\
		exit(1);															\
	} }

/** 
 * Constructor for the GpuInterface class.
 * Allocates memory for all data elements in the GPU. Transfers data from CPU
 * to GPU.
 */
GpuInterface::GpuInterface(HSolve *hsolve)
{
	// Find the required sizes of elements
	data_.nCompts = hsolve->V_.size();
	data_.HJSize = hsolve->HJ_.size();
	data_.junctionSize = hsolve->junction_.size();

	// Allocate memory for array-of-double data members
	_( cudaMalloc( (void **) &data_.HS, 4 * data_.nCompts * sizeof(double) ) );
	_( cudaMalloc( (void **) &data_.HJ, data_.HJSize * sizeof(double) ) );
	_( cudaMalloc( (void **) &data_.V, data_.nCompts * sizeof(double) ) );
	_( cudaMalloc( (void **) &data_.VMid, data_.nCompts * sizeof(double) ) );
	_( cudaMalloc( (void **) &data_.HJCopy, data_.HJSize * sizeof(double) ) );

	// Copy array-of-double data into GPU
	_( cudaMemcpy( data_.HS, &hsolve->HS_[0], 4* data_.nCompts* sizeof(double),
				   cudaMemcpyHostToDevice ) );
	_( cudaMemcpy( data_.HJ, &hsolve->HJ_[0], data_.HJSize * sizeof(double),
				   cudaMemcpyHostToDevice ) );
	_( cudaMemcpy( data_.V, &hsolve->V_[0], data_.nCompts * sizeof(double),
				   cudaMemcpyHostToDevice ) );
	_( cudaMemcpy( data_.VMid, &hsolve->VMid_[0], data_.nCompts*sizeof(double),
				   cudaMemcpyHostToDevice ) );
	_( cudaMemcpy( data_.HJCopy, &hsolve->HJCopy_[0],
				   data_.HJSize * sizeof(double), cudaMemcpyHostToDevice ) );

	// Allocate memory for array-of-structure data members
	_( cudaMalloc( (void **) &data_.compartment,
				   data_.nCompts * sizeof(CompartmentStruct) ) );
	_( cudaMalloc( (void **) &data_.junction,
				   data_.junctionSize * sizeof(JunctionStruct) ) );

	// Copy data for array-of-struct data members
	_( cudaMemcpy( data_.compartment, &hsolve->compartment_[0],
				   data_.nCompts * sizeof(CompartmentStruct),
				   cudaMemcpyHostToDevice ) );
	_( cudaMemcpy( data_.junction, &hsolve->junction_[0],
				   data_.junctionSize * sizeof(JunctionStruct),
				   cudaMemcpyHostToDevice ) );

	// Call to take care of populating GpuInterface::operand_
	makeOperands(hsolve);

	// Allocate and copy memory for operands and backOperands
	data_.operandSize = operand_.size();
	data_.backOperandSize = backOperand_.size();

	_( cudaMalloc((void**)&data_.operand, operand_.size() * sizeof(double*)) );
	_( cudaMemcpy(data_.operand, &operand_[ 0 ],
				  operand_.size() * sizeof(double*), cudaMemcpyHostToDevice) );

	_( cudaMalloc((void**)&data_.backOperand,
				  backOperand_.size() * sizeof(double*)) );
	_( cudaMemcpy(data_.backOperand, &backOperand_[ 0 ],
				  backOperand_.size() * sizeof(double*),
				  cudaMemcpyHostToDevice) );

	// Need to decide how many blocks and threads to use per HSolve object
	// For now, keep each hsolver on its own thread.
	numBlocks_ = 1;
	numThreads_ = 1;
}

/**
 * Function to take care of making operands in the same way that
 * HinesMatrix::makeOperands does.
 */
void GpuInterface::makeOperands(HSolve *hsolve)
{
	typedef vector< double >::iterator vdIterator;

	unsigned int index;
	unsigned int rank;
	unsigned int farIndex;
	double *base;
	vector< JunctionStruct >::iterator junction;
	
	// Operands for forward-elimination
	for ( junction = hsolve->junction_.begin();
		  junction != hsolve->junction_.end();
		  ++junction )
	{
		index = junction->index;
		rank = junction->rank;

		// operandBase_[ index ] maps to the vdIterator corresponding to the
		// position of compartment with Hines index `index` in HJ_.
		// base needs to contain the pointer to HJ (in the GPU) which marks
		// the start of this juction in HJ.
		base = data_.HJ
			   + (long)( &( *hsolve->operandBase_[index] ) - &hsolve->HJ_[0] );

		// This is the list of compartments connected at a junction.
		const vector< unsigned int >& group =
			hsolve->coupled_[ hsolve->groupNumber_[ index ] ];
		
		if ( rank == 1 ) {
			operand_.push_back( base );
			
			// Select last member.
			farIndex = group[ group.size() - 1 ];
			operand_.push_back( &data_.HS[ 0 ] + 4 * farIndex );
			operand_.push_back( &data_.VMid[ 0 ] + farIndex );
		} else if ( rank == 2 ) {
			operand_.push_back( base );
			
			// Select 2nd last member.
			farIndex = group[ group.size() - 2 ];
			operand_.push_back( &data_.HS[ 0 ] + 4 * farIndex );
			operand_.push_back( &data_.VMid[ 0 ] + farIndex );
			
			// Select last member.
			farIndex = group[ group.size() - 1 ];
			operand_.push_back( &data_.HS[ 0 ] + 4 * farIndex );
			operand_.push_back( &data_.VMid[ 0 ] + farIndex );
		} else {
			// Operations on diagonal elements and elements from B
			// (as in Ax = B).
			int start = group.size() - rank;
			for ( unsigned int j = 0; j < rank; ++j ) {
				farIndex = group[ start + j ];
				
				// Diagonal elements
				operand_.push_back( &data_.HS [ 0 ] + 4 * farIndex );
				operand_.push_back( base + 2 * j );
				operand_.push_back( base + 2 * j + 1 );
				
				// Elements from B
				operand_.push_back( &data_.HS[ 0 ] + 4 * farIndex + 3 );
				operand_.push_back( &data_.HS[ 0 ] + 4 * index + 3 );
				operand_.push_back( base + 2 * j + 1 );
			}
			
			// Operations on off-diagonal elements.
			double *left;
			double *above;
			double *target;
			
			// Upper triangle elements
			left = base + 1;
			target = base + 2 * rank;
			for ( unsigned int i = 1; i < rank; ++i ) {
				above = base + 2 * i;
				for ( unsigned int j = 0; j < rank - i; ++j ) {
					operand_.push_back( target );
					operand_.push_back( above );
					operand_.push_back( left );
					
					above += 2;
					target += 2;
				}
				left += 2;
			}
			
			// Lower triangle elements
			target = base + 2 * rank + 1;
			above = base;
			for ( unsigned int i = 1; i < rank; ++i ) {
				left = base + 2 * i + 1;
				for ( unsigned int j = 0; j < rank - i; ++j ) {
					operand_.push_back( target );
					operand_.push_back( above );
					operand_.push_back( left );
					
					/*
					 * This check required because the MS VC++ compiler is
					 * paranoid about iterators going out of bounds, even if
					 * they are never used after that.
					 */
					if ( i == rank - 1 && j == rank - i - 1 )
						continue;
					
					target += 2;
					left += 2;
				}
				above += 2;
			}
		}
	}
	
	// Operands for backward substitution
	for ( junction = hsolve->junction_.begin();
		  junction != hsolve->junction_.end();
		  ++junction )
	{
		if ( junction->rank < 3 )
			continue;
		
		index = junction->index;
		rank = junction->rank;
		base = data_.HJ
			   + (long)( &( *hsolve->operandBase_[index] ) - &hsolve->HJ_[0] );
		
		// This is the list of compartments connected at a junction.
		const vector< unsigned int >& group =
			hsolve->coupled_[ hsolve->groupNumber_[ index ] ];
		
		unsigned int start = group.size() - rank;
		for ( unsigned int j = 0; j < rank; ++j ) {
			farIndex = group[ start + j ];
			
			backOperand_.push_back( base + 2 * j );
			backOperand_.push_back( &data_.VMid[ 0 ] + farIndex );
		}
	}
}

void GpuInterface::gpuUpdateMatrix()
{
	dim3 numBlocks(numBlocks_);
	dim3 numThreads(numThreads_);

	updateMatrixKernel<<< numBlocks, numThreads >>>( data_ );

	stage_ = 0;    // Update done.
}

void GpuInterface::gpuForwardEliminate()
{
	dim3 numBlocks(numBlocks_);
	dim3 numThreads(numThreads_);

	forwardEliminateKernel<<< numBlocks, numThreads >>>( data_ );

	stage_ = 1;    // Forward elimination done.
}

void GpuInterface::gpuBackwardSubstitute()
{
	dim3 numBlocks(numBlocks_);
	dim3 numThreads(numThreads_);

	backwardSubstituteKernel<<< numBlocks, numThreads >>>( data_ );
	
	stage_ = 2;    // Backward substitution done.
}


