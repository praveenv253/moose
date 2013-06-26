/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "GpuInterface.h"
#include "GpuKernels.h"

/*
 * Check CUDA return value and handle appropriately
 */
#define _(value) {															\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/* 
 * Constructor for the GpuInterface class.
 * Allocates memory for all data elements in the GPU
 * Transfers data from CPU to GPU.
 */
GpuInterface::GpuInterface(HSolve *hsolve)
{
	// Find the required sizes of elements
	data_.nCompts = hsolve->V_.size();
	data_.HJSize = hsolve->HJ_.size();
	data_.operandSize = hsolve->operand_.size();
	data_.backOperandSize = hsolve->backOperand_.size();
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

	// Allocate memory for array-of-struct data members that do not contain
	// pointers
	_( cudaMalloc( (void **) &data_.compartment,
				   data_.nCompts * sizeof(Compartment) ) );
	_( cudaMalloc( (void **) &data_.junction,
				   data_.junctionSize * sizeof(Junction) ) );
	
	// Copy data for array-of-struct data members that do not contain pointers
	_( cudaMalloc( data_.compartment, &hsolve->compartment_[0],
				   data_.nCompts * sizeof(Compartment),
				   cudaMemcpyHostToDevice ) );
	_( cudaMalloc( data_.junction, &hsolve->junction_[0],
				   data_.junctionSize * sizeof(Junction),
				   cudaMemcpyHostToDevice ) );
	
	// Allocate data for array-of-struct data members that contain pointers

	// First, we need to create the structs out of the vector of vectors.
	OperandStruct *os = new OperandStruct[data_.operandSize];
	for( int i = 0 ; i < data_.operandSize ; i++ )
	{
		// Find the number of operands in the ith vector of hsolve->operand_
		os[i].nOps = hsolve->operand_[i].size();
		// Allocate memory for the ith vector in hsolve->operand_
		_( cudaMalloc( (void **) &os[i].ops, os[i].nOps * sizeof(double) ) );
		// Copy data for the ith vector in hsolve->operand
		_( cudaMemcpy( &os[i].ops, &hsolve->operand_[i][0],
					   os[i].nOps * sizeof(double), cudaMemcpyHostToDevice ) );
	}
	// Finally, copy the entire set of pointers to these operand arrays into
	// the GPU
	_( cudaMalloc( (void **) &data_.operand,
				   data_.operandSize * sizeof(OperandStruct) ) );
	_( cudaMemcpy( data_.operand, os,
				   data_.operandSize * sizeof(OperandStruct),
				   cudaMemcpyHostToDevice ) );

	// Now, to do the same for hsolve->backOperand_
	OperandStruct *bos = new OperandStruct[data_.backOperandSize];
	for( int i = 0 ; i < data_.backOperandSize ; i++ )
	{
		// Find the number of operands in the ith vector of hsolve->operand_
		bos[i].nOps = hsolve->backOperand_[i].size();
		// Allocate memory for the ith vector in hsolve->operand_
		_( cudaMalloc( (void **) &bos[i].ops, bos[i].nOps * sizeof(double) ) );
		// Copy data for the ith vector in hsolve->operand
		_( cudaMemcpy( &bos[i].ops, &hsolve->backOperand_[i][0],
					   bos[i].nOps * sizeof(double), cudaMemcpyHostToDevice) );
	}
	// Finally, copy the entire set of pointers to these operand arrays into
	// the GPU
	_( cudaMalloc( (void **) &data_.backOperand,
				   data_.backOperandSize * sizeof(OperandStruct) ) );
	_( cudaMemcpy( data_.backOperand, bos,
				   data_.backOperandSize * sizeof(OperandStruct),
				   cudaMemcpyHostToDevice ) );

	// Need to decide how many blocks and threads to use per HSolve object
	// For now, keep each hsolver on its own thread.
	numBlocks_ = 1;
	numThreads_ = 1;
}

void GpuInterface::gpuUpdateMatrix()
{
	dim3 numBlocks(numBlocks_);
	dim3 numThreads(numThreads_);

	gpuUpdateMatrix<<< numBlocks, numThreads >>>( data_ );

	stage_ = 0;    // Update done.
}

void GpuInterface::gpuForwardEliminate()
{
	dim3 numBlocks(numBlocks_);
	dim3 numThreads(numThreads_);

	gpuForwardEliminate<<< numBlocks, numThreads >>>( data_ );

	stage_ = 1;    // Forward elimination done.
}

void GpuInterface::gpuBackwardSubstitute()
{
	dim3 numBlocks(numBlocks_);
	dim3 numThreads(numThreads_);

	gpuBackwardSubstitute<<< numBlocks, numThreads >>>( data_ );
	
	stage_ = 2;    // Backward substitution done.
}
