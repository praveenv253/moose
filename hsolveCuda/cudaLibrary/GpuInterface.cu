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
#include <cuda.h>
#include <cuda_runtime.h>

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
	} 																		\
}

/**
 * Constructor for the GpuInterface class.
 * Allocates memory for all data elements in the GPU. Transfers data from CPU
 * to GPU.
 */
GpuInterface::GpuInterface(HSolve *hsolve)
{
	// Save the pointer!
	hsolve_ = hsolve;

	// Find the required sizes of elements
	data_.nCompts = hsolve->V_.size();
	data_.HJSize = hsolve->HJ_.size();
	data_.junctionSize = hsolve->junction_.size();

	// Allocate memory for data members
	if ( data_.nCompts > 0 ) {
		_( cudaMalloc((void **) &data_.HS, 4* data_.nCompts* sizeof(double)) );
		_( cudaMalloc((void **) &data_.V, data_.nCompts * sizeof(double)) );
		_( cudaMalloc((void **) &data_.VMid, data_.nCompts * sizeof(double)) );
		_( cudaMalloc((void **) &data_.compartment,
					  data_.nCompts * sizeof(CompartmentStruct)) );
	} else {
		data_.HS = NULL;
		data_.V = NULL;
		data_.VMid = NULL;
		data_.compartment = NULL;
	}
	if ( data_.HJSize > 0 ) {
		_( cudaMalloc((void **) &data_.HJ, data_.HJSize * sizeof(double)) );
		_( cudaMalloc((void **) &data_.HJCopy, data_.HJSize* sizeof(double)) );
	} else {
		data_.HJ = NULL;
		data_.HJCopy = NULL;
	}
	if ( data_.junctionSize > 0 ) {
		_( cudaMalloc( (void **) &data_.junction,
					   data_.junctionSize * sizeof(JunctionStruct) ) );
	} else {
		data_.junction = NULL;
	}

	// Copy array-of-double data into GPU
	if ( data_.HS )
		_( cudaMemcpy( data_.HS, &hsolve->HS_[0],
					   4 * data_.nCompts * sizeof(double),
					   cudaMemcpyHostToDevice ) );
	if ( data_.V )
		_( cudaMemcpy( data_.V, &hsolve->V_[0],
					   data_.nCompts * sizeof(double),
					   cudaMemcpyHostToDevice ) );
	if ( data_.VMid )
		_( cudaMemcpy( data_.VMid, &hsolve->VMid_[0],
					   data_.nCompts * sizeof(double),
					   cudaMemcpyHostToDevice ) );
	if ( data_.HJ )
		_( cudaMemcpy( data_.HJ, &hsolve->HJ_[0],
					   data_.HJSize * sizeof(double),
					   cudaMemcpyHostToDevice ) );
	if ( data_.HJCopy )
		_( cudaMemcpy( data_.HJCopy, &hsolve->HJCopy_[0],
					   data_.HJSize * sizeof(double),
					   cudaMemcpyHostToDevice ) );
	// Copy data for array-of-struct data members
	if ( data_.compartment )
		_( cudaMemcpy( data_.compartment, &hsolve->compartment_[0],
					   data_.nCompts * sizeof(CompartmentStruct),
					   cudaMemcpyHostToDevice ) );
	if ( data_.junction )
		_( cudaMemcpy( data_.junction, &hsolve->junction_[0],
					   data_.junctionSize * sizeof(JunctionStruct),
					   cudaMemcpyHostToDevice ) );

	// Call to take care of populating GpuInterface::operand_ and
	// GpuInterface::backOperand_.
	makeOperands(hsolve);

	// Allocate and copy memory for operands and backOperands
	data_.operandSize = operand_.size();
	data_.backOperandSize = backOperand_.size();

	if ( data_.operandSize > 0 ) {
		_( cudaMalloc( (void**)&data_.operand,
					   operand_.size() * sizeof(double*) ) );
		_( cudaMemcpy( data_.operand, &operand_[ 0 ],
					   operand_.size() * sizeof(double*),
					   cudaMemcpyHostToDevice ) );
	} else {
		data_.operand = NULL;
	}

	if ( data_.backOperandSize > 0 ) {
		_( cudaMalloc((void**)&data_.backOperand,
					  backOperand_.size() * sizeof(double*)) );
		_( cudaMemcpy(data_.backOperand, &backOperand_[ 0 ],
					  backOperand_.size() * sizeof(double*),
					  cudaMemcpyHostToDevice) );
	} else {
		data_.backOperand = NULL;
	}

#ifdef DO_UNIT_TESTS
	if ( data_.nCompts > 0 ) {
		_( cudaMalloc( (void **) &data_.inject,
					   data_.nCompts * sizeof(InjectStruct) ) );
		copyInject();
	} else {
		data_.inject = NULL;
	}
#endif

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
			operand_.push_back( data_.HS + 4 * farIndex );
			operand_.push_back( data_.VMid + farIndex );
		} else if ( rank == 2 ) {
			operand_.push_back( base );
			
			// Select 2nd last member.
			farIndex = group[ group.size() - 2 ];
			operand_.push_back( data_.HS + 4 * farIndex );
			operand_.push_back( data_.VMid + farIndex );
			
			// Select last member.
			farIndex = group[ group.size() - 1 ];
			operand_.push_back( data_.HS + 4 * farIndex );
			operand_.push_back( data_.VMid + farIndex );
		} else {
			// Operations on diagonal elements and elements from B
			// (as in Ax = B).
			int start = group.size() - rank;
			for ( unsigned int j = 0; j < rank; ++j ) {
				farIndex = group[ start + j ];
				
				// Diagonal elements
				operand_.push_back( data_.HS + 4 * farIndex );
				operand_.push_back( base + 2 * j );
				operand_.push_back( base + 2 * j + 1 );
				
				// Elements from B
				operand_.push_back( data_.HS + 4 * farIndex + 3 );
				operand_.push_back( data_.HS + 4 * index + 3 );
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
			backOperand_.push_back( data_.VMid + farIndex );
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

void GpuInterface::synchronize()
{
	cudaDeviceSynchronize();
}

void GpuInterface::unsetup()
{
	// Create temporary storage space before assigning the vectors in HSolve.
	double *HS = new double[ 4 * data_.nCompts ];
	double *HJ = new double[ data_.HJSize ];
	double *V = new double[ data_.nCompts ];
	double **operand = new double*[ data_.operandSize ];

	// Copy data from the GPU back to the CPU and then into the HSolve vectors
	_( cudaMemcpy( HS, data_.HS, 4 * data_.nCompts * sizeof(double),
				   cudaMemcpyDeviceToHost ) );
	hsolve_->HS_.assign( HS, HS + 4 * data_.nCompts );

	_( cudaMemcpy( HJ, data_.HJ, data_.HJSize * sizeof(double),
				   cudaMemcpyDeviceToHost ) );
	hsolve_->HJ_.assign( HJ, HJ + data_.HJSize );

	_( cudaMemcpy( HJ, data_.HJCopy, data_.HJSize * sizeof(double),
				   cudaMemcpyDeviceToHost ) );
	hsolve_->HJCopy_.assign( HJ, HJ + data_.HJSize );

	_( cudaMemcpy( V, data_.V, data_.nCompts * sizeof(double),
				   cudaMemcpyDeviceToHost ) );
	hsolve_->V_.assign( V, V + data_.nCompts );

	_( cudaMemcpy( V, data_.VMid, data_.nCompts * sizeof(double),
				   cudaMemcpyDeviceToHost ) );
	hsolve_->VMid_.assign( V, V + data_.nCompts );
	
	_( cudaMemcpy( operand, data_.operand, data_.operandSize * sizeof(double),
				   cudaMemcpyDeviceToHost ) );
	operand_.assign( operand, operand + data_.operandSize );

}

#ifdef DO_UNIT_TESTS

/**
 * Function to copy inject_ from the CPU to the GPU. This is used only for
 * testing the RC-behaviour of a single compartment. Hence it is being defined
 * only if unit tests are performed
 */
void GpuInterface::copyInject()
{
	map< unsigned int, InjectStruct >::iterator i;
	vector< InjectStruct > inject( data_.nCompts, InjectStruct() );

	for ( i = hsolve_->inject_.begin(); i != hsolve_->inject_.end(); ++i ) {
		unsigned int ic = i->first;
		InjectStruct& value = i->second;
		inject[ ic ] = value;
	}

	// Memory must already be allocated. This should have happened during
	// construction of the object.
	_( cudaMemcpy( data_.inject, &inject[ 0 ],
				   data_.nCompts * sizeof(InjectStruct),
				   cudaMemcpyHostToDevice ) );
}

// getA and getB functions used in unit tests for comparing matrix element
// values.

/**
 * Used by getA and getB to retrieve single data elements from the GPU.
 * Horribly inefficient.
 */
template< class T >
T get(T *address) {
	T value;
	// Copy data from GPU to CPU
	_( cudaMemcpy( &value, address, sizeof( T ), cudaMemcpyDeviceToHost ) );
	return value;
}
#define getd( addr ) get< double >( addr )

/**
 * Get the (row, col)-element of the Hines matrix.
 */
double GpuInterface::getA( unsigned int row, unsigned int col ) const
{
	/*
	 * If forward elimination is done, or backward substitution is done, and
	 * if (row, col) is in the lower triangle, then return 0.
	 */
	if ( ( stage_ == 1 || stage_ == 2 ) && row > col )
		return 0.0;

	if ( row >= data_.nCompts || col >= data_.nCompts )
		return 0.0;

	if ( row == col ) {
		return getd( data_.HS + 4 * row );
	}

	unsigned int smaller = row < col ? row : col;
	unsigned int bigger = row > col ? row : col;

	// If find returns end, it means that `smaller` was not found.
	if ( hsolve_->groupNumber_.find(smaller) == hsolve_->groupNumber_.end() ) {
		if ( bigger - smaller == 1 )
			return getd( data_.HS + 4 * smaller + 1 );
		else
			return 0.0;
	} else {
		// We could use: groupNumber = groupNumber_[ smaller ], but this is a
		// const function
		unsigned int groupNumber = hsolve_->groupNumber_.find(smaller)->second;
		const vector< unsigned int >& group = hsolve_->coupled_[ groupNumber ];
		unsigned int location, size;
		unsigned int smallRank, bigRank;

		if ( find( group.begin(), group.end(), bigger ) != group.end() ) {
			location = 0;
			for ( int i = 0; i < static_cast< int >( groupNumber ); ++i ) {
				size = hsolve_->coupled_[ i ].size();
				location += size * ( size - 1 );
			}

			size = group.size();
			smallRank = group.end()
						- find( group.begin(), group.end(), smaller ) - 1;
			bigRank = group.end()
					  - find( group.begin(), group.end(), bigger ) - 1;
			location += size * ( size - 1 ) - smallRank * ( smallRank + 1 );
			location += 2 * ( smallRank - bigRank - 1 );

			if ( row == smaller )
				return getd( data_.HJ + location );
			else
				return getd( data_.HJ + location + 1 );
		} else {
			return 0.0;
		}
	}
}

double GpuInterface::getB( unsigned int row ) const
{
	return getd( data_.HS + 4 * row + 3 );
}

double GpuInterface::getVMid( unsigned int row ) const
{
	return getd( data_.VMid + row );
}

double GpuInterface::getV( unsigned int row ) const
{
	return getd( data_.V + row );
}

#endif // DO_UNIT_TESTS

