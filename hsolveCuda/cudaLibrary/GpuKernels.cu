/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <cstdio>
#include "GpuKernels.h"
#include "../HSolveStruct.h"	// For CompartmentStruct, etc.
#include "../HinesMatrix.h"		// For JunctionStruct
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void updateMatrixKernel(GpuDataStruct ds) {
	/*
	 * Copy contents of HJCopy_ into HJ_. Cannot do a vector assign() because
	 * iterators to HJ_ get invalidated in MS VC++
	 */
	if ( ds.HJSize != 0 )
		memcpy( ds.HJ, ds.HJCopy, sizeof( double ) * ds.HJSize );

	double *ihs = ds.HS;
	double *iv  = ds.V;
	
	CompartmentStruct *ic;
	for ( 	ic = ds.compartment;
			ic < ds.compartment + ds.nCompts * sizeof( CompartmentStruct );
			++ic )
	{
		*ihs         = *( 2 + ihs );
		*( 3 + ihs ) = *iv * ic->CmByDt + ic->EmByRm;
		
		ihs += 4, ++iv;
	}
	
	/* Not going to consider inject at the first implementation level
	map< unsigned int, InjectStruct >::iterator inject;
	for ( inject = inject_.begin(); inject != inject_.end(); inject++ ) {
		unsigned int ic = inject->first;
		InjectStruct& value = inject->second;
		
		HS_[ 4 * ic + 3 ] += value.injectVarying + value.injectBasal;
		
		value.injectVarying = 0.0;
	}
	*/
}

__global__ void forwardEliminateKernel(GpuDataStruct ds) {
	unsigned int ic = 0;
	double *ihs = ds.HS;
	double **iop = ds.operand;
	JunctionStruct *junction;
	
	if ( iop ) {
		for( int x = 0 ; x < 36 ; x++ ) {		//XXX debugging only
			printf( "%p ", *(iop + x) );
		}
		printf("\n");
	}
	
	double pivot;
	double division;
	unsigned int index;
	unsigned int rank;
	double *j, *s;
	for ( junction = ds.junction;
	      junction < ds.junction + ds.junctionSize;
	      junction++ )
	{
		index = junction->index;
		rank = junction->rank;
		
		while ( ic < index ) {
			*( ihs + 4 ) -= *( ihs + 1 ) / *ihs * *( ihs + 1 );
			*( ihs + 7 ) -= *( ihs + 1 ) / *ihs * *( ihs + 3 );
			
			++ic, ihs += 4;
		}
		
		pivot = *ihs;
		if ( rank == 1 ) {
			printf("rank=1; ");
			printf("ic: %d ", ic);
			printf("ihs: %p ", ihs);
			printf("iop: %p ", iop);
			j = *iop;
			s = *(iop + 1);
			
			printf( "s: %p\n", s );
			
			division    = *( j + 1 ) / pivot;
			*( s )     -= division * *j;
			*( s + 3 ) -= division * *( ihs + 3 );
			
			iop += 3;
		} else if ( rank == 2 ) {
			printf("rank=2; ");
			printf("ic: %d ", ic);
			printf("ihs: %p ", ihs);
			printf("iop: %p ", iop);
			j = *iop;
			
			s           = *( iop + 1 );
			printf( "s: %p ", s );
			division    = *( j + 1 ) / pivot;
			*( s )     -= division * *j;
			*( j + 4 ) -= division * *( j + 2 );
			*( s + 3 ) -= division * *( ihs + 3 );
			
			s           = *( iop + 3 );
			printf( "s: %p\n", s );
			division    = *( j + 3 ) / pivot;
			*( j + 5 ) -= division * *j;
			*( s )     -= division * *( j + 2 );
			*( s + 3 ) -= division * *( ihs + 3 );
			
			iop += 5;
		} else {
			printf("rank=%d; ", rank);
			printf("ic: %d ", ic);
			printf("ihs: %p ", ihs);
			printf("iop: %p\n", iop);
			double **end = iop + 3 * rank * ( rank + 1 );
			for ( ; iop < end; iop += 3 )
				**iop -= **( iop + 2 ) / pivot * **( iop + 1 );
		}
		
		++ic, ihs += 4;
	}
	
	while ( ic < ds.nCompts - 1 ) {
		*( ihs + 4 ) -= *( ihs + 1 ) / *ihs * *( ihs + 1 );
		*( ihs + 7 ) -= *( ihs + 1 ) / *ihs * *( ihs + 3 );
		
		++ic, ihs += 4;
	}
}

__global__ void backwardSubstituteKernel(GpuDataStruct ds) {
	// We are reverse iterating here, so all pointers are initialized to the
	// ultimate elements of their respective arrays.
	int ic = ds.nCompts - 1;
	double *ivmid = ds.VMid + ic;
	double *iv = ds.V + ic;
	double *ihs = ds.HS + 4 * ds.nCompts - 1;
	double **iop = ds.operand + ds.operandSize - 1;
	double **ibop = ds.backOperand + ds.backOperandSize - 1;
	JunctionStruct *junction = ds.junction + ds.junctionSize - 1;
	
	*ivmid = *ihs / *( ihs - 3 );
	*iv = 2 * *ivmid - *iv;
	--ic, --ivmid, --iv, ihs -= 4;
	
	int index;
	int rank;
	for ( ;
	      ds.junction != NULL && junction >= ds.junction;
	      junction-- )
	{
		index = junction->index;
		rank = junction->rank;
		
		while ( ic > index ) {
			// ivmid was -1, so now it's +1!
			*ivmid = ( *ihs - *( ihs - 2 ) * *( ivmid + 1 ) ) / *( ihs - 3 );
			*iv = 2 * *ivmid - *iv;
			
			--ic, --ivmid, --iv, ihs -= 4;
		}
		
		if ( rank == 1 ) {
			*ivmid = ( *ihs - **iop * **( iop - 2 ) ) / *( ihs - 3 );
			
			iop -= 3;
		} else if ( rank == 2 ) {
			double *v0 = *( iop );
			double *v1 = *( iop - 2 );
			double *j  = *( iop - 4 );
			
			*ivmid = ( *ihs
			           - *v0 * *( j + 2 )	// j was a vdIterator in forward!
			           - *v1 * *j			// so + remains +!!
			         ) / *( ihs - 3 );
			
			iop -= 5;
		} else {
			*ivmid = *ihs;
			for ( int i = 0; i < rank; ++i ) {
				*ivmid -= **ibop * **( ibop - 1 );
				ibop -= 2;
			}
			*ivmid /= *( ihs - 3 );
			
			iop -= 3 * rank * ( rank + 1 );
		}
		
		*iv = 2 * *ivmid - *iv;
		--ic, --ivmid, --iv, ihs -= 4;
	}
	
	while ( ic >= 0 ) {
		// The ivmid was -1, so now it becomes +1!
		*ivmid = ( *ihs - *( ihs - 2 ) * *( ivmid + 1 ) ) / *( ihs - 3 );
		*iv = 2 * *ivmid - *iv;
		
		--ic, --ivmid, --iv, ihs -= 4;
	}
}

