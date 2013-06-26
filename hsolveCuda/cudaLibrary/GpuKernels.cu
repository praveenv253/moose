/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "GpuKernels.h"
#include "../HsolveStruct.h"	// For CompartmentStruct, etc.
#include "../HinesMatrix.h"		// For JunctionStruct

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
	for ( 	ic = compartment;
			ic < compartment + nCompts * sizeof(CompartmentStruct);
			++ic ) {
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

void HSolvePassive::forwardEliminate(GpuDataStruct ds) {
	unsigned int ic = 0;
	double *ihs = ds.HS;
	OperandStruct *iop = ds.operand;
	JunctionStruct *junction;
	
	double pivot;
	double division;
	unsigned int index;
	unsigned int rank;
	for ( junction = ds.junction;
	      junction != ds.junction + ds.junctionSize;
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
			double *j = iop->ops;
			double *s = iop->ops + 1;
			
			division    = *( j + 1 ) / pivot;
			*( s )     -= division * *j;
			*( s + 3 ) -= division * *( ihs + 3 );
			
			iop += 3;
		} else if ( rank == 2 ) {
			vdIterator j = *iop;
			vdIterator s;
			
			s           = *( iop + 1 );
			division    = *( j + 1 ) / pivot;
			*( s )     -= division * *j;
			*( j + 4 ) -= division * *( j + 2 );
			*( s + 3 ) -= division * *( ihs + 3 );
			
			s           = *( iop + 3 );
			division    = *( j + 3 ) / pivot;
			*( j + 5 ) -= division * *j;
			*( s )     -= division * *( j + 2 );
			*( s + 3 ) -= division * *( ihs + 3 );
			
			iop += 5;
		} else {
			vector< vdIterator >::iterator
				end = iop + 3 * rank * ( rank + 1 );
			for ( ; iop < end; iop += 3 )
				**iop -= **( iop + 2 ) / pivot * **( iop + 1 );
		}
		
		++ic, ihs += 4;
	}
	
	while ( ic < nCompt_ - 1 ) {
		*( ihs + 4 ) -= *( ihs + 1 ) / *ihs * *( ihs + 1 );
		*( ihs + 7 ) -= *( ihs + 1 ) / *ihs * *( ihs + 3 );
		
		++ic, ihs += 4;
	}
}

void HSolvePassive::backwardSubstitute() {
	int ic = nCompt_ - 1;
	vector< double >::reverse_iterator ivmid = VMid_.rbegin();
	vector< double >::reverse_iterator iv = V_.rbegin();
	vector< double >::reverse_iterator ihs = HS_.rbegin();
	vector< vdIterator >::reverse_iterator iop = operand_.rbegin();
	vector< vdIterator >::reverse_iterator ibop = backOperand_.rbegin();
	vector< JunctionStruct >::reverse_iterator junction;
	
	*ivmid = *ihs / *( ihs + 3 );
	*iv = 2 * *ivmid - *iv;
	--ic, ++ivmid, ++iv, ihs += 4;
	
	int index;
	int rank;
	for ( junction = junction_.rbegin();
	      junction != junction_.rend();
	      junction++ )
	{
		index = junction->index;
		rank = junction->rank;
		
		while ( ic > index ) {
			*ivmid = ( *ihs - *( ihs + 2 ) * *( ivmid - 1 ) ) / *( ihs + 3 );
			*iv = 2 * *ivmid - *iv;
			
			--ic, ++ivmid, ++iv, ihs += 4;
		}
		
		if ( rank == 1 ) {
			*ivmid = ( *ihs - **iop * **( iop + 2 ) ) / *( ihs + 3 );
			
			iop += 3;
		} else if ( rank == 2 ) {
			vdIterator v0 = *( iop );
			vdIterator v1 = *( iop + 2 );
			vdIterator j  = *( iop + 4 );
			
			*ivmid = ( *ihs
			           - *v0 * *( j + 2 )
			           - *v1 * *j
			         ) / *( ihs + 3 );
			
			iop += 5;
		} else {
			*ivmid = *ihs;
			for ( int i = 0; i < rank; ++i ) {
				*ivmid -= **ibop * **( ibop + 1 );
				ibop += 2;
			}
			*ivmid /= *( ihs + 3 );
			
			iop += 3 * rank * ( rank + 1 );
		}
		
		*iv = 2 * *ivmid - *iv;
		--ic, ++ivmid, ++iv, ihs += 4;
	}
	
	while ( ic >= 0 ) {
		*ivmid = ( *ihs - *( ihs + 2 ) * *( ivmid - 1 ) ) / *( ihs + 3 );
		*iv = 2 * *ivmid - *iv;
		
		--ic, ++ivmid, ++iv, ihs += 4;
	}
}

