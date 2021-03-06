/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _TWO_DIM_HANDLER_H
#define _TWO_DIM_HANDLER_H

/**
 * This class manages the data part of Elements. It handles a two-
 * dimensional array.
 */
class TwoDimHandler: public BlockHandler
{
	public:

		TwoDimHandler( const DinfoBase* dinfo, 
			const vector< DimInfo >& dimensions,
			unsigned short pathDepth, bool isGlobal );

		TwoDimHandler( const TwoDimHandler* other );

		~TwoDimHandler();

		////////////////////////////////////////////////////////////
		// Information functions
		////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////
		// load balancing functions: defined in BlockHandler
		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////
		// Process and forall functions: defined in BlockHandler
		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////
		// Data Reallocation functions
		////////////////////////////////////////////////////////////////
		// globalize(): defined in BlockHandler
		// unGlobalize(): defined in BlockHandler

		/**
		 * Make a copy to specified depth in the element tree, with
		 * specified number n of copies, doing appropriate node 
		 * partitioning if toGlobal is false.
		 */
		DataHandler* copy( unsigned short newParentDepth,
			unsigned short copyRootDepth,
			bool toGlobal, unsigned int n ) const;

		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo) const;

		DataHandler* addNewDimension( unsigned int size ) const;

		bool resize( unsigned int dimension, unsigned int size );

		/// assign(): defined in BlockHandler

	private:
		unsigned int nx_;
		unsigned int ny_;
};

#endif	// _TWO_DIM_HANDLER_H


