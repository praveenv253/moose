/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "DataHandlerWrapper.h"

DataHandlerWrapper::DataHandlerWrapper( const DataHandler* parentHandler,
	const DataHandler* origHandler )
	: DataHandler( origHandler ),
	parent_( parentHandler )
{;}

DataHandlerWrapper::~DataHandlerWrapper()
{;} // This is the key function. It deletes itself but does not touch parent

////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////

char* DataHandlerWrapper::data( DataId index ) const
{
	return parent_->data( index );
}

/**
 * Returns the number of data entries on local node.
 */
unsigned int DataHandlerWrapper::localEntries() const {
	return parent_->localEntries();
}

bool DataHandlerWrapper::isDataHere( DataId index ) const {
	return parent_->isDataHere( index );
}

bool DataHandlerWrapper::isAllocated() const {
	return parent_->isAllocated();
}

unsigned int DataHandlerWrapper::linearIndex( DataId di ) const
{
	return parent_->linearIndex( di );
}

vector< vector< unsigned int > > 
	DataHandlerWrapper::pathIndices( DataId di ) const
{
	return parent_->pathIndices( di );
}

/// Dummy for now.
DataId DataHandlerWrapper::pathDataId( 
	const vector< vector< unsigned int > >& indices) const
{
	if ( indices.size() != static_cast< unsigned int >( pathDepth_ ) + 1 )
	  return DataId::bad();
	return DataId( 0 );
}
////////////////////////////////////////////////////////////////
// load balancing functions
////////////////////////////////////////////////////////////////

bool DataHandlerWrapper::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

bool DataHandlerWrapper::execThread( ThreadId thread, DataId di ) const
{
	return parent_->execThread( thread, di );
}
////////////////////////////////////////////////////////////////
// Process function
////////////////////////////////////////////////////////////////

void DataHandlerWrapper::process( const ProcInfo* p, Element* e, FuncId fid ) const
{
	parent_->process( p, e, fid );
}

void DataHandlerWrapper:: forall( const OpFunc* f, Element* e, 
	const Qinfo* q, const double* arg, 
	unsigned int argSize, unsigned int numArgs ) const
{
	parent_->forall( f, e, q, arg, argSize, numArgs );
}

unsigned int DataHandlerWrapper::getAllData( vector< char* >& data ) const
{
	return parent_->getAllData( data );
}

////////////////////////////////////////////////////////////////
// Data Reallocation functions
////////////////////////////////////////////////////////////////

void DataHandlerWrapper::globalize( const char* data, unsigned int size)
{
	cout << "Error: DataHandlerWrapper::globalize: parent is const\n";
}

void DataHandlerWrapper::unGlobalize()
{
	cout << "Error: DataHandlerWrapper::unGlobalize: parent is const\n";
}

DataHandler* DataHandlerWrapper::copy( unsigned short newParentDepth, 
	unsigned short copyRootDepth,
	bool toGlobal, unsigned int n ) const
{
	return parent_->copy( newParentDepth, copyRootDepth, toGlobal, n );
}

DataHandler* DataHandlerWrapper::copyUsingNewDinfo(
	const DinfoBase* dinfo ) const
{
	return parent_->copyUsingNewDinfo( dinfo );
}

bool DataHandlerWrapper::resize( 
	unsigned int dimension, unsigned int numEntries    )
{
	cout << "Error: DataHandlerWrapper::resize: parent is const\n";
	return 0;
}

void DataHandlerWrapper::assign( const char* orig, unsigned int numOrig )
{
	cout << "Error: DataHandlerWrapper::assign: parent is const\n";
}

/*
////////////////////////////////////////////////////////////
// Iterators
////////////////////////////////////////////////////////////

DataHandler::iterator DataHandlerWrapper::begin( ThreadId threadNum ) const
{
	return parent_->begin( threadNum );
}

DataHandler::iterator DataHandlerWrapper::end( ThreadId threadNum ) const
{
	return parent_->end( threadNum );
}


void DataHandlerWrapper::rolloverIncrement( iterator* i ) const
{
	parent_->rolloverIncrement( i );
}
*/
