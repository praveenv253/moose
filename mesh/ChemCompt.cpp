/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "Boundary.h"
#include "MeshEntry.h"
// #include "Stencil.h"
#include "ChemCompt.h"
#include "../ksolve/StoichHeaders.h"

// Not used anywhere currently. - Sid
static SrcFinfo0 groupSurfaces( 
		"groupSurfaces", 
		"Goes to all surfaces that define this ChemCompt"
);

SrcFinfo5< 
	double,
	vector< double >,
	vector< unsigned int>, 
	vector< vector< unsigned int > >, 
	vector< vector< unsigned int > >
	>* meshSplit()
{
	static SrcFinfo5< 
			double,
			vector< double >,
			vector< unsigned int >, 
			vector< vector< unsigned int > >, 
			vector< vector< unsigned int > >
		>
	meshSplit(
		"meshSplit",
		"Defines how meshEntries communicate between nodes."
		"Args: oldVol, volListOfAllEntries, localEntryList, "
		"outgoingDiffusion[node#][entry#], incomingDiffusion[node#][entry#]"
		"This message is meant to go to the SimManager and Stoich."
	);
	return &meshSplit;
}

static SrcFinfo2< unsigned int, vector< double > >* meshStats()
{
	static SrcFinfo2< unsigned int, vector< double > > meshStats(
		"meshStats",
		"Basic statistics for mesh: Total # of entries, and a vector of"
		"unique volumes of voxels"
	);
	return &meshStats;
}

const Cinfo* ChemCompt::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ChemCompt, double > size(
			"size",
			"Size of entire chemical domain."
			"Assigning this assumes that the geometry is that of the "
			"default mesh, which may not be what you want. If so, use"
			"a more specific mesh assignment function.",
			&ChemCompt::setEntireSize,
			&ChemCompt::getEntireSize
		);

		static ReadOnlyValueFinfo< ChemCompt, unsigned int > numDimensions(
			"numDimensions",
			"Number of spatial dimensions of this compartment. Usually 3 or 2",
			&ChemCompt::getDimensions
		);

		static ValueFinfo< ChemCompt, string > method(
			"method",
			"Advisory field for SimManager to check when assigning "
			"solution methods. Doesn't do anything unless SimManager scans",
			&ChemCompt::setMethod,
			&ChemCompt::getMethod
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		/*
		static DestFinfo stoich( "stoich",
			"Handle Id of stoich. Used to set up connection from mesh to"
			"stoich for diffusion "
			"calculations. Somewhat messy way of doing it, should really "
			"use messaging.",
			new EpFunc1< ChemCompt, Id >( &ChemCompt::stoich )
		);
		*/

		static DestFinfo buildDefaultMesh( "buildDefaultMesh",
			"Tells ChemCompt derived class to build a default mesh with the"
			"specified size and number of meshEntries.",
			new EpFunc2< ChemCompt, double, unsigned int >( 
				&ChemCompt::buildDefaultMesh )
		);

		static DestFinfo handleRequestMeshStats( "handleRequestMeshStats",
			"Handles request from SimManager for mesh stats",
			new EpFunc0< ChemCompt >(
				&ChemCompt::handleRequestMeshStats
			)
		);

		static DestFinfo handleNodeInfo( "handleNodeInfo",
			"Tells ChemCompt how many nodes and threads per node it is "
			"allowed to use. Triggers a return meshSplit message.",
			new EpFunc2< ChemCompt, unsigned int, unsigned int >(
				&ChemCompt::handleNodeInfo )
		);

		static DestFinfo resetStencil( "resetStencil",
			"Resets the diffusion stencil to the core stencil that only "
			"includes the within-mesh diffusion. This is needed prior to "
			"building up the cross-mesh diffusion through junctions.",
			new OpFunc0< ChemCompt >(
				&ChemCompt::resetStencil )
		);


		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////

		static Finfo* nodeMeshingShared[] = {
			meshSplit(), meshStats(), 
			&handleRequestMeshStats, &handleNodeInfo
		};

		static SharedFinfo nodeMeshing( "nodeMeshing",
			"Connects to SimManager to coordinate meshing with parallel"
			"decomposition and with the Stoich",
			nodeMeshingShared, sizeof( nodeMeshingShared ) / sizeof( const Finfo* )
		);

		/*
		static Finfo* geomShared[] = {
			&requestSize, &handleSize
		};

		static SharedFinfo geom( "geom",
			"Connects to Geometry tree(s) defining compt",
			geomShared, sizeof( geomShared ) / sizeof( const Finfo* )
		);
		*/

		//////////////////////////////////////////////////////////////
		// Field Elements
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< ChemCompt, Boundary > boundaryFinfo( 
			"boundary", 
			"Field Element for Boundaries",
			Boundary::initCinfo(),
			&ChemCompt::lookupBoundary,
			&ChemCompt::setNumBoundary,
			&ChemCompt::getNumBoundary,
			4
		);

		static FieldElementFinfo< ChemCompt, MeshEntry > entryFinfo( 
			"mesh", 
			"Field Element for mesh entries",
			MeshEntry::initCinfo(),
			&ChemCompt::lookupEntry,
			&ChemCompt::setNumEntries,
			&ChemCompt::getNumEntries,
			1
		);

	static Finfo* chemMeshFinfos[] = {
		&size,			// Value
		&numDimensions,	// ReadOnlyValue
		&method,		// Value
		&buildDefaultMesh,	// DestFinfo
		&resetStencil,	// DestFinfo
		&nodeMeshing,	// SharedFinfo
		&entryFinfo,	// FieldElementFinfo
		&boundaryFinfo,	// FieldElementFinfo
	};

	static Cinfo chemMeshCinfo (
		"ChemCompt",
		Neutral::initCinfo(),
		chemMeshFinfos,
		sizeof( chemMeshFinfos ) / sizeof ( Finfo* ),
		new Dinfo< short >()
	);

	return &chemMeshCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* chemMeshCinfo = ChemCompt::initCinfo();

ChemCompt::ChemCompt()
	: 
		size_( 1.0 ),
		entry_( this )
{
	;
}

ChemCompt::~ChemCompt()
{ 
		/*
	for ( unsigned int i = 0; i < stencil_.size(); ++i ) {
		if ( stencil_[i] )
			delete stencil_[i];
	}
	*/
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ChemCompt::buildDefaultMesh( const Eref& e, const Qinfo* q,
	double size, unsigned int numEntries )
{
	this->innerBuildDefaultMesh( e, q, size, numEntries );
}

void ChemCompt::handleRequestMeshStats( const Eref& e, const Qinfo* q )
{
	// Pass it down to derived classes along with the SrcFinfo
	innerHandleRequestMeshStats( e, q, meshStats() );
}

void ChemCompt::handleNodeInfo( const Eref& e, const Qinfo* q,
	unsigned int numNodes, unsigned int numThreads )
{
	// Pass it down to derived classes along with the SrcFinfo
	innerHandleNodeInfo( e, q, numNodes, numThreads );
}

void ChemCompt::resetStencil()
{
	this->innerResetStencil();
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

double ChemCompt::getEntireSize( const Eref& e, const Qinfo* q ) const
{
	return size_;
}

void ChemCompt::setEntireSize( const Eref& e, const Qinfo* q, double size )
{
	buildDefaultMesh( e, q, size, getNumEntries() );
}

unsigned int ChemCompt::getDimensions() const
{
	return this->innerGetDimensions();
}

string ChemCompt::getMethod() const
{
	return method_;
}

void ChemCompt::setMethod( string method )
{
	method_ = method;
}

//////////////////////////////////////////////////////////////
// Element Field Definitions
//////////////////////////////////////////////////////////////

MeshEntry* ChemCompt::lookupEntry( unsigned int index )
{
	return &entry_;
}

void ChemCompt::setNumEntries( unsigned int num )
{
	this->innerSetNumEntries( num );
	// cout << "Warning: ChemCompt::setNumEntries: No effect. Use subclass-specific functions\nto build or resize mesh.\n";
}

unsigned int ChemCompt::getNumEntries() const
{
	return this->innerGetNumEntries();
}

//////////////////////////////////////////////////////////////
// Element Field Definitions for boundary
//////////////////////////////////////////////////////////////

Boundary* ChemCompt::lookupBoundary( unsigned int index )
{
	if ( index < boundaries_.size() )
		return &( boundaries_[index] );
	cout << "Error: ChemCompt::lookupBoundary: Index " << index << 
		" >= vector size " << boundaries_.size() << endl;
	return 0;
}

void ChemCompt::setNumBoundary( unsigned int num )
{
	assert( num < 1000 ); // Pretty unlikely upper limit
	boundaries_.resize( num );
}

unsigned int ChemCompt::getNumBoundary() const
{
	return boundaries_.size();
}

//////////////////////////////////////////////////////////////
// Build the junction between this and another ChemCompt.
// This one function does the work for both meshes.
//////////////////////////////////////////////////////////////
void ChemCompt::buildJunction( ChemCompt* other, vector< VoxelJunction >& ret)
{
	matchMeshEntries( other, ret );
	extendStencil( other, ret );
	/*
	 * No longer having diffusion to abutting voxels in the follower
	 * compartment.
	 *
	flipRet( ret );
	other->extendStencil( this, ret );
	flipRet( ret );
	*/
}

void ChemCompt::flipRet( vector< VoxelJunction >& ret ) const
{
   vector< VoxelJunction >::iterator i;
   for ( i = ret.begin(); i != ret.end(); ++i ) {
		  unsigned int temp = i->first;
		  i->first = i->second;
		  i->second = temp;
   }
}

//////////////////////////////////////////////////////////////
// Orchestrate diffusion calculations in Stoich. This simply updates
// the flux terms (du/dt due to diffusion). Virtual func, has to be
// defined for every Mesh class if it differs from below.
// Called from the MeshEntry.
//////////////////////////////////////////////////////////////

void ChemCompt::lookupStoich( ObjId me ) const
{
	ChemCompt* cm = reinterpret_cast< ChemCompt* >( me.data() );
	assert( cm == this );
	vector< Id > stoichVec;
	unsigned int num = me.element()->getNeighbours( stoichVec, meshSplit());
	if ( num == 1 ) // The solver has been created
		cm->stoich_ = stoichVec[0];
}

/*
void ChemCompt::updateDiffusion( unsigned int meshIndex ) const
{
	// Later we'll have provision for multiple stoich targets.
	if ( stoich_ != Id() ) {
		Stoich* s = reinterpret_cast< Stoich* >( stoich_.eref().data() );
		s->updateDiffusion( meshIndex, stencil_ );
	}
}
*/

////////////////////////////////////////////////////////////////////////
// Utility function

double ChemCompt::distance( double x, double y, double z ) 
{
	return sqrt( x * x + y * y + z * z );
}
