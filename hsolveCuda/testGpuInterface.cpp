#include "cudaLibrary/GpuInterface.h"
#include "../shell/Shell.h"

#include "testGpuInterface.h"

/**
 * This program simply tests whether setup and unsetup are working correctly.
 */
void testGpuInterface()
{
#if 0
	HSolve *hsolve = new HSolve;
	hsolve->nCompt_ = 0;
	hsolve->stage_ = 0;

	// Setup
	GpuInterface gpu( hsolve );

	// Unsetup
	gpu.unsetup();
	ASSERT( hsolve->HS_ == gpu.hsolve_->HS, "Gpu Interface, HS" );
	ASSERT( hsolve->HJ_ == gpu.hsolve_->HJ, "Gpu Interface, HJ" );
	ASSERT( hsolve->HJCopy_ == gpu.hsolve_->HJCopy, "Gpu Interface, HJCopy" );
	ASSERT( hsolve->V_ == gpu.hsolve_->V, "Gpu Interface, V" );
	ASSERT( hsolve->VMid_ == gpu.hsolve_->VMid, "Gpu Interface, VMid" );
#endif

	cout << "\nTesting GpuInterface\n" << flush;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );

	vector< int* > childArray;
	vector< unsigned int > childArraySize;

	/**
	 *  We test passive-cable solver for the following cell:
	 *
	 *   Soma--->  15 - 14 - 13 - 12
	 *              |    |
	 *              |    L 11 - 10
	 *              |
	 *              L 16 - 17 - 18 - 19
	 *                      |
	 *                      L 9 - 8 - 7 - 6 - 5
	 *                      |         |
	 *                      |         L 4 - 3
	 *                      |
	 *                      L 2 - 1 - 0
	 *
	 *  The numbers are the hines indices of compartments. Compartment X is the
	 *  child of compartment Y if X is one level further away from the soma (#15)
	 *  than Y. So #17 is the parent of #'s 2, 9 and 18.
	 */

	int childArray_1[ ] =
	{
		/* c0  */  -1,
		/* c1  */  -1, 0,
		/* c2  */  -1, 1,
		/* c3  */  -1,
		/* c4  */  -1, 3,
		/* c5  */  -1,
		/* c6  */  -1, 5,
		/* c7  */  -1, 4, 6,
		/* c8  */  -1, 7,
		/* c9  */  -1, 8,
		/* c10 */  -1,
		/* c11 */  -1, 10,
		/* c12 */  -1,
		/* c13 */  -1, 12,
		/* c14 */  -1, 11, 13,
		/* c15 */  -1, 14, 16,
		/* c16 */  -1, 17,
		/* c17 */  -1, 2, 9, 18,
		/* c18 */  -1, 19,
		/* c19 */  -1,
	};

	childArray.push_back( childArray_1 );
	childArraySize.push_back( sizeof( childArray_1 ) / sizeof( int ) );

	/**
	 *  Cell 2:
	 *
	 *             3
	 *             |
	 *   Soma--->  2
	 *            / \
	 *           /   \
	 *          1     0
	 *
	 */

	int childArray_2[ ] =
	{
		/* c0  */  -1,
		/* c1  */  -1,
		/* c2  */  -1, 0, 1, 3,
		/* c3  */  -1,
	};

	childArray.push_back( childArray_2 );
	childArraySize.push_back( sizeof( childArray_2 ) / sizeof( int ) );

	/**
	 *  Cell 3:
	 *
	 *             3
	 *             |
	 *             2
	 *            / \
	 *           /   \
	 *          1     0  <--- Soma
	 *
	 */

	int childArray_3[ ] =
	{
		/* c0  */  -1, 2,
		/* c1  */  -1,
		/* c2  */  -1, 1, 3,
		/* c3  */  -1,
	};

	childArray.push_back( childArray_3 );
	childArraySize.push_back( sizeof( childArray_3 ) / sizeof( int ) );

	/**
	 *  Cell 4:
	 *
	 *             3  <--- Soma
	 *             |
	 *             2
	 *            / \
	 *           /   \
	 *          1     0
	 *
	 */

	int childArray_4[ ] =
	{
		/* c0  */  -1,
		/* c1  */  -1,
		/* c2  */  -1, 0, 1,
		/* c3  */  -1, 2,
	};

	childArray.push_back( childArray_4 );
	childArraySize.push_back( sizeof( childArray_4 ) / sizeof( int ) );

	/**
	 *  Cell 5:
	 *
	 *             1  <--- Soma
	 *             |
	 *             2
	 *            / \
	 *           4   0
	 *          / \
	 *         3   5
	 *
	 */

	int childArray_5[ ] =
	{
		/* c0  */  -1,
		/* c1  */  -1, 2,
		/* c2  */  -1, 0, 4,
		/* c3  */  -1,
		/* c4  */  -1, 3, 5,
		/* c5  */  -1,
	};

	childArray.push_back( childArray_5 );
	childArraySize.push_back( sizeof( childArray_5 ) / sizeof( int ) );

	/**
	 *  Cell 6:
	 *
	 *             3  <--- Soma
	 *             L 4
	 *               L 6
	 *               L 5
	 *               L 2
	 *               L 1
	 *               L 0
	 *
	 */

	int childArray_6[ ] =
	{
		/* c0  */  -1,
		/* c1  */  -1,
		/* c2  */  -1,
		/* c3  */  -1, 4,
		/* c4  */  -1, 0, 1, 2, 5, 6,
		/* c5  */  -1,
		/* c6  */  -1,
	};

	childArray.push_back( childArray_6 );
	childArraySize.push_back( sizeof( childArray_6 ) / sizeof( int ) );

	/**
	 *  Cell 7: Single compartment
	 */

	int childArray_7[ ] =
	{
		/* c0  */  -1,
	};

	childArray.push_back( childArray_7 );
	childArraySize.push_back( sizeof( childArray_7 ) / sizeof( int ) );

	/**
	 *  Cell 8: 3 compartments; soma is in the middle.
	 */

	int childArray_8[ ] =
	{
		/* c0  */  -1,
		/* c1  */  -1, 0, 2,
		/* c2  */  -1,
	};

	childArray.push_back( childArray_8 );
	childArraySize.push_back( sizeof( childArray_8 ) / sizeof( int ) );

	/**
	 *  Cell 9: 3 compartments; first compartment is soma.
	 */

	int childArray_9[ ] =
	{
		/* c0  */  -1, 1,
		/* c1  */  -1, 2,
		/* c2  */  -1,
	};

	childArray.push_back( childArray_9 );
	childArraySize.push_back( sizeof( childArray_9 ) / sizeof( int ) );

	////////////////////////////////////////////////////////////////////////////
	// Run tests
	////////////////////////////////////////////////////////////////////////////
	/*
	 * Solver instance.
	 */
	HSolve *hsolve = new HSolve;

	/*
	 * Model details.
	 */
	double dt = 1.0;
	vector< TreeNodeStruct > tree;
	vector< double > Em;
	vector< double > B;
	vector< double > V;
	vector< double > VMid;

	/*
	 * Loop over cells.
	 */
	int i;
	int j;
	//~ bool success;
	int nCompt;
	int* array;
	unsigned int arraySize;
	for ( int cell = childArray.size()-1; cell >= 0; cell-- ) {
		cout << "Cell number: " << cell << endl;

		array = childArray[ cell ];
		arraySize = childArraySize[ cell ];
		nCompt = count( array, array + arraySize, -1 );

		//////////////////////////////////////////
		// Prepare local information on cell
		//////////////////////////////////////////
		tree.clear();
		tree.resize( nCompt );
		Em.clear();
		V.clear();
		cout << "First for loop" << endl;
		for ( i = 0; i < nCompt; i++ ) {
			tree[ i ].Ra = 15.0 + 3.0 * i;
			tree[ i ].Rm = 45.0 + 15.0 * i;
			tree[ i ].Cm = 500.0 + 200.0 * i * i;
			Em.push_back( -0.06 );
			V.push_back( -0.06 + 0.01 * i );
		}

		int count = -1;
		cout << "Second for loop; arraysize=" << arraySize << endl;
		for ( unsigned int a = 0; a < arraySize; a++ ) {
			if ( array[ a ] == -1 )
				count++;
			else
				tree[ count ].children.push_back( array[ a ] );
		}

		//////////////////////////////////////////
		// Create cell inside moose; setup solver.
		//////////////////////////////////////////
		cout << "Creating id" << endl;
		Id n = shell->doCreate( "Neutral", Id(), "n" );

		vector< Id > c( nCompt );
		cout << "Third for loop" << endl;
		for ( i = 0; i < nCompt; i++ ) {
			ostringstream name;
			name << "c" << i;
			c[ i ] = shell->doCreate( "Compartment", n, name.str() );

			Field< double >::set( c[ i ], "Ra", tree[ i ].Ra );
			Field< double >::set( c[ i ], "Rm", tree[ i ].Rm );
			Field< double >::set( c[ i ], "Cm", tree[ i ].Cm );
			Field< double >::set( c[ i ], "Em", Em[ i ] );
			Field< double >::set( c[ i ], "initVm", V[ i ] );
			Field< double >::set( c[ i ], "Vm", V[ i ] );
		}

		cout << "Fourth for loop" << endl;
		for ( i = 0; i < nCompt; i++ ) {
			vector< unsigned int >& child = tree[ i ].children;
			for ( j = 0; j < ( int )( child.size() ); j++ ) {
				MsgId mid = shell->doAddMsg(
						"Single", c[ i ], "axial", c[ child[ j ] ], "raxial" );
				ASSERT( mid != Msg::bad, "Creating test model" );
			}
		}

		hsolve->HSolvePassive::setup( c[ 0 ], dt );
		HSolve hsolve_copy = *hsolve;

		cout << "Starting setup" << endl;
		GpuInterface gpu( &hsolve_copy );
		cout << "Starting unsetup" << endl;
		gpu.unsetup();
		cout << "Asserting..." << endl;
		ASSERT( hsolve_copy == *hsolve, "GpuInterface setup error" );

		// cleanup
		shell->doDelete( n );
	}
}
