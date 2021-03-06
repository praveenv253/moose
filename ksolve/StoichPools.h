/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _STOICH_POOLS_H
#define _STOICH_POOLS_H

#ifndef _CHEM_COMPT_H
class ChemCompt;
#endif
class StoichPools
{
	public: 
		StoichPools();
		virtual ~StoichPools();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////
		// Model traversal and building functions
		//////////////////////////////////////////////////////////////////
		/// Using the computed array sizes, now allocate space for them.
		void resizeArrays( unsigned int totNumPools );

		/**
		 * Assign spatial grid of pools
		 */
		void meshSplit( 
						vector< double > initConc,
						vector< double > vols,
						vector< unsigned int > localEntryList );


		/**
		 * Returns the array of doubles of current mol #s at the specified
		 * mesh index
		 */
		const double* S( unsigned int meshIndex ) const;

		/**
		 * Returns the array of doubles of current mol #s at the specified
		 * mesh index. Dangerous, allows one to modify the values.
		 */
		double* varS( unsigned int meshIndex );

		/**
		 * Returns the array of doubles of initial mol #s at the specified
		 * mesh index
		 */
		const double* Sinit( unsigned int meshIndex ) const;

		/**
		 * Returns the array of doubles of initial mol #s at the specified
		 * mesh index, as a writable array.
		 */
		double* varSinit( unsigned int meshIndex );

		/**
		 * Returns size of S and Sinit vectors for local calculations
		 */
		unsigned int numMeshEntries() const;

		/**
		 * Returns size of S and Sinit vectors including boundaries.
		 */
		unsigned int numAllMeshEntries() const;

		/**
		 * Returns # of pools in S[meshEntry]. 0 if empty or out of range.
		 */
		unsigned int numPoolEntries( unsigned int meshEntry ) const;

		
		/**
		 * Looks for the pools participating in cross-solver reactions,
		 * where the reactions are on the current Solver, and the pool is
		 * on the other solver. As a side-effect, sets up the local
		 * PoolIndices of this set of pools as the remoteReacPools on the
		 * current junction.
		 */
		virtual void findPoolsOnOther( Id other, vector< Id >& pools ) = 0;

		/**
		 * Takes the provided list of pools which go to a reaction on
		 * the other solver participating in the latest junction. Figures
		 * out poolIndices and assigns to junction.
		 */
		virtual void setLocalCrossReactingPools( 
						const vector< Id >& pools ) = 0;

		//////////////////////////////////////////////////////////////////
		// Field assignment functions
		//////////////////////////////////////////////////////////////////

		virtual void setN( const Eref& e, double v ) = 0;
		virtual double getN( const Eref& e ) const = 0;
		virtual void setNinit( const Eref& e, double v ) = 0;
		virtual double getNinit( const Eref& e ) const = 0;
		virtual void setSpecies( const Eref& e, unsigned int v ) = 0;
		virtual unsigned int getSpecies( const Eref& e ) = 0;
		virtual void setDiffConst( const Eref& e, double v ) = 0;
		virtual double getDiffConst( const Eref& e ) const = 0;

		//////////////////////////////////////////////////////////////////
		// Junction FieldElement access functions
		//////////////////////////////////////////////////////////////////
		
		/// Returns pointer to specified Junction
		SolverJunction* getJunction( unsigned int i );
		
		/// Returns number of junctions.
		unsigned int getNumJunctions() const;

		/// Dummy function, we have to manipulate the junctions using the
		// add/drop junction functions.
		void setNumJunctions( unsigned int v );
		//////////////////////////////////////////////////////////////////
		// Cross-solver computation functions
		//////////////////////////////////////////////////////////////////
		//
		/// Handles arriving messages through junction
		void handleJunctionPoolDelta( unsigned int poolIndex, 
						vector< double > delta );
		void handleJunctionPoolNum( unsigned int poolIndex, 
						vector< double > num );

		/// Create a junction between self and specified other StoichPool
		void addJunction( const Eref& e, const Qinfo* q, Id other );
		/// Remove the junction between self and specified other StoichPool
		void dropJunction( const Eref& e, const Qinfo* q, Id other );

		void innerConnectJunctions( 
						Id me, Id other, StoichPools* otherSP );


		virtual void expandSforDiffusion( 
			const vector< unsigned int > & otherMeshIndex,
			const vector< unsigned int > & selfDiffPoolIndex,
			SolverJunction& j );

		void findDiffusionTerms(
				const StoichPools* otherSP,
				vector< unsigned int >& selfTerms,
				vector< unsigned int >& otherTerms
			) const;

		//////////////////////////////////////////////////////////////////
		// Matching virtual functions
		//////////////////////////////////////////////////////////////////
		/// Sends messages through junction. Called during Process.
		virtual void vUpdateJunction( const Eref& e, 
				const vector< vector< double > >& lastS,
				unsigned int threadNum, double dt ) = 0;

		/// Handles arriving messages through junction. Callsed 
		virtual void vHandleJunctionPoolDelta( unsigned int poolIndex, 
						const vector< double >& v ) = 0;
		virtual void vHandleJunctionPoolNum( unsigned int poolIndex, 
						const vector< double >& v ) = 0;

		/// Create a junction between self and specified other StoichPool
		virtual void vAddJunction( const Eref& e, const Qinfo* q, Id other ) = 0;
		/// Remove the junction between self and specified other StoichPool
		virtual void vDropJunction( const Eref& e, const Qinfo* q, Id other ) = 0;

		/**
		 * Generate the map of varPools that diffuse. 
		 */
		virtual void vBuildDiffTerms( map< string, unsigned int >& 
						diffTerms ) const = 0;

		/**
		 * Works out which meshEntries talk to each other. The outcome
		 * is reported as 
		 * - MeshIndex vectors for self and other, to specify which entries
		 *   to use
		 * - MeshMap vectors for self and other, to map from the index
		 *   in the data transfer vector between solvers, to the meshIndex
		 *   on the solver.
		 * This function is expected to refer to a ChemCompt in order to
		 * figure out the details. Even derived classes of StoichPools
		 * that do not use a mesh (e.g., Smoldyn) will use a dummy 
		 * ChemCompt to do the needful.
		 */
		virtual void matchMeshEntries( StoichPools* other,
			vector< unsigned int >& selfMeshIndex, 
			vector< VoxelJunction >& selfMeshMap,
			vector< unsigned int >& otherMeshIndex, 
			vector< VoxelJunction >& otherMeshMap
		) const = 0;

		virtual ChemCompt* compartmentMesh() const = 0;

		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
		//////////////////////////////////////////////////////////////////
	private:
		/**
		 * 
		 * S_ is the array of molecules. Stored as n, number of molecules
		 * per mesh entry. 
		 * The array looks like n = S_[meshIndex][poolIndex]
		 * The meshIndex specifies which spatial mesh entry to use.
		 * The poolIndex specifies which molecular species pool to use.
		 * We choose the poolIndex as the right-hand index because we need
		 * to be able to pass the entire block of pools around for 
		 * integration.
		 *
		 * The entire S_ vector is allocated, but the pools are only 
		 * allocated for local meshEntries and for pools on
		 * diffusive boundaries with other nodes.
		 *
		 * The first numVarPools_ in the poolIndex are state variables and
		 * are integrated using the ODE solver. 
		 * The last numEfflux_ molecules within numVarPools are those that
		 * go out to another solver. They are also integrated by the ODE
		 * solver, so that at the end of dt each has exactly as many
		 * molecules as diffused away.
		 * The next numBufPools_ are fixed but can be changed by the script.
		 * The next numFuncPools_ are computed using arbitrary functions of
		 * any of the molecule levels, and the time.
		 * The functions evaluate _before_ the ODE. 
		 * The functions should not cascade as there is no guarantee of
		 * execution order.
		 */
		vector< vector< double > > S_;

		/**
		 * Sinit_ specifies initial conditions at t = 0. Whenever the reac
		 * system is rebuilt or reinited, all S_ values become set to Sinit.
		 * Also used for buffered molecules as the fixed values of these
		 * molecules.
		 * The array looks like Sinit_[meshIndex][poolIndex]
		 * The entire Sinit_ vector is allocated, but the pools are only 
		 * allocated for local meshEntries and for pools on
		 * diffusive boundaries with other nodes.
		 */
		vector< vector< double > > Sinit_;

		/**
		 * vector of indices of meshEntries to be computed locally.
		 * This may not necessarily be a contiguous set, depending on
		 * how boundaries and voxelization is done.
		 * globalMeshIndex = localMeshEntries_[localIndex]
		 */
		vector< unsigned int > localMeshEntries_;

		/**
		 * Vector of Junctions between solvers. These specify how solvers
		 * communicate between each other in cases of diffusion, motors,
		 * or reactions. Accessed through FieldElements.
		 */
		vector< SolverJunction > junctions_;
};

#endif	// _STOICH_POOLS_H
