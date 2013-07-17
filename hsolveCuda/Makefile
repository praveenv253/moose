#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = _hsolveCuda.o

SUBDIR = \
	cudaLibrary \


OBJ = \
	HSolveStruct.o \
	HinesMatrix.o \
	HSolvePassive.o \
	RateLookup.o \
	HSolveActive.o \
	HSolveActiveSetup.o \
	HSolveInterface.o \
	HSolve.o \
	HSolveUtils.o \
	testHSolve.o \
	ZombieCompartment.o \
	ZombieCaConc.o \
	ZombieHHChannel.o \

HEADERS = \
	../basecode/header.h

default: $(TARGET)

$(OBJ)	: $(HEADERS)
HSolveStruct.o:	HSolveStruct.h
HinesMatrix.o:	HinesMatrix.h TestHSolve.h
HSolvePassive.o:	HSolvePassive.h HinesMatrix.h HSolveStruct.h HSolveUtils.h TestHSolve.h ../biophysics/Compartment.h
RateLookup.o:	RateLookup.h
HSolveActive.o:	HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h
HSolveActiveSetup.o:	HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h HSolveUtils.h ../biophysics/HHChannel.h ../biophysics/ChanBase.h ../biophysics/HHGate.h ../biophysics/CaConc.h
HSolveInterface.o:	HSolve.h HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h
HSolve.o:	../biophysics/Compartment.h ZombieCompartment.h ../biophysics/CaConc.h ZombieCaConc.h ../biophysics/HHGate.h ../biophysics/ChanBase.h ../biophysics/HHChannel.h ZombieHHChannel.h HSolve.h HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h ../basecode/ElementValueFinfo.h
ZombieCompartment.o:	ZombieCompartment.h ../randnum/randnum.h ../biophysics/Compartment.h HSolve.h HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h ../basecode/ElementValueFinfo.h
ZombieCaConc.o:	ZombieCaConc.h ../biophysics/CaConc.h HSolve.h HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h ../basecode/ElementValueFinfo.h
ZombieHHChannel.o:	ZombieHHChannel.h ../biophysics/HHChannel.h ../biophysics/ChanBase.h ../biophysics/HHGate.h HSolve.h HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h ../basecode/ElementValueFinfo.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I/usr/local/cuda/include -I. -I../basecode -I../msg $< -c

$(TARGET): subdirs $(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) $(GSL_LIBS)

subdirs:
	@(for i in $(SUBDIR) $(PARALLEL_DIR); do $(MAKE) -C $$i; done)

clean:
	rm -f *.o $(TARGET) core core.*

BASECODE_OBJ = \
	../basecode/consts.o	\
	../basecode/Element.o	\
	../basecode/Qinfo.o \
	../basecode/DataHandler.o \
	../basecode/ZeroDimHandler.o \
	../basecode/ZeroDimParallelHandler.o \
	../basecode/BlockHandler.o \
	../basecode/OneDimHandler.o \
	../basecode/TwoDimHandler.o \
	../basecode/AnyDimHandler.o \
	../basecode/DataHandlerWrapper.o \
	../basecode/FieldDataHandlerBase.o \
	../basecode/MsgDataHandler.o \
	../basecode/ZombieHandler.o \
	../basecode/Eref.o	\
	../basecode/Finfo.o	\
	../basecode/DestFinfo.o	\
	../basecode/Cinfo.o	\
	../basecode/SrcFinfo.o \
	../basecode/ValueFinfo.o \
	../basecode/SharedFinfo.o \
	../basecode/FieldElementFinfo.o \
	../basecode/Id.o \
	../basecode/ObjId.o \
	../basecode/DataId.o \
	../basecode/SetGet.o \
	../basecode/OpFunc.o \
	../basecode/FieldOpFunc.o \
	../basecode/SparseMatrix.o \
	../basecode/PrepackedBuffer.o \
	../basecode/FuncBarrier.o \
	../basecode/doubleEq.o \
	../basecode/ReduceBase.o \
	../basecode/testAsync.o	\

test: _hsolveCuda.o testGpuInterface.cpp testGpuInterface.h ../shell/_shell.o ../basecode/_basecode.o cudaLibrary/GpuInterface.o
	g++ -c testGpuInterface.cpp
	g++ ../shell/_shell.o $(BASECODE_OBJ) ../biophysics/_biophysics.o ../msg/_msg.o ../builtins/_builtins.o ../external/muparser/_muparser.o ../scheduling/_scheduling.o -lhdf5 -lhdf5_hl ../randnum/_randnum.o ../kinetics/_kinetics.o ../manager/_manager.o ../utility/_utility.o ../mesh/_mesh.o -lgsl -lgslcblas _hsolveCuda.o cudaLibrary/GpuInterface.o testGpuInterface.o -lcuda -L/usr/local/cuda-5.0/lib64 -lcudart -LcudaLibrary -lmooseCudaLibrary -o testGpuInterface
