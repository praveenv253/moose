CU_SRCS += \
	GpuInterface.cu \
	GpuKernels.cu \

CU_DEPS += \
	CudaModule.d \

OBJS += \
	GpuInterface.o \
	GpuKernels.o \

NVCC_FLAGS = --compile -G -g -O0 -Xcompiler -fPIC -x cu -arch=sm_20

all: libmooseCudaLibrary.so

libmooseCudaLibrary.so: $(OBJS) $(HEADERS)
	nvcc -shared -link -o "libmooseCudaLibrary.so" $(OBJS)

GpuInterface.o: GpuInterface.cu GpuInterface.h GpuKernels.h
	nvcc $(NVCC_FLAGS) -o GpuInterface.o GpuInterface.cu

GpuKernels.o: GpuKernels.cu GpuKernels.h ../HSolveStruct.h ../HinesMatrix.h
	nvcc $(NVCC_FLAGS) -o GpuKernels.o GpuKernels.cu

#%.o: %.cu
	#nvcc -G -g -O0 -Xcompiler -fPIC -odir "" -M -o "$(@:%.o=%.d)" "$<"
	#nvcc --compile -G -O0 -Xcompiler -fPIC -g -x cu -o  "$@" "$<"

clean:
	rm -rf $(CU_DEPS) $(OBJS) libmooseCudaLibrary.so
