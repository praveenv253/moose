CU_SRCS += CudaModule.cu 

CU_DEPS += CudaModule.d 

OBJS += CudaModule.o 

%.o: %.cu
	nvcc -G -g -O0 -Xcompiler -fPIC -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -Xcompiler -fPIC -g -x cu -o  "$@" "$<"
	
all: libmooseCudaLibrary.so


libmooseCudaLibrary.so: $(OBJS) 
	nvcc -shared -link -o  "libmooseCudaLibrary.so" $(OBJS) 

clean:
	rm -rf $(CU_DEPS)$(OBJS) libmooseCudaLibrary.so
