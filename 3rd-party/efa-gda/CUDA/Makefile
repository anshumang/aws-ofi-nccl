NVCC=nvcc
NVCC_FLAGS=-G -rdc=true
CUDA_FLAGS=-Xcompiler -fPIC
INCLUDES=-Ihost -Idevice -Icommon

BUILDDIR=build

HOST_OBJECTS= \
	$(BUILDDIR)/efa_cuda_dp.o \
	$(BUILDDIR)/efa_cuda_dp_v1.o \
	$(BUILDDIR)/efa_cuda_dp_v2.o

HOST_HEADERS= \
	host/efa_cuda_dp.h \
	host/efa_cuda_dp_v1.h \
	host/efa_cuda_dp_v2.h

COMMON_HEADERS= \
	common/efa_cuda_dp_types.h \
	common/efa_cuda_dp_types_v1.h \
	common/efa_cuda_dp_types_v2.h

all: $(BUILDDIR)/libefacudadp.so

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/libefacudadp.so: $(HOST_OBJECTS) | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^ -lcuda

$(BUILDDIR)/%.o: host/%.cpp $(HOST_HEADERS) $(COMMON_HEADERS) device/efa_io_defs.h | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< $(CUDA_FLAGS) -o $@

clean:
	rm -rf $(BUILDDIR)
